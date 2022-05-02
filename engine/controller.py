from typing import Optional, List

import numpy as np
import pytorch_lightning
import torch
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import autograd, Tensor


def compute_grad2(real_pred, x):
    batch_size = x.size(0)
    grad_dout = autograd.grad(
        outputs=real_pred.sum(), inputs=x,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class Controller(pytorch_lightning.LightningModule):
    logger: MLFlowLogger

    def __init__(self, config):
        super(Controller, self).__init__()
        self.config = config
        self.vae = self.config.get_vae()
        self.generator = self.config.get_generator()
        self.discriminator = self.config.get_discriminator()
        self.distribution = torch.distributions.Normal(0, 1)
        self.reg_param = self.config.get('reg_param', 10.)
        self.w_info = self.config.get('w_info', 0.001)
        self.automatic_optimization = False
        self.save_hyperparameters({i: repr(j) for i, j in config.items()})

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        optimizers: List[LightningOptimizer] = self.optimizers()
        x = batch['x'].requires_grad_()
        z = self.distribution.sample((x.shape[0], self.config.get('z_dim', 256))).to(x.device)

        # Discriminator
        optimizers[0].optimizer.zero_grad()
        real_pred = self.discriminator(x)
        dloss_real = F.binary_cross_entropy_with_logits(
            real_pred,
            real_pred.new_full(size=real_pred.size(), fill_value=1)
        )
        self.manual_backward(dloss_real, option=1, real_pred=real_pred, x=x)

        with torch.no_grad():
            c, c_mu, c_logvar = self.vae(x, encode_only=True)
            x_gen = self.generator(torch.cat([z, c], dim=1))
        x_gen.requires_grad_()
        gen_pred = self.discriminator(x_gen)
        dloss_gen = F.binary_cross_entropy_with_logits(
            gen_pred,
            gen_pred.new_full(size=gen_pred.size(), fill_value=0)
        )
        self.manual_backward(dloss_gen)
        optimizers[0].step()

        # Generator
        z = self.distribution.sample((x.shape[0], self.config.get('z_dim', 256))).to(x.device)
        self.vae.zero_grad()
        optimizers[1].optimizer.zero_grad()

        x_gen = self.generator(torch.cat([z, c], dim=1))
        gen_pred = self.discriminator(x_gen)
        gloss = F.binary_cross_entropy_with_logits(
            gen_pred,
            gen_pred.new_full(size=gen_pred.size(), fill_value=1)
        )

        ch, ch_mu, ch_logvar = self.vae(x_gen, encode_only=True)
        encloss = (np.log(2 * np.pi) + ch_logvar + (c - ch_mu).pow(2).div(ch_logvar.exp() + 1e-8)).div(2).sum(1).mean()
        loss = gloss + self.w_info * encloss
        self.manual_backward(loss)
        optimizers[1].step()

        if batch_idx % 100 == 0:
            self.log('Train discriminator loss', dloss_gen.item() + dloss_real.item())
            self.log('Train generator loss', gloss.item())
            self.log('Train encode loss', encloss.item())

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return None

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._evaluate(outputs)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._evaluate(outputs)

    def _evaluate(self, outputs: EPOCH_OUTPUT) -> None:
        return

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.config.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.config.val_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.config.test_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.config.test_dataloader()

    def configure_optimizers(self):
        return self.config.optimizer(self.discriminator, self.generator)

    def manual_backward(self, loss: Tensor, option=0, *args, **kwargs) -> None:
        if option == 0:
            loss.backward()
        elif option == 1:
            loss.backward(retain_graph=True)
            real_pred = kwargs['real_pred']
            x = kwargs['x']
            batch_size = x.size(0)
            grad_dout = autograd.grad(
                outputs=real_pred.sum(), inputs=x,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad_dout2 = grad_dout.pow(2)
            assert (grad_dout2.size() == x.size())
            reg = grad_dout2.view(batch_size, -1).sum(1)
            reg = self.reg_param * reg.mean()
            reg.backward()
