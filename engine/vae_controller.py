from typing import Optional

import pytorch_lightning
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS


class VAEController(pytorch_lightning.LightningModule):
    logger: MLFlowLogger

    def __init__(self, config):
        super(VAEController, self).__init__()
        self.config = config
        # model = self.config.model()
        # self.model_loss = self.config.loss(config, model)
        self.alpha = config.get('alpha', 4.)
        self.module = self.config.model()
        self.save_hyperparameters({i: repr(j) for i, j in config.items()})

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x = batch['x']
        reconstructed, c, mu, logvar = self.module(x)
        reconstructed = torch.sigmoid(reconstructed)
        loss = F.mse_loss(reconstructed, x, reduction='sum').div(x.shape[0])
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))
        kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean(0, True)
        final_loss = loss + self.alpha * kld
        if batch_idx % 100 == 0:
            self.log('Rec Train loss', loss.item())
            self.log('KLDiv Train loss', kld.item())
            self.log('Train loss', final_loss.item())
        return final_loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        x = batch['x']
        reconstructed, c, mu, logvar = self.module(x)
        reconstructed = torch.sigmoid(reconstructed)
        return torch.sum((x - reconstructed) ** 2, dim=(1, 2, 3))

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._evaluate(outputs)
        self.logger.experiment.log_artifacts(self.logger.run_id, self.config.output)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self._evaluate(outputs)

    def _evaluate(self, outputs: EPOCH_OUTPUT) -> None:
        for dataset_idx in range(len(outputs)):
            sq = torch.cat(outputs[dataset_idx], dim=0)
            self.log(f'Val {dataset_idx}', torch.mean(sq))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.config.train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.config.val_dataloader()

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.config.test_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.config.test_dataloader()

    def configure_optimizers(self):
        # return self.config.optimizer(self.model_loss)
        return self.config.optimizer(self.module)
