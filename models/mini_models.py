import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(mean=0, std=0.02)
        if m.bias.data is not None:
            m.bias.data.zero_()


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std * eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Encoder(nn.Module):
    def __init__(self, c_dim=10, nc=3, infodistil_mode=False):
        super(Encoder, self).__init__()
        self.c_dim = c_dim
        self.nc = nc
        self.infodistil_mode = infodistil_mode
        self.layer = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, c_dim * 2),  # B, c_dim*2
        )

    def forward(self, x):
        if self.infodistil_mode:
            x = x.add(1).div(2)
            if (x.size(2) > 64) or (x.size(3) > 64):
                x = F.adaptive_avg_pool2d(x, (64, 64))

        h = self.layer(x)
        return h


class Decoder(nn.Module):
    def __init__(self, c_dim=10, nc=3):
        super(Decoder, self).__init__()
        self.c_dim = c_dim
        self.nc = nc
        self.layer = nn.Sequential(
            nn.Linear(c_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

    def forward(self, c):
        x = self.layer(c)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, size, nfilter=64, nfilter_max=512, **kwargs):
        super().__init__()
        self.z_dim = z_dim

        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        self.fc = nn.Linear(z_dim, self.nf0 * s0 * s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        out = self.fc(z)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(out)
        out = torch.tanh(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, z_dim, size, nfilter=64, nfilter_max=512):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv1 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.fhidden)
        self.nonlin1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.fout)
        self.nonlin2 = nn.LeakyReLU()
        self.bypass = fin == fout

    def forward(self, x):
        dx = self.conv1(x)
        dx = self.bn1(dx)
        dx = self.nonlin1(dx)
        dx = self.conv2(dx)
        dx = self.bn2(dx)
        out = self.nonlin2(dx)
        if self.bypass:
            out = x + out

        return out


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, c_dim=10, channels=3, infodistil_mode=False):
        super(BetaVAE_H, self).__init__()
        self.c_dim = c_dim
        self.channels = channels
        self.encoder = Encoder(c_dim, channels, infodistil_mode)
        self.decoder = Decoder(c_dim, channels)
        self.apply(normal_init)

    def forward(self, x=None, c=None, encode_only=False, decode_only=False):
        assert int(encode_only) + int(decode_only) != 2
        if encode_only:
            c, mu, logvar = self._encode(x)
            return c, mu, logvar
        elif decode_only:
            x_recon = self._decode(c)
            return x_recon
        else:
            c, mu, logvar = self._encode(x)
            x_recon = self._decode(c)
            return x_recon, c, mu, logvar

    def _encode(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.c_dim]
        logvar = distributions[:, self.c_dim:]
        c = reparametrize(mu, logvar)
        return c, mu, logvar

    def _decode(self, c):
        return self.decoder(c)
