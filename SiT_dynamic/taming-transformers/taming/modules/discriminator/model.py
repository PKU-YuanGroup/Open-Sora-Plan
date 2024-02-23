import functools
import torch.nn as nn
from einops import rearrange

from taming.modules.util import ActNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 3
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)


        # norm_layer = nn.BatchNorm1d
        # kw = 3
        # padw = 1
        # sequence_1d = [nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        # nf_mult = 1
        # nf_mult_prev = 1
        # for n in range(1, n_layers):  # gradually increase the number of filters
        #     nf_mult_prev = nf_mult
        #     nf_mult = min(2 ** n, 8)
        #     sequence_1d += [
        #         nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
        #         norm_layer(ndf * nf_mult),
        #         nn.LeakyReLU(0.2, True)
        #     ]
        #
        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence_1d += [
        #     nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]
        #
        # sequence_1d += [
        #     nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        # self.main_1d = nn.Sequential(*sequence_1d)

    def forward(self, input):
        """Standard forward."""

        b, _, t, _, _ = input.shape
        # input_2d = rearrange(input, 'b c t h w -> (b t) c h w')
        # print('input', input.shape)
        out = self.main(input)
        # print('out', out.shape)
        #
        # _, _, _, w = input_2d.shape
        # input_1d = rearrange(input_2d, '(b t) c h w -> (b h w) c t', t=t)
        # print('input_1d', input_1d.shape)
        # out_1d = self.main_1d(input_1d)
        # print('out_1d', out_1d.shape)
        return out
