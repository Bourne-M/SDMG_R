import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 norm_eval=False
                 ):
        super().__init__()
        assert len(strides) == num_stages, (
            'The length of strides should be equal to num_stages, '
            f'while the strides is {strides}, the length of '
            f'strides is {len(strides)}, and the num_stages is '
            f'{num_stages}.')
        assert len(enc_num_convs) == num_stages, (
            'The length of enc_num_convs should be equal to num_stages, '
            f'while the enc_num_convs is {enc_num_convs}, the length of '
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '
            f'{num_stages}.')
        assert len(dec_num_convs) == (num_stages - 1), (
            'The length of dec_num_convs should be equal to (num_stages-1), '
            f'while the dec_num_convs is {dec_num_convs}, the length of '
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '
            f'{num_stages}.')
        assert len(downsamples) == (num_stages - 1), (
            'The length of downsamples should be equal to (num_stages-1), '
            f'while the downsamples is {downsamples}, the length of '
            f'downsamples is {len(downsamples)}, and the num_stages is '
            f'{num_stages}.')
        assert len(enc_dilations) == num_stages, (
            'The length of enc_dilations should be equal to num_stages, '
            f'while the enc_dilations is {enc_dilations}, the length of '
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is '
            f'{num_stages}.')
        assert len(dec_dilations) == (num_stages - 1), (
            'The length of dec_dilations should be equal to (num_stages-1), '
            f'while the dec_dilations is {dec_dilations}, the length of '
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '
            f'{num_stages}.')
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2 ** i,
                        skip_channels=base_channels * 2 ** (i - 1),
                        out_channels=base_channels * 2 ** (i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        upsample_cfg='InterpConv' if upsample else None
                    ))

            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels * 2 ** i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i]))
            self.encoder.append((nn.Sequential(*enc_conv_block)))
            in_channels = base_channels * 2 ** i

    def forward(self, x):
        self._check_input_divisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        return dec_outs[-1]

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2


class UpConvBlock(nn.Module):
    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1, upsample_cfg='InterpConv'):
        super().__init__()

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation)
        if upsample_cfg is not None:
            self.upsample = InterpConv(
                in_channels=in_channels,
                out_channels=skip_channels, )
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0)

    def forward(self, skip, x):
        """Forward function."""
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)
        return out


class InterpConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 upsample_cfg=dict(scale_factor=2, mode='bilinear', align_corners=True)):
        super().__init__()
        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        upsample = nn.Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        out = self.interp_upsample(x)
        return out


class BasicConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1
                 ):
        super().__init__()
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation, ))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        out = self.convs(x)
        return out


class ConvModule(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 groups=1,
                 dilation=1,
                 bias=False,
                 inplace=True,
                 ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)
        self.init_weights()

    def init_weights(self):
        if isinstance(self.act, nn.LeakyReLU):
            nonlinearity = 'leaky_relu'
            a = 0.01
        else:
            nonlinearity = 'relu'
            a = 0

        if hasattr(self.conv, 'weight') and self.conv.weight is not None:
            nn.init.kaiming_normal_(self.conv.weight, a=a, nonlinearity=nonlinearity)
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

        if hasattr(self.bn, 'weight') and self.bn.weight is not None:
            nn.init.constant_(self.bn.weight, 1)
        if hasattr(self.bn, 'bias') and self.bn.bias is not None:
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
