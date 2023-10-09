import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def init_weights(net, init_type='kaiming'):
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, se=False):
        super().__init__()

        self.dconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.se = se
        if se:
            self.se_layer = SELayer(out_channel)

    def forward(self, x):
        x = self.dconv(x)
        if self.se:
            x = self.se_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channel_h, channel_l):
        super().__init__()
        self.d_conv = DoubleConv(channel_l * 2, channel_l)
        self.upconv = nn.Conv2d(channel_h, channel_l, 1)

    def forward(self, x_h, x_l):
        x_h_up = F.interpolate(x_h, size=x_l.size(
        )[-2:], mode="bilinear", align_corners=True)
        x_h_up = self.upconv(x_h_up)
        x = torch.cat((x_h_up, x_l), dim=1)

        return self.d_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, n_class, channel_reduction=2, aux=False):
        super().__init__()
        self.aux = aux
        channels = [64, 128, 256, 512, 1024]
        channels = [int(c / channel_reduction) for c in channels]

        self.donv1 = DoubleConv(in_channels, channels[0])
        self.donv2 = DoubleConv(channels[0], channels[1])
        self.donv3 = DoubleConv(channels[1], channels[2])
        self.donv4 = DoubleConv(channels[2], channels[3])
        self.donv_mid = DoubleConv(channels[3], channels[4])
        self.down_pool = nn.MaxPool2d(kernel_size=2)

        self.donv5 = Decoder(channels[4], channels[3])
        self.donv6 = Decoder(channels[3], channels[2])
        self.donv7 = Decoder(channels[2], channels[1])
        self.donv8 = Decoder(channels[1], channels[0])

        self.out_conv = nn.Conv2d(channels[0], n_class, kernel_size=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.ConvTranspose2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):

        x1 = self.donv1(x)  # 256
        x2 = self.donv2(self.down_pool(x1))  # 128
        x3 = self.donv3(self.down_pool(x2))  # 64
        x4 = self.donv4(self.down_pool(x3))  # 32
        x_mid = self.donv_mid(self.down_pool(x4))  # 16
        x = self.donv5(x_mid, x4)
        x = self.donv6(x, x3)
        x = self.donv7(x, x2)
        x = self.donv8(x, x1)
        x = self.out_conv(x)

        return x


if __name__ == "__main__":
    a = torch.randn([1, 1, 256, 256])
    net = UNet(1, 3)
    out = net(a)
    print(out.shape)
