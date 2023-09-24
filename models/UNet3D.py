import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        if in_channels < out_channels:
            mid_channels = out_channels // 2
        else:
            mid_channels = in_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)

        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_trilinear: bool = True):
        super(UpConv, self).__init__()
        if is_trilinear:
            self.up_sample = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=True)
        else:
            self.up_sample = nn.ConvTranspose3d(
                in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_sample(x1)

        x = torch.cat([x1, x2], dim=1)
        out = self.conv(x)
        return out


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownConv, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.max_pool(x)
        out = self.conv(x)
        return out


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super(UNet3D, self).__init__()

        self.stage1 = DoubleConv(in_channels, 64)
        self.stage2 = DownConv(64, 128)
        self.stage3 = DownConv(128, 256)
        self.stage4 = DownConv(256, 512)

        self.stage5 = UpConv(512, 256)
        self.stage6 = UpConv(256, 128)
        self.stage7 = UpConv(128, 64)

        self.projection = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        x5 = self.stage5(x4, x3)
        x6 = self.stage6(x5, x2)
        x7 = self.stage7(x6, x1)

        out = self.projection(x7)
        return out


class AmpUNet3D(UNet3D):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpUNet3D, self).forward(*args)


if __name__ == "__main__":
    a = torch.randn((1, 1, 8, 256, 256))
    b = UNet3D(1, 2)
    print(b(a).shape)
