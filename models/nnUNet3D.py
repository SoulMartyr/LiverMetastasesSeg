import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)

        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_trilinear: bool = False, is_z = True):
        super(UpConv, self).__init__()
        if is_z:
            kernel_size = 2
            stride = 2
        else:
            kernel_size = (1, 2, 2)
            stride = (1, 2, 2)
            
        if is_trilinear:
            self.up_sample = nn.Upsample(scale_factor=kernel_size, mode="trilinear", align_corners=True)
        else:
            self.up_sample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride)
        
        if in_channels != 320:
            self.conv = DoubleConv(in_channels + in_channels // 2, out_channels)
        else:
            self.conv = DoubleConv(in_channels + 256, out_channels)
            
    def forward(self, x1, x2):
        
        x1 = self.up_sample(x1)
        x = torch.cat([x1, x2], dim=1)
        out = self.conv(x)
        return out


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_z = True):
        super(DownConv, self).__init__()
        if is_z:
            self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        else:
            self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.max_pool(x)
        out = self.conv(x)
        return out


class nnUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3):
        super(nnUNet3D, self).__init__()

        self.stage1 = DoubleConv(in_channels, 32)
        self.stage2 = DownConv(32, 64, is_z=False)
        self.stage3 = DownConv(64, 128, is_z=False)
        self.stage4 = DownConv(128, 256)
        self.stage5 = DownConv(256, 320)
        
        self.stage6 = UpConv(320, 256)
        self.stage7 = UpConv(256, 128)
        self.stage8 = UpConv(128, 64, is_z=False)
        self.stage9 = UpConv(64, 32, is_z=False)

        self.projection = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        x6 = self.stage6(x5, x4)
        x7 = self.stage7(x6, x3)
        x8 = self.stage8(x7, x2)
        x9 = self.stage9(x8, x1)
        
        out = self.projection(x9)
        return out


if __name__ == "__main__":
    a = torch.ones((1, 1, 4, 256, 256))
    b = nnUNet3D(1, 2)
    print(b(a).shape)
