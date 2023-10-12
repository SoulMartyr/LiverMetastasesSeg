import torch
import torch.nn as nn

from .nets.DenseUNet2D import DenseUNet2D
from .nets.DenseUNet3D import DenseUNet3D


class HDenseUNet(nn.Module):
    def __init__(self, num_slices: int, out_channels: int = 3):
        super(HDenseUNet, self).__init__()
        self.num_slices = num_slices
        self.dense_unet2d = DenseUNet2D(3, out_channels, reduction=0.5)
        self.dense_unet3d = DenseUNet3D(out_channels + 1, out_channels, reduction=0.5)
        self.conv = nn.Conv3d(64, 64, kernel_size=3, padding="same")
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.projection = nn.Conv3d(64, out_channels, kernel_size=1, padding="same")

    def forward(self, x):
        slices = self.num_slices
        input2d = x[:, :, 0:2, :, :]
        single = x[:, :, 0:1, :, :]
        input2d = torch.cat([single, input2d], dim=2)
        for i in range(slices - 2):
            input2d_tmp = x[:, :, i:i + 3, :, :]
            input2d = torch.cat([input2d, input2d_tmp], dim=0)
            if i == slices - 3:
                final1 = x[:, :, slices - 2:slices, :, :]
                final2 = x[:, :, slices - 1:slices, :, :]
                final = torch.cat([final1, final2], dim=2)
                input2d = torch.cat([input2d, final], dim=0)

        input2d.squeeze_(1)

        feature2d, classifer2d = self.dense_unet2d(input2d)

        cla2d = classifer2d[0:1, :, :, :].permute(1, 0, 2, 3).unsqueeze(0)
        fea2d = feature2d[0:1, :, :, :].permute(1, 0, 2, 3).unsqueeze(0)

        for i in range(slices - 1):
            score = classifer2d[i + 1:i + 2, :, :, :].permute(1, 0, 2, 3).unsqueeze(0)
            fea2d_slice = feature2d[i + 1:i + 2, :, :, :].permute(1, 0, 2, 3).unsqueeze(0)
            cla2d = torch.cat([cla2d, score], dim=2)
            fea2d = torch.cat([fea2d, fea2d_slice], dim=2)
        
        input3d_ori = x[:, :, 0:slices, :, :]
        input3d = torch.cat([input3d_ori, cla2d], dim=1)

        feature3d, classifer3d = self.dense_unet3d(input3d)

        final = feature3d + fea2d
        final_conv = self.conv(final)
        # final_conv = self.dropout(final_conv)
        final_bn = self.bn(final_conv)
        final_relu = self.relu(final_bn)

        out = self.projection(final_relu)
        return out


class AmpHDenseUNet(HDenseUNet):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpHDenseUNet, self).forward(*args)


if __name__ == "__main__":
    a = torch.randn([1, 1, 8, 256, 256])
    model = HDenseUNet(num_slices=8, out_channels=2)
    res = model(a)
    print(res.shape)
