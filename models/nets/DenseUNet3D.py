import torch
import torch.nn as nn


class Scale3D(nn.Module):
    def __init__(self, channels: int, dim: int = 1):
        super(Scale3D, self).__init__()
        self.channels = channels
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones([1, channels, 1, 1, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, channels, 1, 1, 1]), requires_grad=True)

    def forward(self, x):
        x_shape = list(x.shape)

        assert self.channels == x_shape[self.dim], \
            "Channels should be equal to the number of input corresponding dimension "

        out = x * self.gamma + self.beta
        return out


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, dropout_rate: float = 0.):
        super(ConvBlock3D, self).__init__()
        inter_channels = growth_rate * 4

        self.dropout_rate = dropout_rate
        self.bn1 = nn.BatchNorm3d(in_channels, momentum=1)
        self.scale1 = Scale3D(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)

        if dropout_rate > 0:
            self.dropout1 = nn.Dropout(p=dropout_rate)

        self.bn2 = nn.BatchNorm3d(inter_channels, momentum=1)
        self.scale2 = Scale3D(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

        if dropout_rate:
            self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.bn1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv1(out)

        if self.dropout_rate > 0:
            out = self.dropout1(out)

        out = self.bn2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.dropout_rate > 0:
            out = self.dropout2(out)

        return out


class DenseBlock3D(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, dropout_rate: float = 0.):
        super(DenseBlock3D, self).__init__()

        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module("conv" + str(i), ConvBlock3D(in_channels, growth_rate))
            in_channels += growth_rate

    def forward(self, x):
        concat_out = x
        for layer in self.layers:
            out = layer(concat_out)
            concat_out = torch.cat([concat_out, out], dim=1)

        return concat_out


class TransitionBlock3D(nn.Module):
    def __init__(self, in_channels: int, compression: float = 1., dropout_rate: float = 0.):
        super(TransitionBlock3D, self).__init__()

        self.dropout_rate = dropout_rate

        self.bn = nn.BatchNorm3d(in_channels, momentum=1)
        self.scale = Scale3D(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, int(in_channels * compression), kernel_size=1, bias=False)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        out = self.bn(x)
        out = self.scale(out)
        out = self.relu(out)
        out = self.conv(out)

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.avg_pool(out)
        return out


class DenseUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, inter_channels: int = 96, growth_rate: int = 32,
                 num_dense_block: int = 4,
                 reduction: float = 0.,
                 dropout_rate: float = 0.):
        super(DenseUNet3D, self).__init__()

        num_blocks = [3, 4, 12, 8]
        compression = 1.0 - reduction

        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(inter_channels)
        self.scale1 = Scale3D(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.dense_blocks = nn.Sequential()
        for idx in range(num_dense_block):
            self.dense_blocks.add_module("dense_block_stage" + str(idx + 1),
                                         DenseBlock3D(num_blocks[idx], inter_channels, growth_rate, dropout_rate))
            inter_channels += num_blocks[idx] * growth_rate
            if idx != num_dense_block - 1:
                self.dense_blocks.add_module("transition_block_stage" + str(idx + 1),
                                             TransitionBlock3D(inter_channels, compression, dropout_rate))
                inter_channels = int(inter_channels * compression)

        self.bn2 = nn.BatchNorm3d(inter_channels)
        self.scale2 = Scale3D(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.up_stage1 = nn.Sequential(nn.Upsample(scale_factor=(1, 2, 2)),
                                       nn.Conv3d(inter_channels, 504, kernel_size=3, padding="same"),
                                       nn.BatchNorm3d(504, momentum=1),
                                       nn.ReLU(inplace=True))

        self.up_stage2 = nn.Sequential(nn.Upsample(scale_factor=(1, 2, 2)),
                                       nn.Conv3d(504, 224, kernel_size=3, padding="same"),
                                       nn.BatchNorm3d(224, momentum=1),
                                       nn.ReLU(inplace=True))

        self.up_stage3 = nn.Sequential(nn.Upsample(scale_factor=(1, 2, 2)),
                                       nn.Conv3d(224, 192, kernel_size=3, padding="same"),
                                       nn.BatchNorm3d(192, momentum=1),
                                       nn.ReLU(inplace=True))

        self.up_stage4 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2)),
                                       nn.Conv3d(192, 96, kernel_size=3, padding="same"),
                                       nn.BatchNorm3d(96, momentum=1),
                                       nn.ReLU(inplace=True))

        self.up_stage5 = nn.Sequential(nn.Upsample(scale_factor=(2, 2, 2)),
                                       nn.Conv3d(96, 64, kernel_size=3, padding="same"),
                                       nn.BatchNorm3d(64, momentum=1),
                                       nn.ReLU(inplace=True))
        self.projection = nn.Conv3d(64, out_channels, kernel_size=1, padding="same")

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.max_pool(out)

        out = self.dense_blocks(out)

        out = self.bn2(out)
        out = self.scale2(out)
        out = self.relu2(out)

        out = self.up_stage1(out)
        out = self.up_stage2(out)
        out = self.up_stage3(out)
        out = self.up_stage4(out)
        out = self.up_stage5(out)

        pro_out = self.projection(out)
        return out, pro_out


if __name__ == "__main__":
    a = torch.randn([2, 1, 12, 256, 256])
    model = DenseUNet3D(in_channels=1)
    _, res = model(a)
    print(res.shape)
