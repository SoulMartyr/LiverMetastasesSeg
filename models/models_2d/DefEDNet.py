import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return DefC(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class DefC(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None):
        super(DefC, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = SeparableConv2d(
            inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = SeparableConv2d(
            inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        # nn.init.constant_(self.p_conv.weight, 0)
        # self.p_conv.register_backward_hook(self._set_lr)

    def forward(self, x):
        offset = self.p_conv(x)

        dtype = offset.data.type()
        device = offset.device
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype, device)

        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(
            2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(
            2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1),
                      torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * \
            (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))

        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * \
            (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * \
            (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * \
            (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
            g_rb.unsqueeze(dim=1) * x_q_rb + \
            g_lb.unsqueeze(dim=1) * x_q_lb + \
            g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        # if self.modulation:
        #     m = m.contiguous().permute(0, 2, 3, 1)
        #     m = m.unsqueeze(dim=1)
        #     m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
        #     x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype, device):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).to(device).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype, device):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).to(device).type(dtype)

        return p_0

    def _get_p(self, offset, dtype, device):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype, device)
        p_0 = self._get_p_0(h, w, N, dtype, device)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N]*padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -
                                                           1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(
            dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat(
            [x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.relu(out)

        return out


class def_resnet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(def_resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = DefC(3, self.inplanes, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _def_resnet(block, layers, **kwargs):
    model = def_resnet(block, layers, **kwargs)
    return model


def def_resnet34(**kwargs):

    return _def_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


class Ladder_ASPP(nn.Module):
    def __init__(self, channel):
        super(Ladder_ASPP, self).__init__()
        self.dilate1 = SeparableConv2d(
            channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = SeparableConv2d(
            channel*2, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = SeparableConv2d(
            channel*3, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate4 = SeparableConv2d(
            channel*4, channel, kernel_size=3, dilation=7, padding=7)
        self.bn = nn.BatchNorm2d(channel)
        self.drop = nn.Dropout2d(0.5)
        self.sg = nn.Sigmoid()

        self.finalchannel = channel

        self.conv1x1_1 = SeparableConv2d(
            channel*5, channel*3, kernel_size=1, dilation=1, padding=0)
        self.conv1x1_2 = SeparableConv2d(
            channel*3, channel*2, kernel_size=1, dilation=1, padding=0)

        # Master branch
        self.conv_master = SeparableConv2d(
            channel, channel, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channel)

        # Global pooling branch
        self.conv_gpb = SeparableConv2d(
            channel, channel, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channel)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x_gpb = self.avg_pool(x)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)
        x_gpb = self.sg(x_gpb)

        x_se = x_gpb * x

        # first block rate1
        d1 = self.dilate1(x)
        d1 = self.bn(d1)

        # second block rate3
        d2 = torch.cat([d1, x], 1)
        d2 = self.dilate2(d2)
        d2 = self.bn(d2)

        # third block rate5
        d3 = torch.cat([d1, d2, x], 1)
        d3 = self.dilate3(d3)
        d3 = self.bn(d3)

        # last block rate7
        d4 = torch.cat([d1, d2, d3, x], 1)
        d4 = self.dilate4(d4)
        d4 = self.bn(d4)

        out = torch.cat([d1, d2, d3, d4, x_se], 1)
        out = self.drop(out)
        out = self.conv1x1_1(out)
        out = self.conv1x1_2(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = SeparableConv2d(
            in_channels, in_channels//4, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = SeparableConv2d(
            in_channels // 4, n_filters, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DefEDNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super(DefEDNet, self).__init__()

        filters = [64, 128, 256, 512, 1024]

        def_resnet = def_resnet34()

        self.firstconv = DefC(in_channels, 64, 7,
                              stride=2, padding=3, bias=False)
        self.firstbn = nn.BatchNorm2d(64)
        self.firstrelu = nn.ReLU(inplace=True)
        self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.sconv = DefC(64, 64, 3, 2, 1)

        self.xe4 = DefC(256, 256, 3)
        self.xe3 = DefC(128, 128, 3, 2, 1)
        self.xe2 = DefC(64, 64, 3, 1, 1)

        self.encoder1 = def_resnet.layer1
        self.encoder2 = def_resnet.layer2
        self.encoder3 = def_resnet.layer3
        self.encoder4 = def_resnet.layer4

        self.ladder_aspp = Ladder_ASPP(512)

        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(768, 256)
        self.decoder2 = DecoderBlock(384, 128)
        self.decoder1 = DecoderBlock(192, 64)

        self.finalconv1 = SeparableConv2d(filters[0], 32, 3, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = SeparableConv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = SeparableConv2d(32, num_classes, 3, padding=1)

        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)   # 1,64,128,128
        x = self.firstbn(x)   # 1,64,128,128
        x = self.firstrelu(x)   # 1,64,128,128
        x_p = self.sconv(x)   # 1,64,128,128
        x_p = self.drop(x_p)   # 1,64,128,128

        e1 = self.encoder1(x_p)   # 1,64,128,128
        e1 = self.drop(e1)   # 1,64,128,128
        xe_2 = self.xe2(e1)   # 1,64,128,128

        e2 = self.encoder2(e1)   # 1,128,64,64
        e2 = self.drop(e2)
        xe_3 = self.xe3(e2)  # 1,128,64,64

        e3 = self.encoder3(e2)   # 1,256,32,32
        e3 = self.drop(e3)
        xe_4 = self.xe4(e3)   # 1,256,32,32

        e4 = self.encoder4(e3)   # 1,512,16,16
        e4 = self.drop(e4)

        # Center
        e4 = self.ladder_aspp(e4)   # 1,1024,16,16

        # Decoder
        d4 = self.decoder4(e4)   # 1,512,32,32
        d4 = self.drop(d4)
        d3 = self.decoder3(torch.cat([d4, xe_4], 1))  # 512 256
        d3 = self.drop(d3)  # 1,256,64,64
        d2 = self.decoder2(torch.cat([d3, xe_3], 1))  # 256 128
        d2 = self.drop(d2)  # 1,128,128,128
        d1 = self.decoder1(torch.cat([d2, xe_2], 1))  # 128 64
        d1 = self.drop(d1)

        out = self.finalconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


if __name__ == '__main__':
    model = DefEDNet(3, 1)
    img = torch.randn(4, 3, 224, 224)
    output = model(img)
    print(output.shape)
