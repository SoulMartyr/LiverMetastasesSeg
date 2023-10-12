import torch
import torch.nn as nn

from .nets.Transformer import LearnedPositionalEncoding, TransformerLayer
from .nets.UNet_3D_ED import Unet3DEncoder, Unet3DDecoder


class TransBTS(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, img_dim: int = 128, patch_dim: int = 8,
                 embed_dim: int = 512, num_heads: int = 8, num_layers: int = 4, hidden_dim: int = 4096,
                 dropout_rate: float = 0.0):
        super(TransBTS, self).__init__()

        assert embed_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.dropout_rate = dropout_rate

        self.seq_length = 2048

        self.position_encoding = LearnedPositionalEncoding(self.embed_dim, self.seq_length)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerLayer(embed_dim, num_layers, num_heads, hidden_dim, self.dropout_rate)
        self.pre_head_ln = nn.LayerNorm(embed_dim)

        self.conv_x = nn.Conv3d(128, self.embed_dim, kernel_size=3, stride=1, padding=1)

        self.encoder = Unet3DEncoder(in_channels=in_channels, base_channels=16)
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)

        self.decoder = Unet3DDecoder(in_channels=embed_dim, out_channels=out_channels, img_dim=img_dim,
                                     patch_dim=patch_dim, embed_dim=embed_dim)

    def forward(self, x):
        # Encode
        x1, x2, x3, x4 = self.encoder(x)
        en_out = self.bn(x4)
        en_out = self.relu(en_out)
        en_out = self.conv_x(en_out)
        en_out = en_out.permute(0, 2, 3, 4, 1).contiguous()
        en_out = en_out.view(x.size(0), -1, self.embed_dim)

        # Transformer Projection
        trans_inp = self.position_encoding(en_out)
        trans_inp = self.pe_dropout(trans_inp)
        trans_out = self.transformer(trans_inp)
        trans_out = self.pre_head_ln(trans_out)
        # Decode
        de_out = self.decoder(x1, x2, x3, x4, trans_out)
        return de_out


class AmpTransBTS(TransBTS):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpTransBTS, self).forward(*args)


if __name__ == "__main__":

    a = torch.randn([2, 1, 16, 256, 256])
    model = TransBTS(in_channels=1, out_channels=3)
    out = model(a)
