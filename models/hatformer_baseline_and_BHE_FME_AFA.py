import torch
import torch.nn as nn
import torch.nn.functional
from functools import partial
from models._utils.ChangeFormerBaseNetworks import *
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from models.CM2_decoder_BHE_FME_AFA import CM2_decoder
from models.CGBlock import cgblock

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel, norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

class AuxFeaAggregation(nn.Module):
    def __init__(self, in_ch, filter, dilation, reduction, chs):
        super(AuxFeaAggregation, self).__init__()
        self.ccorr_block = decode(chs, chs, chs)
        self.context_block = cgblock(in_ch+64, filter, dilation, reduction)

    def forward(self, x, y, z):
        x = torch.cat((self.ccorr_block(x, y), z),dim=1)
        x = self.context_block(x)
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, num_classes=2, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # main  encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        # intra-patch encoder
        self.patch_block1 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(1)])
        self.pnorm1 = norm_layer(embed_dims[1])
        # main  encoder
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        # intra-patch encoder
        self.patch_block2 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(1)])
        self.pnorm2 = norm_layer(embed_dims[2])
        # main  encoder
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        # intra-patch encoder
        self.patch_block3 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[1], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(1)])
        self.pnorm3 = norm_layer(embed_dims[3])
        # main  encoder
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        embed_dims = [64, 128, 320, 512]
        # stage 1
        x1, H1, W1 = self.patch_embed1(x)

        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[1], H1, W1)

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[2], H1, W1)

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        x1 = x1.permute(0, 2, 1).reshape(B, embed_dims[3], H1, W1)  # +x2

        x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # pdb.set_trace()
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_head=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 ):
        super().__init__()
        assert dim % num_head == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_head
        head_dim = dim // num_head
        self.scale = qk_scale or head_dim ** -0.5

        self.q_I = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_H = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_I = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_H = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop_IH = nn.Dropout(attn_drop)
        self.attn_drop_HI = nn.Dropout(attn_drop)
        self.proj_IH = nn.Linear(dim, dim)
        self.proj_HI = nn.Linear(dim, dim)
        self.proj_drop_IH = nn.Dropout(proj_drop)
        self.proj_drop_HI = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xI_emb, xH_emb, H, W):

        # get q,k,v from multi-branch embeddings
        B, N, C = xI_emb.shape
        qI = self.q_I(xI_emb).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        qH = self.q_H(xH_emb).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            xI_emb_ = xI_emb.permute(0, 2, 1).reshape(B, C, H, W)
            xI_emb_ = self.sr(xI_emb_).reshape(B, C, -1).permute(0, 2, 1)
            xI_emb_ = self.norm(xI_emb_)
            kv_I = self.kv_I(xI_emb_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            xH_emb_ = xH_emb.permute(0, 2, 1).reshape(B, C, H, W)
            xH_emb_ = self.sr(xH_emb_).reshape(B, C, -1).permute(0, 2, 1)
            xH_emb_ = self.norm(xH_emb_)
            kv_H = self.kv_H(xH_emb_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv_I = self.kv_I(xI_emb).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_H = self.kv_H(xH_emb).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        kI, vI = kv_I[0], kv_I[1]
        kH, vH = kv_H[0], kv_H[1]

        # cross-attention
        attn_IH = (qI @ kH.transpose(-2, -1)) * self.scale
        attn_IH = attn_IH.softmax(dim=-1)
        attn_IH = self.attn_drop_IH(attn_IH)

        x_IH = (attn_IH @ vH).transpose(1, 2).reshape(B, N, C)
        x_IH = self.proj_IH(x_IH)
        x_IH = self.proj_drop_IH(x_IH)

        attn_HI = (qH @ kI.transpose(-2, -1)) * self.scale
        attn_HI = attn_HI.softmax(dim=-1)
        attn_HI = self.attn_drop_HI(attn_HI)

        x_HI = (attn_HI @ vI).transpose(1, 2).reshape(B, N, C)
        x_HI = self.proj_HI(x_HI)
        x_HI = self.proj_drop_HI(x_HI)

        # return attention scores

        return x_IH, x_HI


class crossAttBlock(nn.Module):

    def __init__(self, dim, num_head=None, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=1,
                 in_chans=None):
        super().__init__()
        self.norm1_I = norm_layer(dim)
        self.norm1_H = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_I = norm_layer(dim)
        self.norm2_H = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_I = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_H = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3_I = norm_layer(dim)
        self.norm3_H = norm_layer(dim)

        self.linear = MLP(input_dim=dim, embed_dim=64)

        self.patch_embed_I = OverlapPatchEmbed(img_size=1024, patch_size=3, stride=1, in_chans=in_chans,
                                               embed_dim=dim)
        self.patch_embed_H = OverlapPatchEmbed(img_size=1024, patch_size=3, stride=1, in_chans=in_chans,
                                               embed_dim=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xI, xH):
        # get embedding from multi-branch features
        # xI,xH: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = xI.shape
        xI, HI, WI = self.patch_embed_I(xI)
        xH, HH, WH = self.patch_embed_H(xH)
        xI_att, xH_att = self.attn(self.norm1_I(xI), self.norm1_H(xH), H, W)
        xI = xI + self.drop_path(xI_att)
        # xI = xI + self.drop_path(self.mlp_I(self.norm2_I(xI), H, W))

        xH = xH + self.drop_path(xH_att)
        # xH = xH + self.drop_path(self.mlp_H(self.norm2_H(xH), H, W))

        xI = self.norm3_I(xI)
        xI = xI.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xI = self.linear(xI).permute(0, 2, 1).reshape(B, -1, H, W)

        xH = self.norm3_H(xH)
        xH = xH.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xH = self.linear(xH).permute(0, 2, 1).reshape(B, -1, H, W)

        return xI, xH


class Attention_dec(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = nn.Parameter(torch.randn(1, 48, dim))
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        B, N, C = x.shape
        task_q = self.task_query

        # This is because we fix the task parameters to be of a certain dimension, so with varying batch size, we just stack up the same queries to operate on the entire batch
        if B > 1:
            task_q = task_q.unsqueeze(0).repeat(B, 1, 1, 1)
            task_q = task_q.squeeze(1)

        q = self.q(task_q).reshape(B, task_q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q, size=(v.shape[2], v.shape[3]))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block_dec(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Tenc(EncoderTransformer):
    def __init__(self, **kwargs):
        super(Tenc, self).__init__(
            patch_size=16, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class convprojection(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection, self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential(ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)
        self.conv_output = ConvLayer(8, 2, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()

    def forward(self, x1, x2):

        res32x = self.convd32x(x2[0])

        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0, -1, 0, -1)
            res32x = F.pad(res32x, p2d, "constant", 0)

        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0, -1, 0, 0)
            res32x = F.pad(res32x, p2d, "constant", 0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0, 0, 0, -1)
            res32x = F.pad(res32x, p2d, "constant", 0)

        res16x = res32x + x1[3]
        res16x = self.convd16x(res16x)

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, -1, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0, -1, 0, 0)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, 0, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x)
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x


class convprojection_base(nn.Module):
    def __init__(self, path=None, **kwargs):
        super(convprojection_base, self).__init__()

        # self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential(ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)

    def forward(self, x1):

        #         if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
        #             p2d = (0,-1,0,-1)
        #             res32x = F.pad(res32x,p2d,"constant",0)

        #         elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
        #             p2d = (0,-1,0,0)
        #             res32x = F.pad(res32x,p2d,"constant",0)
        #         elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
        #             p2d = (0,0,0,-1)
        #             res32x = F.pad(res32x,p2d,"constant",0)

        #         res16x = res32x + x1[3]
        res16x = self.convd16x(x1[3])

        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, -1, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0, -1, 0, 0)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, 0, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)

        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x)
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)
        return x


# Transformer Decoder
class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )


# Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


# Transormer Ecoder with x2, x4, x8, x16 scales
class EncoderTransformer_v3(nn.Module):
    def __init__(self, img_size=512, patch_size=3, in_chans=3, num_classes=2, embed_dims=[32, 64, 128, 256],
                 num_heads=[2, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 3, 6, 18], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch embedding definitions
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=patch_size, stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Stage-1 (x1/4 scale)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        # Stage-2 (x1/8 scale)
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        # Stage-3 (x1/16 scale)
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        # Stage-4 (x1/32 scale)
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x1, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x1 = blk(x1, H1, W1)
        x1 = self.norm1(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 2
        x1, H1, W1 = self.patch_embed2(x1)
        for i, blk in enumerate(self.block2):
            x1 = blk(x1, H1, W1)
        x1 = self.norm2(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 3
        x1, H1, W1 = self.patch_embed3(x1)
        for i, blk in enumerate(self.block3):
            x1 = blk(x1, H1, W1)
        x1 = self.norm3(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)

        # stage 4
        x1, H1, W1 = self.patch_embed4(x1)
        for i, blk in enumerate(self.block4):
            x1 = blk(x1, H1, W1)
        x1 = self.norm4(x1)
        x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DecoderTransformer_v3(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[32, 64, 128, 256], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16], embed_dims=[], num_heads=[],
                 mlp_ratios=[], sr_ratios=[]):
        super(DecoderTransformer_v3, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        # convolutional Difference Modules

        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)

        self.CrossMM = CM2_decoder(in_channels)

        dilations = (1, 2, 4, 8, 16)
        reductions = (4, 8, 16, 32, 64)
        chs = [32, 64, 128, 320]
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.Translayers = nn.ModuleList([
            BasicConv2d(chs[-3], chs[-2], 1),
            BasicConv2d(chs[-2], chs[-3], 1),
            BasicConv2d(chs[-3], chs[-4], 1)])
        self.AFA1 = AuxFeaAggregation(chs[-2], chs[-2], dilations[0], reductions[0], chs[-2])
        self.AFA2 = AuxFeaAggregation(chs[-3], chs[-3], dilations[1], reductions[1], chs[-3])
        self.AFA3 = AuxFeaAggregation(chs[-4], chs[-4], dilations[2], reductions[2], chs[-4])
        # taking outputs from middle of the encoder
        # self.make_pred_c4 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c3 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c2 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)
        # self.make_pred_c1 = make_prediction(in_channels=self.embedding_dim, out_channels=self.output_nc)

        # self.change_3d_c3 = nn.Sequential(ConvLayer(self.embedding_dim, 1, kernel_size=3, stride=1, padding=1),
        #                                nn.Tanh())
        # self.change_3d_c2 = nn.Sequential(ConvLayer(self.embedding_dim, 1, kernel_size=3, stride=1, padding=1),
        #                                nn.Tanh())
        # self.change_3d_c1 = nn.Sequential(ConvLayer(self.embedding_dim, 1, kernel_size=3, stride=1, padding=1),
        #                                nn.Tanh())
        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        # self.dense_2x = cgblock(self.embedding_dim, self.embedding_dim,
        #                       dilation=2, reduction=8, skip_connect=True)
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        # self.dense_1x = cgblock(self.embedding_dim, self.embedding_dim,
        #                        dilation=1, reduction=4, skip_connect=True)
        self.change_2d = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)
        self.change_3d = nn.Sequential(
            ConvLayer(self.embedding_dim, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        # self.change_3d_bg = nn.Sequential(
        #                 ConvLayer(self.embedding_dim, 1, kernel_size=3, stride=1, padding=1),
        #                 nn.Tanh())
        self.change_plabel = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):

        out2d, out3d, out3d_bg, CMMs1, CMMs2 = self.CrossMM(inputs1, inputs2)

        # Transforming encoder features (select layers)
        inputs1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        inputs2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = inputs1[-1].shape

        outputs = []
        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(inputs1[-1]).permute(0, 2, 1).reshape(n, -1, h, w)
        _c4_2 = self.linear_c4(inputs2[-1]).permute(0, 2, 1).reshape(n, -1, h, w)
        _c4 = self.diff_c4(torch.cat([_c4_1, _c4_2], dim=1))
        # p_c4  = self.make_pred_c4(_c4)
        # outputs.append(p_c4)

        # Stage 3: x1/16 scale
        n, _, h, w = inputs1[-2].shape
        _c3_1 = self.linear_c3(inputs1[-2]).permute(0, 2, 1).reshape(n, -1, h, w)
        _c3_2 = self.linear_c3(inputs2[-2]).permute(0, 2, 1).reshape(n, -1, h, w)
        _c3 = self.diff_c3(torch.cat([_c3_1, _c3_2], dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        # p_c3  = self.make_pred_c3(_c3)
        # p_c3_3d = self.change_3d_c3(_c3)
        # outputs.append(p_c3)

        # Stage 2: x1/8 scale
        n, _, h, w = inputs1[-3].shape
        _c2_1 = self.linear_c2(inputs1[-3]).permute(0, 2, 1).reshape(n, -1, h, w)
        _c2_2 = self.linear_c2(inputs2[-3]).permute(0, 2, 1).reshape(n, -1, h, w)
        _c2 = self.diff_c2(torch.cat([_c2_1, _c2_2], dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        # p_c2  = self.make_pred_c2(_c2)
        # p_c2_3d = self.change_3d_c2(_c2)
        # outputs.append(p_c2)

        # Stage 1: x1/4 scale
        n, _, h, w = inputs1[-4].shape
        _c1_1 = self.linear_c1(inputs1[-4]).permute(0, 2, 1).reshape(n, -1, h, w)
        _c1_2 = self.linear_c1(inputs2[-4]).permute(0, 2, 1).reshape(n, -1, h, w)
        _c1 = self.diff_c1(torch.cat([_c1_1, _c1_2], dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")
        # p_c1  = self.make_pred_c1(_c1)
        # p_c1_3d = self.change_3d_c1(_c1)
        # outputs.append(p_c1)

        # Linear Fusion of difference image from all scales

        # _c4 = resize(_c4, size=c1_size[2:], mode='bilinear', align_corners=False)
        # _c3 = resize(_c3, size=c1_size[2:], mode='bilinear', align_corners=False)
        # _c2 = resize(_c2, size=c1_size[2:], mode='bilinear', align_corners=False)
        # CMMs = [resize(CMM_, size=c1_size[2:], mode='bilinear', align_corners=False) for CMM_ in CMMs]

        # x = self.linear_fuse(torch.cat(([_c4, _c3, _c2, _c1] + CMMs), dim=1))

        AF1 = self.AFA1(CMMs1[0], CMMs2[0], _c3) + self.Translayers[0](self.upsamplex2(_c4))
        AF2 = self.AFA2(CMMs1[1], CMMs2[1], _c2) + self.Translayers[1](self.upsamplex2(AF1))
        AF3 = self.AFA3(CMMs1[2], CMMs2[2], _c1) + self.Translayers[2](self.upsamplex2(AF2))
        c1_size = AF3.size()
        AF1 = resize(AF1, size=c1_size[2:], mode='bilinear', align_corners=False)
        AF2 = resize(AF2, size=c1_size[2:], mode='bilinear', align_corners=False)
        #

        x = self.linear_fuse(torch.cat(([AF1, AF2, AF3]), dim=1))


        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None
        vis_feature = x.detach()
        x = self.convd2x(x)
        # Residual block
        x = self.dense_2x(x)

        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)
        #vis_feature = x.detach()
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        x3d = self.change_3d(x)
        x2d = self.change_2d(x)

        # xplabel = self.change_plabel(x)
        # outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return [x2d]+out2d, [x3d]+out3d, out3d_bg#, vis_feature  # x3d, x2d, xplabel
        # return [cp, p_c1, p_c2, p_c1], [cp_3d, p_c1_3d, p_c2_3d, p_c1_3d]


# ChangeFormerV6:
class hatformer_baseline_and_BHE_FME_AFA(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, decoder_softmax=False, embed_dim=64):
        super(hatformer_baseline_and_BHE_FME_AFA, self).__init__()
        # Transformer Encoder
        self.embed_dims = [32, 64, 128, 320]  # [64, 128, 320, 512]#
        self.depths = [3, 3, 4, 3]  # [3, 3, 6, 18, 3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.Tenc_x2 = EncoderTransformer_v3(img_size=512, patch_size=7, in_chans=input_nc, num_classes=output_nc,
                                             embed_dims=self.embed_dims,
                                             num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                             qk_scale=None, drop_rate=self.drop_rate,
                                             attn_drop_rate=self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             depths=self.depths, sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.TDec_x2 = DecoderTransformer_v3(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                             align_corners=False,
                                             in_channels=self.embed_dims, embedding_dim=self.embedding_dim,
                                             output_nc=output_nc,
                                             decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                             mlp_ratios=[4, 4, 4, 4],
                                             embed_dims=self.embed_dims[::-1], num_heads=[8, 4, 2, 1],
                                             sr_ratios=[1, 2, 4, 8])

    def forward(self, x1=None, x2=None):
        # input wrapping for feature visualization that
        # shapes as torch.cat([x1,x2],dim=0)
        # x1: image data x2: dsm data
        vis_flag = 0
        if x1 is None or x2 is None:
            vis_flag = 1
            bsize = x1.shape[0] // 2
            x1, x2 = x1[:bsize, :, :, :].requires_grad_(), x1[bsize:, :, :, :].requires_grad_()

        [fx1, fx2] = [self.Tenc_x2(x1), self.Tenc_x2(x2)]
        if not vis_flag:
            return self.TDec_x2(fx1, fx2)
        else:
            # return the 2d prediction for feature visualization
            # import pdb;pdb.set_trace()
            return [self.TDec_x2(fx1, fx2)[0]]
