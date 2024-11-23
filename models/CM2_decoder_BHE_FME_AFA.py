import torch
import torch.nn as nn
#from .resnet import resnet18
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
#from models.CGBlock import cgblock


def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
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


class CrossAtt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.query1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.query2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key2 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.conv_cat = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())  # conv_f

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        q1 = self.query1(input1)
        k1 = self.key1(input1).view(batch_size, -1, height * width)
        v1 = self.value1(input1).view(batch_size, -1, height * width)

        q2 = self.query2(input2)
        k2 = self.key2(input2).view(batch_size, -1, height * width)
        v2 = self.value2(input2).view(batch_size, -1, height * width)

        q = torch.cat([q1, q2], 1).view(batch_size, -1, height * width).permute(0, 2, 1)
        attn_matrix1 = torch.bmm(q, k1)
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))
        out1 = out1.view(*input1.shape)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q, k2)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2.view(*input2.shape)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(torch.cat([out1, out2], 1))
        return feat_sum, out1, out2


def draw_features(x, savename, width=8, height=8):
    # tic=time.time()
    fig = plt.figure(figsize=(60, 60))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    # print("time:{}".format(time.time()-tic))





class pred2mask(nn.Module):
    def __init__(self):
        super(pred2mask, self).__init__()
        kernel = torch.ones((5,5))
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        #kernel = np.repeat(kernel, 1, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.training = True
    def forward(self, out2d, fea, training=False):
        distmap = out2d[:,0,:,:].unsqueeze(1).detach()
        distmap = F.conv2d(distmap, self.weight, padding=3, groups=1)
        distmap = F.conv2d(distmap, self.weight, padding=3, groups=1)
        distmap = F.conv2d(distmap, self.weight, padding=3, groups=1)
        distmap = (distmap - distmap.min()) / (distmap.max() - distmap.min())
        # if training:
        #     rand_mask = distmap < torch.Tensor(np.random.random(distmap.size())).to(distmap.device)
        #     return fea * rand_mask, rand_mask
        # else:
        #     return fea * distmap, distmap
        return distmap

class CM2_decoder(nn.Module):
    def __init__(self, chs=[32, 64, 128, 320], num_classes=3, drop_rate=0.2):
        super(CM2_decoder, self).__init__()
        # ch: [32,64,128,320] size: [128,64,32,16]
        self.cross1 = CrossAtt(chs[-1], chs[-1])
        self.cross2 = CrossAtt(chs[-2], chs[-2])
        self.cross3 = CrossAtt(chs[-3], chs[-3])
        self.cross4 = CrossAtt(chs[-4], chs[-4])



        self.Translayer1_1 = BasicConv2d(chs[-1], chs[-2], 1)
        self.fam21_1 = decode(chs[-2], chs[-2], chs[-2])  # AlignBlock(128) # decode(128,128,128)
        self.Translayer2_1 = BasicConv2d(chs[-2], chs[-3], 1)
        self.fam32_1 = decode(chs[-3], chs[-3], chs[-3])  # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_1 = BasicConv2d(chs[-3], chs[-4], 1)
        self.fam43_1 = decode(chs[-4], chs[-4], chs[-4])  # AlignBlock(64) # decode(64,64,64)

        self.Translayer1_2 = BasicConv2d(chs[-1], chs[-2], 1)
        self.fam21_2 = decode(chs[-2], chs[-2], chs[-2])  # AlignBlock(128) # decode(128,128,128)
        self.Translayer2_2 = BasicConv2d(chs[-2], chs[-3], 1)
        self.fam32_2 = decode(chs[-3], chs[-3], chs[-3])  # AlignBlock(128) # decode(128,128,128)
        self.Translayer3_2 = BasicConv2d(chs[-3], chs[-4], 1)
        self.fam43_2 = decode(chs[-4], chs[-4], chs[-4])  # AlignBlock(64) # decode(64,64,64)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsamplex16 = nn.Upsample(scale_factor=16, mode='bilinear')

        # semantic change decoding
        self.final2 = nn.Sequential(
            Conv(chs[-4], chs[-4]//2, 3, bn=True, relu=True),
            Conv(chs[-4]//2, num_classes, 3, bn=False, relu=False)
        )
        self.final2_2 = nn.Sequential(
            Conv(chs[-3], chs[-3]//2, 3, bn=True, relu=True),
            Conv(chs[-3]//2, num_classes, 3, bn=False, relu=False)
        )
        self.final2_3 = nn.Sequential(
            Conv(chs[-2], chs[-2]//2, 3, bn=True, relu=True),
            Conv(chs[-2]//2, num_classes, 3, bn=False, relu=False)
        )
        # height change decoding
        self.final2_3d = nn.Sequential(
            Conv(chs[-4], chs[-4]//2, 3, bn=True, relu=True),
            Conv(chs[-4]//2, 1, 3, bn=False, relu=False)
        )
        self.final2_2_3d = nn.Sequential(
            Conv(chs[-3], chs[-3]//2, 3, bn=True, relu=True),
            Conv(chs[-3]//2, 1, 3, bn=False, relu=False)
        )
        self.final2_3_3d = nn.Sequential(
            Conv(chs[-2], chs[-2]//2, 3, bn=True, relu=True),
            Conv(chs[-2]//2, 1, 3, bn=False, relu=False)
        )
        # background height estimation
        self.final_3d = nn.Sequential(
            Conv(chs[-4], chs[-4]//2, 3, bn=True, relu=True),
            Conv(chs[-4]//2, 1, 3, bn=False, relu=False)
        )
        self.final_2_3d = nn.Sequential(
            Conv(chs[-3], chs[-3]//2, 3, bn=True, relu=True),
            Conv(chs[-3]//2, 1, 3, bn=False, relu=False)
        )
        self.final_3_3d = nn.Sequential(
            Conv(chs[-2], chs[-2]//2, 3, bn=True, relu=True),
            Conv(chs[-2]//2, 1, 3, bn=False, relu=False)
        )


        self.pred2mask = pred2mask()

        self.fuse_convs = nn.ModuleList()
        for i in [2, 3, 4]:
            self.fuse_convs.append(nn.Sequential(
            nn.Conv2d(chs[-i], chs[-i], kernel_size=3, padding=1),
            nn.BatchNorm2d(chs[-i]),
            nn.ReLU()))

        self.fuse_convs1 = nn.ModuleList()
        for i in [2, 3, 4]:
            self.fuse_convs1.append(nn.Sequential(
            nn.Conv2d(chs[-i], chs[-i], kernel_size=3, padding=1),
            nn.BatchNorm2d(chs[-i]),
            nn.ReLU()))

        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]


        self.init_weights()


    def fuse(self, x1, x2, fuse_conv):
        return fuse_conv(x1 + F.interpolate(x2, scale_factor=2, mode="bilinear"))

    def forward(self, inputs1, inputs2):
        # ch: [32,64,128,320] size: [128,64,32,16]
        c1_1, c1_2, c1_3, c1_4 = inputs1
        c2_1, c2_2, c2_3, c2_4 = inputs2

        cross_result1, cur1_1, cur2_1 = self.cross1(c1_4, c2_4)# [320,16,16]
        cross_result2, cur1_2, cur2_2 = self.cross2(c1_3, c2_3)# [128,32,32]
        cross_result3, cur1_3, cur2_3 = self.cross3(c1_2, c2_2)# [64,64,64]
        cross_result4, cur1_4, cur2_4 = self.cross4(c1_1, c2_1)# [32,128,128]

        out2_2 = self.fuse(cross_result2, self.Translayer1_2(cross_result1), self.fuse_convs1[0])
        out3_2 = self.fuse(cross_result3, self.Translayer2_2(out2_2), self.fuse_convs1[1])
        out4_2 = self.fuse(cross_result4, self.Translayer3_2(out3_2), self.fuse_convs1[2])

        out2 = self.fuse(cur1_2, self.Translayer1_1(cur1_1), self.fuse_convs[0])
        out3 = self.fuse(cur1_3, self.Translayer2_1(out2), self.fuse_convs[1])
        out4 = self.fuse(cur1_4, self.Translayer3_1(out3), self.fuse_convs[2])

        # ''' Auxiliary Feature Fusion '''
        # AF1 = self.AFA1(out2, out2_2)
        # AF2 = self.AFA2(out3, out3_2) + self.Translayers[0](self.upsamplex2(AF1))
        # AF3 = self.AFA3(out4, out4_2) + self.Translayers[1](self.upsamplex2(AF2))

        ''' change decoding with subtraction features '''
        # semantic decoding
        out4_2_up = self.upsamplex4(out4_2)
        out3_2_up = self.upsamplex8(out3_2)
        out2_2_up = self.upsamplex16(out2_2)

        pred_2d_1 = self.final2(out4_2_up)
        pred_2d_2 = self.final2_2(out3_2_up)
        pred_2d_3 = self.final2_3(out2_2_up)

        # height change decoding
        pred_3d_1 = self.final2_3d(out4_2_up)
        pred_3d_2 = self.final2_2_3d(out3_2_up)
        pred_3d_3 = self.final2_3_3d(out2_2_up)

        ''' background height decoding with fused features '''
        out4_up = self.upsamplex4(out4)
        out3_up = self.upsamplex8(out3)
        out2_up = self.upsamplex16(out2)
        # distmap1 = self.pred2mask(pred_2d_1, out4_up)
        # distmap2 = self.pred2mask(pred_2d_2, out3_up)
        # out_1 = self.final(out4_up)
        # out_1_2 = self.final_2(self.upsamplex8(out3))
        pred_bg_3d_1 = self.final_3d(out4_up)
        pred_bg_3d_2 = self.final_2_3d(out3_up)
        pred_bg_3d_3 = self.final_3_3d(out2_up)

        _, C, H, W = out4_2.shape
        #vis_feature = AF3.detach()#torch.cat([cur1_1.view(C, H * W), cur2_1.view(C, H * W)], dim=1).clone().detach()
        return [pred_2d_1, pred_2d_2, pred_2d_3], [pred_3d_1, pred_3d_2, pred_3d_3], \
               [pred_bg_3d_1, pred_bg_3d_2, pred_bg_3d_3], [out2, out3, out4],[out2_2, out3_2, out4_2],#vis_feature#[out3_2, out4_2, out4, out3]



    def init_weights(self):
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)
        self.cross4.apply(init_weights)

        for i in range(3):
            self.fuse_convs[i].apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final2.apply(init_weights)
        self.final2_2.apply(init_weights)
        self.final2_3.apply(init_weights)

        self.final2_3d.apply(init_weights)
        self.final2_2_3d.apply(init_weights)
        self.final2_3_3d.apply(init_weights)

        self.final_3d.apply(init_weights)
        self.final_2_3d.apply(init_weights)
        self.final_3_3d.apply(init_weights)