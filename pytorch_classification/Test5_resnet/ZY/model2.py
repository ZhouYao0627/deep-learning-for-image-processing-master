import torch
import torch.nn as nn
import math
import numpy as np
from pytorch_classification.Test5_resnet.reparams.se_block import SEBlock


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)  # 参考论文m = n/s
        new_channels = init_channels * (ratio - 1)  # n(s-1)/s

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GlobalContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


# 构造conv+bn组合
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class Branch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False, use_se=False):
        super(Branch, self).__init__()
        self.deploy = deploy
        self.groups = groups  # 输入的特征层分为几组，这是分组卷积概念，单卡GPU不用考虑，默认为1，分组卷积概念详见下面
        self.in_channels = in_channels  # 3

        assert kernel_size == 3
        assert padding == 1  # 图像padding=1后经过 3x3 卷积之后图像大小不变

        padding_11 = padding - kernel_size // 2  # 1 - 3 // 2 = 0

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)  # SEBlock(48, 3)
        else:
            self.se = nn.Identity()

        if deploy:  # 定义推理模型时，基本block就是一个简单的 conv2D
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        else:  # 定义训练模型时，基本block是 identity、1x1 conv_bn、3x3 conv_bn 组合
            # 直接连接，类似resnet残差连接，注意当输入通道和输出通道不同时候，只有 1x1 和 3x3 卷积，没有identity
            # 当RepVGG Block的in_channel ≠ out_channel时，在RepVGG里面负责下采样（stride=2），输入特征图空间维度缩小，特征通道增加，
            # 以便提取高层语义特征，此时Block没有Identity
            # 当RepVGG Block的in_channel = out_channel时，Block包含三个Branch，stride=1
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)  # 这句话就是判断这个block没有identity，没有的话返回None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):  # 推理阶段, conv2D 后 ReLU，但这里使用了senet
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        # 训练阶段，3x3、1x1、identity 相加后 ReLU
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight
              / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight
              / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (
                t3 ** 2 + t1 ** 2)).sum()  # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   该函数以可微分的方式推导出等效的内核和偏差。
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    #   您可以随时获得等效的内核和偏差，并做任何您想做的事情，例如，在训练期间应用一些惩罚或约束，就像您对其他模型所做的那样。
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)  # 卷积核两个参数 W 和 b 提出来
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            # 这代码是将 1x1 conv padding 一圈成 3x3 conv，填充的是0
            #                     [0  0  0]
            # [1]  >>>padding>>>  [0  1  0]
            #                     [0  0  0]
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:  # 当branch不是3x3、1x1、BN，那就返回 W=0, b=0
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight  # conv权重
            running_mean = branch.bn.running_mean  # BN mean
            running_var = branch.bn.running_var  # BN var
            gamma = branch.bn.weight  # BN γ
            beta = branch.bn.bias  # BN β
            eps = branch.bn.eps  # 防止分母为0
            # 当branch是3x3、1x1时候，返回以上数据，为后面做融合
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups  # 通道分组，单个GPU不用考虑
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)  # 定义新的3x3卷积核，参数为0，这里用到DepthWise
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1  # 将卷积核对角线部分赋予1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)

            kernel = self.id_tensor  # conv权重
            running_mean = branch.running_mean  # BN mean
            running_var = branch.running_var  # BN var
            gamma = branch.weight  # BN γ
            beta = branch.bias  # BN β
            eps = branch.eps  # 防止分母为0
            # 当branch是 identity，也即只有BN时候返回以上数据

        # 提取W、b，不管是 3x3 1x1 identity都要提取
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        mid_channel = in_channel

        # Point-wise expansion
        self.ghost1 = GhostModule(in_channel, mid_channel, relu=True)

        # multi-channel
        self.deploy = False
        self.use_se = True
        self.branch = Branch(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                             padding=1, deploy=self.deploy, use_se=self.use_se)

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_channel, out_channel, relu=False)

        # GCB Attention Module
        self.GCB = GlobalContextBlock(inplanes=in_channel, ratio=0.25)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Multi-branch
        x = self.branch(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        # GCB
        x = self.GCB(x)

        x += identity
        x = self.relu(x)

        return x


class ZYNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        super(ZYNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 112*112*64
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 56*56*64

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def zynet(num_classes=1000, include_top=True):
    return ZYNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
