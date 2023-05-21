import torch.nn as nn
import numpy as np
import torch
import copy
from se_block import SEBlock


# 构造conv+bn组合
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
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


class RepVGG(nn.Module):
    # 以create_RepVGG_A0为基础标注
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False,
                 use_se=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4  # 瘦身因子，减小网络的宽度，就是输出通道乘以权重变小还是变大 # width_multiplier=[0.75, 0.75, 0.75, 2.5]

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()  # 这部分是分组卷积，单个GPU不用考虑
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))  # 64*0.75=48

        # 第一个stage处理大分辨率，只设计一个3x3卷积而减小参数量
        # 最后一层channel很多，只设计一个3x3卷积而减小参数量
        # 按照ResNet，更多层放到倒数第二个stage
        # 为了实现下采样，每个stage第一个3x3卷积将stride设置2
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1  # 分组卷积
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)  # 64*0.75=48  2
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)  # 128*0.75=96  4
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)  # 256*0.75=192  14
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)  # 512*2.5=1280  1
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)  # 全局池化，变成 Nx1x1（CxHxW），类似 flatten
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)  # 512*2.5=1280  1000

    def _make_stage(self, planes, num_blocks, stride):  # 第一次：48，2，2  第二次：96，4，2  第三次：192，14，2 第四次：1280，1，2
        strides = [stride] + [1] * (num_blocks - 1)  # [2]+[1]*(2-1)=3,[2]+[1]*(4-1)=5,[2]+[1]*(14-1)=15,[2]+[1]*(1-1)=2
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)  # 分组卷积
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ============ for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =============  example_pspnet.py shows an example

def repvgg_model_convert(model: torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model
