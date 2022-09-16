import torch
from torch import nn
import torch.nn.functional as F
from torchstat import stat  # 查看网络参数
from collections import OrderedDict

COEFF = 12.0


class SoftGate(nn.Module):
    def __init__(self):
        super(SoftGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x).mul(COEFF)


def lip2d(x, logit, kernel_size=3, stride=2, padding=1):
    weight = torch.exp(logit)
    return F.avg_pool2d(x * weight, kernel_size, stride, padding) / F.avg_pool2d(weight, kernel_size, stride, padding)


# (1) 通道注意力机制
class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(channel_attention, self).__init__()

        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层, 通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入特征图的shape
        b, c, h, w = inputs.shape

        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # 调整池化结果的维度 [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        # 激活函数
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # 将这两种池化结果相加 [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        # sigmoid函数权值归一化
        x = self.sigmoid(x)
        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 输入特征图和通道权重相乘 [b,c,h,w]
        outputs = inputs * x

        outputs = self.sigmoid(outputs)

        return outputs


# (2) 空间注意力机制
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        # self.conv1 = nn.Conv2d(in_channels=4096, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)

        # # 反卷积复原特征图大小 [b,1,h/2,w/2]==>[b,1,h,w]
        # self.deconv = nn.ConvTranspose2d(in_channels=4096, out_channels=1, kernel_size=1, stride=2, padding=0,
        #                                  bias=False)
        # # sigmoid函数
        # self.sigmoid = nn.Sigmoid()
        #
        # # rp = in_channel
        #
        # self.logit = nn.Sequential(
        #     OrderedDict((
        #         ('conv', nn.Conv2d(2048, 2048, 3, padding=1, bias=False)),
        #         ('bn', nn.InstanceNorm2d(2048, affine=True)),
        #         ('gate', SoftGate()),
        #     ))
        # )

        # 反卷积复原特征图大小 [b,1,h/2,w/2]==>[b,1,h,w]
        self.deconv = nn.ConvTranspose2d(in_channels=4096, out_channels=1, kernel_size=1, stride=2, padding=0,
                                         bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

        # rp = in_channel

        self.logit = nn.Sequential(
            OrderedDict((
                ('conv', nn.Conv2d(2048, 2048, 3, padding=1, bias=False)),
                ('bn', nn.InstanceNorm2d(2048, affine=True)),
                ('gate', SoftGate()),
            ))
        )


    def init_layer(self):
        self.logit[0].weight.data.fill_(0.0)

    # 前向传播
    def forward(self, inputs):
        # 分别得出Local Maxpooling和Local Avgpooling
        LMP = lip2d(inputs, self.logit(inputs))
        LAP = lip2d(inputs, self.logit(inputs))
        # print("LMP.shape and LAP.shape", LMP.shape, LAP.shape)

        # 在通道维度上最大池化 [b,1,h/2,w/2]  keepdim保留原有深度  # 返回值是在某维度的最大值和对应的索引
        GMP1, _ = torch.max(LMP, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h/2,w/2]
        GAP1 = torch.mean(LMP, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h/2,w/2]
        # print("GMP1.shape and GAP1.shape", GMP1.shape, GAP1.shape)
        x1 = torch.cat((GMP1, GAP1), dim=1)
        # print(x1.shape)

        # 卷积融合通道信息 [b,2,h/2,w/2]==>[b,1,h/2,w/2]
        x1 = self.conv(x1)
        # print("x1", x1.shape)
        # print("inputs", inputs.shape)
        # print("LMP", LMP.shape)

        # LMP特征图和空间权重相乘
        outputs1 = LMP * x1

        # 在通道维度上最大池化 [b,1,h/2,w/2]  keepdim保留原有深度  # 返回值是在某维度的最大值和对应的索引
        GMP2, _ = torch.max(LAP, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h/2,w/2]
        GAP2 = torch.mean(LAP, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h/2,w/2]
        x2 = torch.cat([GMP2, GAP2], dim=1)
        # 卷积融合通道信息 [b,2,h/2,w/2]==>[b,1,h/2,w/2]
        x2 = self.conv(x2)
        # LAP特征图和空间权重相乘
        outputs2 = LAP * x2

        # print("outputs1", outputs1.shape)
        # print("outputs2", outputs2.shape)

        # 两个池化后的结果在通道维度上堆叠 [b,2,h/2,w/2]
        outputs3 = torch.cat([outputs1, outputs2], dim=1)
        # 卷积融合通道信息 [b,2,h/2,w/2]==>[b,1,h/2,w/2]
        # print("outputs3", outputs3.shape)
        # outputs3 = self.conv1(outputs3)

        # 对特征图进行反卷积操作 [b,1,h/2,w/2]==>[b,1,h,w]
        outputs4 = self.deconv(outputs3)

        outputs = self.sigmoid(outputs4)

        return outputs


# (3) CSAM注意力机制
class csam(nn.Module):
    # 初始化，in_channel和ratio=4代表通道注意力机制的输入通道数和第一个全连接下降的通道数
    # kernel_size代表空间注意力机制的卷积核大小
    def __init__(self, in_channel, ratio=4, kernel_size=7):
        # 继承父类初始化方法
        super(csam, self).__init__()

        # 实例化通道注意力机制
        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        # 实例化空间注意力机制
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)
        # sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 先将输入图像经过通道注意力机制
        x1 = self.channel_attention(inputs)
        # 然后经过空间注意力机制
        x2 = self.spatial_attention(inputs)
        # 通道注意力和空间注意力进行点乘
        # print("x1.shape", x1.shape)
        # print("x2.shape", x2.shape)

        x3 = x1 * x2
        x3 = self.sigmoid(x3)
        x = inputs * x3

        return x

# # 查看网络结构:构造输入层，查看一次前向传播的输出结果，打印网络结构
# # 构造输入层 [b,c,h,w]==[4,32,16,16]
# inputs = torch.rand([64, 4256, 7, 7])
# # 获取输入图像的通道数
# in_channel = inputs.shape[1]
# # 模型实例化
# model = csam(in_channel=in_channel)
# # 前向传播
# outputs = model(inputs)
#
# print("outputs.shape", outputs.shape)  # 查看输出结果
# print("model", model)  # 查看网络结构
# # stat(model, input_size=[64, 7, 7])  # 查看网络参数
