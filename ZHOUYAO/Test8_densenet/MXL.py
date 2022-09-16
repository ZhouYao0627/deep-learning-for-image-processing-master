import torch
from torch import nn


# from torchstat import stat  # 查看网络参数
#
#
# # (1) 通道注意力机制
# class channel_attention(nn.Module):
#     # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
#     def __init__(self, in_channel, ratio=4):
#         # 继承父类初始化方法
#         super(channel_attention, self).__init__()
#
#         # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
#         self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
#         # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
#         self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
#
#         # 第一个全连接层, 通道数下降4倍
#         self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
#         # 第二个全连接层, 恢复通道数
#         self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
#
#         # relu激活函数
#         self.relu = nn.ReLU()
#         # sigmoid激活函数
#         self.sigmoid = nn.Sigmoid()
#
#     # 前向传播
#     def forward(self, inputs):
#         # 获取输入特征图的shape
#         b, c, h, w = inputs.shape
#
#         # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]
#         max_pool = self.max_pool(inputs)
#         # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]
#         avg_pool = self.avg_pool(inputs)
#
#         # 调整池化结果的维度 [b,c,1,1]==>[b,c]
#         max_pool = max_pool.view([b, c])
#         avg_pool = avg_pool.view([b, c])
#
#         # 第一个全连接层下降通道数 [b,c]==>[b,c//4]
#         x_maxpool = self.fc1(max_pool)
#         x_avgpool = self.fc1(avg_pool)
#
#         # 激活函数
#         x_maxpool = self.relu(x_maxpool)
#         x_avgpool = self.relu(x_avgpool)
#
#         # 第二个全连接层恢复通道数 [b,c//4]==>[b,c]
#         x_maxpool = self.fc2(x_maxpool)
#         x_avgpool = self.fc2(x_avgpool)
#
#         # 将这两种池化结果相加 [b,c]==>[b,c]
#         x = x_maxpool + x_avgpool
#         # sigmoid函数权值归一化
#         x = self.sigmoid(x)
#         # 调整维度 [b,c]==>[b,c,1,1]
#         x = x.view([b, c, 1, 1])
#         # 输入特征图和通道权重相乘 [b,c,h,w]
#         outputs = inputs * x
#
#         return outputs
#
#
# # (2) 空间注意力机制
# class spatial_attention(nn.Module):
#     # 初始化，卷积核大小为7*7
#     def __init__(self, kernel_size=7):
#         # 继承父类初始化方法
#         super(spatial_attention, self).__init__()
#
#         # 为了保持卷积前后的特征图shape相同，卷积时需要padding
#         padding = kernel_size // 2
#         # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
#         self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
#         # sigmoid函数
#         self.sigmoid = nn.Sigmoid()
#
#     # 前向传播
#     def forward(self, inputs):
#         # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
#         # 返回值是在某维度的最大值和对应的索引
#         x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
#
#         # 在通道维度上平均池化 [b,1,h,w]
#         x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
#         # 池化后的结果在通道维度上堆叠 [b,2,h,w]
#         x = torch.cat([x_maxpool, x_avgpool], dim=1)
#
#         # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
#         x = self.conv(x)
#         # 空间权重归一化
#         x = self.sigmoid(x)
#         # 输入特征图和空间权重相乘
#         outputs = inputs * x
#
#         return outputs


# (3) CBAM注意力机制
class mxl(nn.Module):
    # 初始化，in_channel和ratio=4代表通道注意力机制的输入通道数和第一个全连接下降的通道数
    # kernel_size代表空间注意力机制的卷积核大小
    def __init__(self):
        # 继承父类初始化方法
        super(mxl, self).__init__()

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.sigmoid = nn.Sigmoid()

        self.conv_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, bias=False)

    # 前向传播
    def forward(self, input1, input2):
        x1 = input2 * self.sigmoid(self.avg_pool(input1))
        x2 = input1 * self.sigmoid(self.avg_pool(input2))

        x3 = self.conv_1(x1)
        x4 = self.conv_1(x2)

        x5 = torch.cat((x3, x4), dim=1)
        x6 = self.sigmoid(x5)

        x7 = x5 * x6

        return x7


# 查看网络结构:构造输入层，查看一次前向传播的输出结果，打印网络结构
# 构造输入层 [b,c,h,w]==[4,32,16,16]
input1 = torch.rand([4, 32, 16, 16])
input2 = torch.rand([4, 32, 16, 16])
# 获取输入图像的通道数
# in_channel = inputs.shape[1]
# 模型实例化
model = mxl()
# 前向传播
outputs = model.forward(input1, input2)

print(outputs.shape)  # 查看输出结果
# print(model)  # 查看网络结构
# stat(model, input_size=[32, 16, 16])  # 查看网络参数