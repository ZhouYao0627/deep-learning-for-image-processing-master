import os
import json
import torch.nn as nn
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from model_resnet import resnet50, resnext50_32x4d, resnext101_32x8d
from model_densenet import densenet121, load_state_dict1, densenet161
from CSAM_before_than_cat1 import csam as csam1
from CSAM_before_than_cat2 import csam as csam2
import itertools
import seaborn as sns


class mergeblock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1), padding=0))
        self.conv2 = nn.Sequential(nn.Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1), padding=0))
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        # R-D
        self.conv1_1_dw = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1, groups=2048, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
        )
        self.conv1_1_pw = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
        )
        # self.conv1_2_dw = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1, groups=2048, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(2048),
        # )
        # self.conv1_2_pw = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(2048),
        # )
        # R-D
        self.conv2_1_dw = nn.Sequential(
            nn.Conv2d(in_channels=2208, out_channels=2208, kernel_size=3, stride=1, padding=1, groups=2208, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(2208),
        )
        self.conv2_1_pw = nn.Sequential(
            nn.Conv2d(in_channels=2208, out_channels=2208, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(2208),
        )
        # self.conv2_2_dw = nn.Sequential(
        #     nn.Conv2d(in_channels=2208, out_channels=2208, kernel_size=3, stride=1, padding=1, groups=2208, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(2208),
        # )
        # self.conv2_2_pw = nn.Sequential(
        #     nn.Conv2d(in_channels=2208, out_channels=2208, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(2208),
        # )

        # 修改通道数
        self.conv_1 = nn.Conv2d(in_channels=4256, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_channels=4256, out_channels=2208, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, InData1, InData2):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = self.conv1.to(device)
        self.conv2 = self.conv2.to(device)
        self.max_pool = self.max_pool.to(device)
        self.avg_pool = self.avg_pool.to(device)
        self.sigmoid = self.sigmoid.to(device)
        self.conv1_1_dw = self.conv1_1_dw.to(device)
        self.conv1_1_pw = self.conv1_1_pw.to(device)
        self.conv2_1_dw = self.conv2_1_dw.to(device)
        self.conv2_1_pw = self.conv2_1_pw.to(device)
        self.conv_1 = self.conv_1.to(device)
        self.conv_2 = self.conv_2.to(device)

        # 第一阶段
        max_out1 = self.max_pool(InData1)
        avg_out1 = self.avg_pool(InData2)
        sig1 = self.sigmoid(max_out1)
        sig2 = self.sigmoid(avg_out1)
        fea1 = InData1 * sig1
        fea2 = InData2 * sig2
        # print(fea1.shape,fea2.shape)

        # 第二阶段
        # max_out2, _ = torch.max(fea1, dim=1, keepdim=True)
        # avg_out2 = torch.mean(fea1, dim=1, keepdim=True)
        # am1 = self.conv1(torch.cat([max_out2, avg_out2], dim=1))
        #
        # max_out3, _ = torch.max(fea2, dim=1, keepdim=True)
        # avg_out3 = torch.mean(fea2, dim=1, keepdim=True)
        # am2 = self.conv2(torch.cat([max_out3, avg_out3], dim=1))
        #
        # sig3 = self.sigmoid(am1)
        # sig4 = self.sigmoid(am2)
        # # print('123131321',am1.shape,fea1.shape,sig3.shape)
        # fea3 = fea1 * sig3
        # fea4 = fea2 * sig4
        # result = torch.cat([fea3, fea4], dim=1)
        # print()
        # 第二阶段-> R-D
        x1 = fea1
        x2 = fea2
        x1 = self.conv1_1_dw(x1)
        x1 = self.conv1_1_pw(x1)
        x2 = self.conv2_1_dw(x2)
        x2 = self.conv2_1_pw(x2)
        x3 = torch.cat((x1, x2), dim=1)  # 7*7*4256
        x3 = self.conv_1(x3)  # 7*7*2048
        x4 = torch.add(fea1, x3)

        # 第二阶段-> D-R
        y1 = fea1
        y2 = fea2
        y1 = self.conv1_1_dw(y1)
        y1 = self.conv1_1_pw(y1)
        y2 = self.conv2_1_dw(y2)
        y2 = self.conv2_1_pw(y2)
        y3 = torch.cat((y1, y2), dim=1)  # 7*7*4256
        y3 = self.conv_2(y3)  # 7*7*2208
        y4 = torch.add(fea2, y3)

        result = torch.cat((x4, y4), dim=1)

        return result


class EF(nn.Module):

    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

        self.csam1 = csam1(2048)
        self.csam2 = csam2(2208)

        # 3x3卷积
        self.conv_3x3_1 = nn.Conv2d(in_channels=4256, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(1024)

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(nn.Linear(1024, 30))

    def forward(self, input):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        self.net1 = self.net1.to(device)
        self.net2 = self.net2.to(device)
        self.csam1 = self.csam1.to(device)
        self.csam2 = self.csam2.to(device)
        self.avgpool = self.avgpool.to(device)
        self.dropout = self.dropout.to(device)
        self.fc = self.fc.to(device)
        self.conv_3x3_1 = self.conv_3x3_1.to(device)
        self.bn = self.bn.to(device)

        f1 = self.net1(input)  # resNext101: [128, 2048, 7, 7]
        f2 = self.net2(input)  # densenet161: [128, 2208, 7, 7]  densenet121: [128, 1024, 7, 7]

        f1 = self.csam1(f1)
        f2 = self.csam2(f2)

        net = mergeblock()
        x = net.forward(f1, f2)  # 7*7*4256

        x = self.conv_3x3_1(x)
        x = self.bn(x)

        x = self.dropout(x)  # 2dropout  效果比1dropout好
        x = self.avgpool(x)
        # x = self.dropout(x)  # 3dropout  效果比1dropout好，比2dropout差
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # 1dropout
        x = self.fc(x)
        return x

class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化一个空矩阵
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):  # preds：预测的值，labels：真实标签
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    # 统计计算各项指标
    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 对角线上元素之和
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()  # 初始化一张表
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.  # 小数部分只取三位
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    # 绘制混淆矩阵
    def plot(self):
        matrix = self.matrix
        print(matrix)  # 打印混淆矩阵
        # 展示混淆矩阵；颜色从白色->蓝色  plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示。其后跟着plt.show()才能显示出来。
        plt.imshow(matrix, cmap=plt.cm.Blues)  # 显示数据

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90, fontsize=3)  # x轴的信息替换成labels，旋转45°
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, fontsize=3)  # x轴的信息替换成labels
        # 显示colorbar
        plt.colorbar()  # 混淆矩阵右侧的色谱
        # plt.clim(0, 1)
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')
        # plt.savefig('D_R_AID30_8020_Merge_new_CM1.jpg')
        # plt.savefig("./plot/D_R_AID30_8020_Merge_new_CM1.svg", dpi=300, format="svg")

        sns.set(font_scale=0.2)  # 调整混淆矩阵中字体的大小
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        print("matrix", matrix)

        # 在图中标注数量/概率信息
        fmt = '.2f'
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.tight_layout()  # 图形显示更加紧凑 #保证图不重叠
        # plt.savefig("./plot/D_R_AID30_9010_Merge_new_CM2.svg", dpi=300, format="svg")
        plt.show()  # 展示混淆矩阵


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "AID30", "90_10")  # flower data set path
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)

    batch_size = 64
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=2)

    net1 = resnext101_32x8d(include_top=False)
    net2 = densenet161()
    net = EF(net1, net2)

    # load pretrain weights  加载的是自己在数据集上跑出来的模型权重
    model_weight_path = "../Test8_densenet/save_weights/D_R_AID30_9010_Merge_new1.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = '../Test8_densenet/class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]  # 将标签信息提取出来
    confusion = ConfusionMatrix(num_classes=30, labels=labels)
    net.eval()  # 验证模式
    with torch.no_grad():  # 停止pytorch对变量的梯度跟踪
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()  # 绘制混淆矩阵
    confusion.summary()  # 打印指标信息
