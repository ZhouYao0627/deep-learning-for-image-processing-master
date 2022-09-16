import os
import json
import torch.nn as nn
import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# from pytorch_classification.Test8_mobilenet.model_v2 import MobileNetV2
from model_resnet import resnet50, resnext50_32x4d, resnext101_32x8d
from model_densenet import densenet121, load_state_dict1, densenet161
from CSAM_before_than_cat1 import csam as csam1
from CSAM_before_than_cat2 import csam as csam2


class EF(nn.Module):

    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

        # self.cbam1 = cbam(2048)
        # self.cbam2 = cbam(2208)
        # self.se_block1 = se_block(2048)
        # self.se_block2 = se_block(2208)

        self.csam1 = csam1(2048)
        self.csam2 = csam2(2208)

        # self.cbam = cbam(4256)
        # self.csam = csam(4256)
        # self.se_block = se_block(4256)

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(nn.Linear(4256, 7))

    def forward(self, input):
        f1 = self.net1(input)  # resNext101: [128, 2048, 7, 7]
        f2 = self.net2(input)  # densenet161: [128, 2208, 7, 7]

        # f1 = self.cbam1(f1)
        # f2 = self.cbam2(f2)
        # f1 = self.se_block1(f1)
        # f2 = self.se_block2(f2)
        f1 = self.csam1(f1)
        f2 = self.csam2(f2)

        # 4-5dropout
        # f1 = self.dropout(f1)
        # f2 = self.dropout(f2)

        x = torch.cat([f1, f2], dim=1)
        # x = self.csam(x)
        # x = self.cbam(x)
        # x = self.se_block(x)
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

    def summary(self):  # 统计计算各项指标
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

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)  # 打印混淆矩阵
        # 展示混淆矩阵；颜色从白色->蓝色  plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示。其后跟着plt.show()才能显示出来。
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90)  # x轴的信息替换成labels，旋转45°
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)  # x轴的信息替换成labels
        # 显示colorbar
        plt.colorbar()  # 混淆矩阵右侧的色谱
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')
        plt.savefig('D_R_cat_twocsam_drop05_2_RSSCN7_1.jpg')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2  # 最大值的一半作为阈值
        for x in range(self.num_classes):  # x从左到右
            for y in range(self.num_classes):  # y从上到下
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])  # 这里的行对应的是y坐标，列对应的是x坐标
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()  # 图形显示更加紧凑
        plt.show()  # 展示混淆矩阵


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "RSSCN7", "50_50")  # flower data set path
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)

    batch_size = 64
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=2)
    # net = MobileNetV2(num_classes=30)

    net1 = resnext101_32x8d(include_top=False)
    net2 = densenet161()
    net = EF(net1, net2)

    # load pretrain weights  加载的是自己在数据集上跑出来的模型权重
    model_weight_path = "../Test8_densenet/D_R_cat_twocsam_drop05_2_RSSCN7_1.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = '../Test8_densenet/class_indices_RSSCN7.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]  # 将标签信息提取出来
    confusion = ConfusionMatrix(num_classes=7, labels=labels)
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
