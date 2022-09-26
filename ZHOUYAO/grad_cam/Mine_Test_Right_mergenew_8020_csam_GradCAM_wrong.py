import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from CSAM_before_than_cat1 import csam as csam1
from CSAM_before_than_cat2 import csam as csam2
from model_resnet import resnet50, resnext50_32x4d, resnext101_32x8d, resnet101
from model_densenet import densenet121, load_state_dict1, densenet161, densenet201

import argparse


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
        self.fc = nn.Sequential(nn.Linear(1024, 21))

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

        print("f1.shape", f1.shape)
        print("f2.shape", f2.shape)

        f1 = self.csam1(f1)
        f2 = self.csam2(f2)

        net = mergeblock()
        x = net.forward(f1, f2)  # 7*7*4256

        x = self.conv_3x3_1(x)
        x = self.bn(x)

        # x = self.dropout(x)  # 2dropout  效果比1dropout好
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)  # 1dropout
        x = self.fc(x)
        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net1 = resnext101_32x8d(include_top=False)
    net2 = densenet161()

    model = EF(net1, net2)
    target_layers = [model.conv_3x3_1]

    # create model
    # model = resnext101_32x8d(num_classes=21).to(device)
    # target_layers = [model.layer4[-1]]

    # load model weights
    weights_path = "../Test8_densenet/save_weights/D_R_UCM21_95_5_Merge_0dropout_csam.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    img_path = os.path.join(data_root, "data_set", "UCM21", "50_50", "train", "overpass", "overpass15.tif")
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # [N, C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = None  # 类别真实数减1

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.colorbar()  # 显示colorbar
    # plt.savefig("./plot/GradCAM_UCM21_overpass00.png", dpi=500, format="png")
    plt.show()


if __name__ == '__main__':
    main()
