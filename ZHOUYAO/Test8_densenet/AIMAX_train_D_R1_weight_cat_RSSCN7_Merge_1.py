import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse

from model_resnet import resnet50, resnext50_32x4d, resnext101_32x8d, resnet101
from model_densenet import densenet121, load_state_dict1, densenet161, densenet201
from CSAM_before_than_cat1 import csam as csam1
from CSAM_before_than_cat2 import csam as csam2
from CSAM_later_than_cat import csam
from SE import se_block


class R2D(nn.Module):
    def __init__(self):
        super(R2D, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=3072, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3072),
        )
        # self.conv1_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=3072, out_channels=3072, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(3072),
        # )
        self.conv1_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=3072, kernel_size=3, stride=1, padding=1, groups=3072, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3072),
        )
        self.conv1_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=3072, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3072),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=2048, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.BatchNorm2d(2048),
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=2208, out_channels=3312, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3312),
        )
        # self.conv2_2 = nn.Sequential(
        #     nn.Conv2d(in_channels=3312, out_channels=3312, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(3312),
        # )
        self.conv2_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=3312, out_channels=3312, kernel_size=3, stride=1, padding=1, groups=3312, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3312),
        )
        self.conv2_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=3312, out_channels=3312, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3312),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=3312, out_channels=2208, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.BatchNorm2d(2208),
        )

        # self.conv_1x1_1_1 = nn.Conv2d(in_channels=c1, out_channels=c1 * 2, kernel_size=1, padding=0, bias=False)
        # self.conv_3x3_1_2 = nn.Conv2d(in_channels=c1 * 2, out_channels=c1 * 2, kernel_size=3, stride=1, padding=1,
        #                               bias=False)
        # self.conv_1x1_1_3 = nn.Conv2d(in_channels=c1 * 2, out_channels=c1, kernel_size=1, padding=0, bias=False)

        # self.conv_1x1_2_1 = nn.Conv2d(in_channels=c2, out_channels=c2 * 2, kernel_size=1, padding=0, bias=False)
        # self.conv_3x3_2_2 = nn.Conv2d(in_channels=c2 * 2, out_channels=c2 * 2, kernel_size=3, stride=1, padding=1,
        #                               bias=False)
        # self.conv_1x1_2_3 = nn.Conv2d(in_channels=c2 * 2, out_channels=c2, kernel_size=1, padding=0, bias=False)

        # 修改通道数
        self.conv_1x1 = nn.Conv2d(in_channels=4256, out_channels=2048, kernel_size=1, padding=0, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()

        # bn
        self.bn = nn.BatchNorm2d(2048)

    def forward(self, Input1, Input2):
        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # Input1 = Input1.to(device)
        # Input2 = Input2.to(device)
        # self.conv1_1 = self.conv1_1.to(device)
        # # self.conv1_2 = self.conv1_2.to(device)
        # self.conv1_2_dw = self.conv1_2_dw.to(device)
        # self.conv1_2_pw = self.conv1_2_pw.to(device)
        # self.conv1_3 = self.conv1_3.to(device)
        #
        # self.conv2_1 = self.conv2_1.to(device)
        # # self.conv2_2 = self.conv2_2.to(device)
        # self.conv2_2_dw = self.conv2_2_dw.to(device)
        # self.conv2_2_pw = self.conv2_2_pw.to(device)
        # self.conv2_3 = self.conv2_3.to(device)
        #
        # self.conv_1x1 = self.conv_1x1.to(device)

        # x1 = self.conv_1x1_1_1(self.Input1)
        # self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(len(x1))
        x1 = self.conv1_1(Input1)

        # x1 = self.conv_3x3_1_2(x1)
        # self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(len(x1))
        # x1 = self.conv1_2(x1)
        x1 = self.conv1_2_dw(x1)
        x1 = self.conv1_2_pw(x1)

        # x1 = self.conv_1x1_1_3(x1)
        # self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(len(x1))
        x1 = self.conv1_3(x1)

        x1 = torch.add(x1, Input1)
        x1 = self.relu(x1)

        # x2 = self.conv_1x1_2_1(self.Input2)
        # self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(len(x2))
        x2 = self.conv2_1(Input2)

        # x2 = self.conv_3x3_2_2(x2)
        # self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(len(x2))
        # x2 = self.conv2_2(x2)
        x2 = self.conv2_2_dw(x2)
        x2 = self.conv2_2_pw(x2)

        # x2 = self.conv_1x1_2_3(x2)
        # self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(len(x2))
        x2 = self.conv2_3(x2)

        x2 = torch.add(x2, Input2)
        x2 = self.relu(x2)

        x3 = torch.cat([x1, x2], dim=1)  # 7*7*4256

        x3 = self.conv_1x1(x3)

        x = torch.add(x3, Input1)
        x = self.relu(x)

        return x


class D2R(nn.Module):
    def __init__(self):
        super(D2R, self).__init__()
        # self.Input1 = Input1
        # self.Input2 = Input2
        #
        # b1, c1, h1, w1 = Input1.shape  # [64, 2048, 7, 7]
        # b2, c2, h2, w2 = Input2.shape  # [64, 2208, 7, 7]

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=3072, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3072),
        )
        self.conv1_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=3072, kernel_size=3, stride=1, padding=1, groups=3072, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3072),
        )
        self.conv1_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=3072, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3072),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3072, out_channels=2048, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.BatchNorm2d(2048),
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=2208, out_channels=3312, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3312),
        )
        self.conv2_2_dw = nn.Sequential(
            nn.Conv2d(in_channels=3312, out_channels=3312, kernel_size=3, stride=1, padding=1, groups=3312, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3312),
        )
        self.conv2_2_pw = nn.Sequential(
            nn.Conv2d(in_channels=3312, out_channels=3312, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3312),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=3312, out_channels=2208, kernel_size=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.BatchNorm2d(2208),
        )

        # self.conv_1x1_1_1 = nn.Conv2d(in_channels=c1, out_channels=c1 * 2, kernel_size=1, padding=0, bias=False)
        # self.conv_3x3_1_2 = nn.Conv2d(in_channels=c1 * 2, out_channels=c1 * 2, kernel_size=3, stride=1, padding=1,
        #                               bias=False)
        # self.conv_1x1_1_3 = nn.Conv2d(in_channels=c1 * 2, out_channels=c1, kernel_size=1, padding=0, bias=False)

        # self.conv_1x1_2_1 = nn.Conv2d(in_channels=c2, out_channels=c2 * 2, kernel_size=1, padding=0, bias=False)
        # self.conv_3x3_2_2 = nn.Conv2d(in_channels=c2 * 2, out_channels=c2 * 2, kernel_size=3, stride=1, padding=1,
        #                               bias=False)
        # self.conv_1x1_2_3 = nn.Conv2d(in_channels=c2 * 2, out_channels=c2, kernel_size=1, padding=0, bias=False)

        # 修改通道数
        self.conv_1x1 = nn.Conv2d(in_channels=4256, out_channels=2208, kernel_size=1, padding=0, bias=False)

        # relu激活函数
        self.relu = nn.ReLU()

        # bn
        self.bn = nn.BatchNorm2d(2208)

    def forward(self, Input1, Input2):
        x1 = self.conv1_1(Input1)
        # x1 = self.conv1_2(x1)
        x1 = self.conv1_2_dw(x1)
        x1 = self.conv1_2_pw(x1)
        x1 = self.conv1_3(x1)
        x1 = torch.add(x1, Input1)
        x1 = self.relu(x1)

        x2 = self.conv2_1(Input2)
        # x2 = self.conv2_2(x2)
        x2 = self.conv2_2_dw(x2)
        x2 = self.conv2_2_pw(x2)
        x2 = self.conv2_3(x2)
        x2 = torch.add(x2, Input2)
        x2 = self.relu(x2)

        x3 = torch.cat([x1, x2], dim=1)  # 7*7*4256

        x3 = self.conv_1x1(x3)
        # x3 = self.bn(x3)

        x = torch.add(x3, Input2)
        x = self.relu(x)

        return x


class mergeblock(nn.Module):
    def __init__(self, InData1, InData2):
        super().__init__()
        self.InData1 = InData1  # 7*7*2048
        self.InData2 = InData2  # 7*7*2208

        self.R2D = R2D()  # 7*7*2048
        self.D2R = D2R()  # 7*7*2208

        # relu激活函数
        self.relu = nn.ReLU()

        # 修改通道数
        self.conv_1x1 = nn.Conv2d(in_channels=4256, out_channels=1024, kernel_size=1, padding=0, bias=False)

    def forward(self):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.InData1 = self.InData1.to(device)
        self.InData2 = self.InData2.to(device)
        self.R2D = self.R2D.to(device)
        self.D2R = self.D2R.to(device)
        self.relu = self.relu.to(device)
        self.conv_1x1 = self.conv_1x1.to(device)

        x1 = self.InData1  # 7*7*2048
        x2 = self.InData2  # 7*7*2208

        x3 = self.R2D.forward(x1, x2)  # 7*7*2048
        x4 = self.D2R.forward(x1, x2)  # 7*7*2208

        x = torch.cat([x3, x4], dim=1)  # 7*7*4256

        # 修改通道数
        x = self.conv_1x1(x)  # # 7*7*1024

        x = self.relu(x)

        return x


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

        # 修改通道数
        # self.conv_1x1 = nn.Conv2d(in_channels=4256, out_channels=2048, kernel_size=1, padding=0, bias=False)

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(nn.Linear(1024, 7))

    def forward(self, input):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        input = input.to(device)
        self.net1 = self.net1.to(device)
        self.net2 = self.net2.to(device)
        self.csam1 = self.csam1.to(device)
        self.csam2 = self.csam2.to(device)
        self.avgpool = self.avgpool.to(device)
        self.dropout = self.dropout.to(device)
        self.fc = self.fc.to(device)

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

        # x = torch.cat([f1, f2], dim=1)
        # 修改通道数
        # x = self.conv_1x1(x)

        net = mergeblock(f1, f2)
        x = net.forward()  # 7*7*1024

        # print("x***", x)
        # print("x.shape", x.shape)  # torch.Size([64, 1024, 7, 7])

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


def main(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "RSSCN7", "50_50")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices_RSSCN7.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net1 = resnext101_32x8d(include_top=False)
    net2 = densenet161()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location="cpu")
        # 删除有关分类类别的权重
        print(weights_dict.keys())

        for k in list(weights_dict.keys()):
            if "fc.weight" in k or "fc.bias" in k:
                del weights_dict[k]
        print("----", net1.load_state_dict(weights_dict, strict=False))

    # 是否冻结权重
    for name, para in net1.named_parameters():
        # 除最后的全连接层外，其他权重全部冻结
        if "fc" not in name:
            para.requires_grad_(False)

    model_weight_path2 = "./weights/densenet161-8d451a50.pth"
    assert os.path.exists(model_weight_path2), "file {} does not exist.".format(model_weight_path2)
    load_state_dict1(net2, "./weights/densenet161-8d451a50.pth")
    # net2.load_state_dict(torch.load(model_weight_path2, map_location="cpu"))
    # 是否冻结权重
    for name1, para1 in net2.named_parameters():
        # 除最后的全连接层外，其他权重全部冻结
        if "classifier" not in name1:
            para1.requires_grad_(False)

    net = EF(net1, net2)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.01)

    # optimizer = torch.optim.SGD(params, lr=0.001)
    # CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

    epochs = 100
    best_acc = 0.0
    save_path = './D_R_cat_twocsam_drop05_2_RSSCN7_Merge_2.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        acc1 = 0.0
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # CosineLR.step()

            # print statistics
            running_loss += loss.item()

            # 训练集的准确率
            predict_y = torch.max(logits, dim=1)[1]
            acc1 += torch.eq(predict_y, labels.to(device)).sum().item()
            train_accurate = acc1 / train_num

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        # print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
        #       (epoch + 1, running_loss / train_steps, val_accurate))
        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        with open('./Draw/train_accurate_D_R_cat_twocsam_RSSCN7_Merge_2.txt', 'a+') as f:
            f.write(str(train_accurate) + ',')

        with open('./Draw/val_accurate_D_R_cat_twocsam_RSSCN7_Merge_2.txt', 'a+') as f:
            f.write(str(val_accurate) + ',')

        train_loss1 = running_loss / train_steps
        with open('./Draw/train_loss_D_R_cat_twocsam_RSSCN7_Merge_2.txt', 'a+') as f:
            f.write(str(train_loss1) + ',')

    print("best_acc", best_acc)
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/resnext101_32x8d-8ba56ff5.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    opt = parser.parse_args()

    main(opt)
