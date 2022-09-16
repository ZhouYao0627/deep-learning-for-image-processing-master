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


# UCM21_80_20_无dropout: best_acc: 0.9738095238095238
# UCM21_80_20_1dropout: best_acc: 0.969047619047619
# UCM21_80_20_2dropout: best_acc: 0.9761904761904762
# UCM21_90_10_无dropout: best_acc: 0.9904761904761905
# UCM21_90_10_2dropout: best_acc: 0.9857142857142858

# RSSCN7_50_50_2dropout: best_acc: 0.921
# RSSCN7_50_50_0dropout: best_acc: 0.9192857142857143
# RSSCN7_80_20_2dropout: best_acc:
# RSSCN7_80_20_0dropout: best_acc:

# AID30_80_20_2dropout: best_acc: 0.9355
# AID30_80_20_0dropout: best_acc:


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    image_path = os.path.join(data_root, "data_set", "AID30", "80_20")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
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
        # print(weights_dict.keys())
        for k in list(weights_dict.keys()):
            if "fc.weight" in k or "fc.bias" in k:
                del weights_dict[k]
        print("successfully load pretrain-weights1", net1.load_state_dict(weights_dict, strict=False))

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
    loss_function1 = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    # optimizer = torch.optim.SGD(params, lr=0.001)
    # CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

    epochs = 10
    best_acc = 0.0
    save_path = './save_weights/D_R_AID30_8020_Merge_new_CM.pth'
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
            loss = loss_function1(logits, labels.to(device))
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

        with open('./Draw/train_accurate_AID30_8020_Merge_new_2dropout_CM.txt', 'a+') as f:
            f.write(str(train_accurate) + ',')

        with open('./Draw/val_accurate_AID30_8020_Merge_new_2dropout_CM.txt', 'a+') as f:
            f.write(str(val_accurate) + ',')

        train_loss1 = running_loss / train_steps
        with open('./Draw/train_loss_AID30_8020_Merge_new_2dropout_CM.txt', 'a+') as f:
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
