import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse

from model_resnet import resnet50, resnext101_32x8d
from model_densenet import densenet121, load_state_dict1, load_state_dict2, densenet161
from model_convnext import convnext_base
from CBAM import cbam


class bilinear_pooling():
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_size = x.size()
        y_size = y.size()

        assert (x_size[:-1] == y_size[:-1])

        out_size = list(x_size)
        out_size[-1] = x_size[-1] * y_size[-1]  # 特征x和特征y维数之积

        x = x.view([-1, x_size[-1]])  # [N*C,F]
        y = y.view([-1, y_size[-1]])

        out_stack = []
        for i in range(x.size()[0]):
            out_stack.append(torch.ger(x[i], y[i]))  # torch.ger()向量的外积操作

        out = torch.stack(out_stack)  # 将list堆叠成tensor

        return out.view(out_size)  # [N,C,F*F]


# def bilinear_pooling(x, y):
#     x_size = x.shape
#     y_size = y.shape
#
#     assert (x_size[:-1] == y_size[:-1])
#
#     out_size = list(x_size)
#     out_size[-1] = x_size[-1] * y_size[-1]  # 特征x和特征y维数之积
#
#     x = x.view([-1, x_size[-1]])  # [N*C,F]
#     y = y.view([-1, y_size[-1]])
#
#     out_stack = []
#     for i in range(x.size()[0]):
#         out_stack.append(torch.ger(x[i], y[i]))  # torch.ger()向量的外积操作
#
#     out = torch.stack(out_stack)  # 将list堆叠成tensor
#
#     return out.view(out_size)  # [N,C,F*F]


class EF(nn.Module):

    def __init__(self, net1, net2, bilinear_pooling):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

        # self.cbam = cbam(3072)
        self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, bias=False)
        self.bilinear_pooling = bilinear_pooling()

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Sequential(nn.Linear(1024, 30))

    def forward(self, input):
        f1 = self.net1(input)  # [64, 2048, 7, 7]
        f2 = self.net2(input)  # [64, 1024, 7, 7]

        f1 = self.conv1(f1)  # [64, 1024, 7, 7]

        # x = self.cbam(x)
        x = self.bilinear_pooling.forward(f1, f2)
        # print(x.shape)  #torch.Size([64, 1024, 7, 49])
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(x.shape)  # torch.Size([64, 1024])
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
    image_path = os.path.join(data_root, "data_set", "AID30", "50_50")  # flower data set path
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

    batch_size = 16
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

    net1 = resnet50(include_top=False)
    # net1 = convnext_base(1000)
    net2 = densenet121()

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

    model_weight_path2 = "./weights/densenet121-a639ec97.pth"
    assert os.path.exists(model_weight_path2), "file {} does not exist.".format(model_weight_path2)
    load_state_dict1(net2, "./weights/densenet121-a639ec97.pth")
    # net2.load_state_dict(torch.load(model_weight_path2, map_location="cpu"))
    # 是否冻结权重
    for name1, para1 in net2.named_parameters():
        # 除最后的全连接层外，其他权重全部冻结
        if "classifier" not in name1:
            para1.requires_grad_(False)

    net = EF(net1, net2, bilinear_pooling)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    epochs = 100
    best_acc = 0.0
    save_path = './D_R_onecbam_cat.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

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

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        with open('./Draw/val_accurate_D_R_onecbame_cat.txt', 'a+') as f:
            f.write(str(val_accurate) + ',')

        train_loss1 = running_loss / train_steps
        with open('./Draw/train_loss_D_R__onecbam_cat.txt', 'a+') as f:
            f.write(str(train_loss1) + ',')

    print("best_acc", best_acc)
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/resnet50-19c8e357.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    opt = parser.parse_args()

    main(opt)
