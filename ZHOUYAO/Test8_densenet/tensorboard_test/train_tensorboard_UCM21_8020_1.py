import os
import math
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse

from model import resnet34
from my_dataset import MyDataSet
from data_utils import read_split_data, plot_class_preds
from train_eval_utils import train_one_epoch, evaluate

from model_resnet import resnet50, resnext50_32x4d, resnext101_32x8d, resnet101
from model_densenet import densenet121, load_state_dict1, densenet161, densenet201
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
        self.fc = nn.Sequential(nn.Linear(4256, 21))

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


def main(args):
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir="runs/UCM21_8020_experiment")
    # if os.path.exists("../weights") is False:
    #     os.makedirs("./weights")

    # 划分数据为训练集和验证集
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    # 定义训练以及预测时的预处理方法
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # # 实例化训练数据集
    # train_data_set = MyDataSet(images_path=train_images_path, images_class=train_images_label,
    #                            transform=data_transform["train"])
    #
    # # 实例化验证数据集
    # val_data_set = MyDataSet(images_path=val_images_path, images_class=val_images_label,
    #                          transform=data_transform["val"])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "UCM21", "80_20")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices_UCM21.json', 'w') as json_file:
        json_file.write(json_str)

    # batch_size = args.batch_size
    batch_size = 64

    # 计算使用num_workers的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # train_loader = torch.utils.data.DataLoader(train_data_set,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            pin_memory=True,
    #                                            num_workers=nw,
    #                                            collate_fn=train_data_set.collate_fn)
    #
    # val_loader = torch.utils.data.DataLoader(val_data_set,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          pin_memory=True,
    #                                          num_workers=nw,
    #                                          collate_fn=val_data_set.collate_fn)
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

    # 实例化模型
    # model = resnet34(num_classes=args.num_classes).to(device)

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

    model = EF(net1, net2)
    model.to(device)

    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 256, 256), device=device)
    tb_writer.add_graph(model, init_img)

    # # 如果存在预训练权重则载入
    # if os.path.exists(args.weights):
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     load_weights_dict = {k: v for k, v in weights_dict.items()
    #                          if model.state_dict()[k].numel() == v.numel()}
    #     model.load_state_dict(load_weights_dict, strict=False)
    # else:
    #     print("not using pretrain-weights.")

    # # 是否冻结权重
    # if args.freeze_layers:
    #     print("freeze layers except fc layer.")
    #     for name, para in model.named_parameters():
    #         # 除最后的全连接层外，其他权重全部冻结
    #         if "fc" not in name:
    #             para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device,
                                    epoch=epoch)
        # update learning rate
        scheduler.step()

        # validate
        acc = evaluate(model=model, data_loader=validate_loader, device=device)

        # add loss, acc and lr into tensorboard
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["train_loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # add figure into tensorboard
        fig = plot_class_preds(net=model,
                               images_dir="./plot_img",
                               transform=data_transform["val"],
                               num_plot=5,
                               device=device)
        if fig is not None:
            tb_writer.add_figure("predictions vs. actuals",
                                 figure=fig,
                                 global_step=epoch)

        # add conv1 weights into tensorboard
        tb_writer.add_histogram(tag="conv1",
                                values=model.conv1.weight,
                                global_step=epoch)
        tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=model.layer1[0].conv1.weight,
                                global_step=epoch)

        # save weights
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    img_root = "/home/wz/my_project/my_github/data_set/flower_data/flower_photos"
    parser.add_argument('--data-path', type=str, default=img_root)

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='resNet34.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
