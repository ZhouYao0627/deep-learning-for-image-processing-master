import os
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse

from model import densenet121, load_state_dict


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
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)  # 3306

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers  -> 4
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=nw)  # 207

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])  # 364
    val_num = len(validate_dataset)  # 364
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=nw)  # 23

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = densenet121(num_classes=30)
    net.to(device)

    if args.weights != "":
        if os.path.exists(args.weights):
            load_state_dict(net, args.weights)
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # # 是否冻结权重
    # if args.freeze_layers:
    #     for name, para in net.named_parameters():
    #         # 除最后的全连接层外，其他权重全部冻结
    #         if "classifier" not in name:
    #             para.requires_grad_(False)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 100  # 原来是3
    best_acc = 0.0
    save_path = './densenet121_weight.pth'
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

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        with open('./Draw/val_accurate_densenet_weight.txt', 'a+') as f:
            f.write(str(val_accurate) + ',')

        train_loss1 = running_loss / train_steps
        with open('./Draw/train_loss_densenet_weight.txt', 'a+') as f:
            f.write(str(train_loss1) + ',')

    print("best_acc", best_acc)
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 预训练权重路径，如果不想载入就设置为空字符
    # 链接: https://pan.baidu.com/s/1aNqQW4n_RrUlWUBNlaJRHA  密码: i83t
    parser.add_argument('--weights', type=str, default='./densenet121-a639ec97.pth',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
