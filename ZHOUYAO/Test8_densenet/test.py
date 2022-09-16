import os
import json
import sys
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch

from model_resnet import resnet50
from model_densenet import densenet121


class mergeblock(nn.Module):
    def __init__(self, InData1, InData2):
        super().__init__()
        self.InData1 = InData1
        self.InData2 = InData2
        self.conv = nn.Sequential(nn.Conv2d(2048, 1, kernel_size=(1, 1), stride=(1, 1), padding="same"))

    def forward(self):
        x1 = self.conv(self.InData2)
        x2 = self.conv(self.InData2)
        result1 = torch.matmul(self.InData1, x1)
        result2 = torch.add(result1, x2)
        return result2


class EF(nn.Module):

    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Sequential(nn.Linear(3072, 21))

    def forward(self, input):
        f1 = self.net1(input)
        f2 = self.net2(input)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # c = mergeblock(f1, f2)
        # x = c.forward()
        # x = torch.add(f1, f2)
        x = torch.cat([f1, f2], dim=1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "validation": transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    image_path = os.path.join(data_root, "data_set", "UCM")  # flower data set path
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

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "validation"),
                                            transform=data_transform["validation"])  # 364
    val_num = len(validate_dataset)  # 364
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=nw)  # 23

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net1 = resnet50(num_classes=21, include_top=False)
    net2 = densenet121(num_classes=21)
    # 使用了迁移学习 59-70
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = "./resnet34-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    # in_channel = net.fc.in_features  # 使用的是34的权重，故in_channel = 512*1 = 512
    # net.fc = nn.Linear(in_channel, 5)
    # net1.to(device)
    # net2.to(device)
    net = EF(net1, net2)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001)

    epochs = 1  # 原来是3
    best_acc = 0.0
    save_path = './AIMAX_Densenet121_resnet50.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        net.to(device)
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

        with open('./Draw/val_accurate_D_R.txt', 'a+') as f:
            f.write(str(val_accurate) + ',')

        train_loss1 = running_loss / train_steps
        with open('./Draw/train_loss_D_R.txt', 'a+') as f:
            f.write(str(train_loss1) + ',')

    print("best_acc", best_acc)
    print('Finished Training')


if __name__ == '__main__':
    main()