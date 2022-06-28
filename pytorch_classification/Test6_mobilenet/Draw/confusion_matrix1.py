import torch
import matplotlib.pyplot as plt
import itertools
import numpy as np
from pytorch_classification.Test6_mobilenet.model_v2 import MobileNetV2
import os
from torchvision import transforms, datasets
from tensorflow.python.platform import analytics

batch_size = 16

data_transform = {
    "validation": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../"))  # get data root path
image_path = os.path.join(data_root, "data_set", "RS_19")  # flower data set path

nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "validation"),
                                        transform=data_transform["validation"])
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=nw)

# 分类模型测试阶段代码

# 创建一个空矩阵存储混淆矩阵
conf_matrix = torch.zeros(19, 19)
for batch_images, batch_labels in validate_loader:
    # print(batch_labels)
    with torch.no_grad():
        if torch.cuda.is_available():
            batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

    out = MobileNetV2(batch_images)

    prediction = torch.max(out, 1)[1]
    conf_matrix = analytics.confusion_matrix(prediction, labels=batch_labels, conf_matrix=conf_matrix)

# conf_matrix需要是numpy格式
# attack_types是分类实验的类别，eg：attack_types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
analytics.plot_confusion_matrix(conf_matrix.numpy(),
                                classes=['Airport', 'Beach', 'Bridge', 'Commercial', 'Desert', 'Farmland',
                                         'footballField', 'Forest',
                                         'Industrial', 'Meadow', 'Mountain', 'Park', 'Parking', 'Pond', 'Port',
                                         'railwayStation', 'Residential',
                                         'River', 'Viaduct'],
                                normalize=False,
                                title='Normalized confusion matrix')


# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    '''

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
