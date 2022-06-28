# 混淆矩阵
from keras.models import load_model
import itertools
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

starttime = datetime.datetime.now()

plt.figure(None, (24, 24), 300)  # 绘图

# 标签名称
labels_name = ['Airport', 'Beach', 'Bridge', 'Commercial', 'Desert', 'Farmland', 'footballField', 'Forest',
               'Industrial', 'Meadow', 'Mountain', 'Park', 'Parking', 'Pond', 'Port', 'railwayStation', 'Residential',
               'River', 'Viaduct']

validation_dir = '../../../data_set/RS_19/train'  # 数据集中训练集
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(600, 600), batch_size=1,
                                                        class_mode='categorical', shuffle=False)

model = load_model('../Multichannel2.h5')  # 模型的权重
# 输出准确率和损失值
h = model.evaluate_generator(validation_generator)
print(h)
endtime = datetime.datetime.now()
print((endtime - starttime))

k = 0
y_pre = []
y_true = []
for data_batch, label_batch in validation_generator:
    k += 1
    # K＜验证集数据
    if k <= 1920:
        x = data_batch
        pre = model.predict(x)
        y1 = np.argmax(pre, axis=1)
        y1 = int(y1)
        y_pre.append(y1)
        y = label_batch
        y2 = np.argmax(label_batch, axis=1)
        y2 = int(y2)
        y_true.append(y2)
    if k == 1920:
        break
print(y_pre)
print(y_true)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy Score:", accuracy_score(y_true, y_pre, normalize=True))
print("Precision Score", precision_score(y_true, y_pre, average='macro'))
print("Recall Score", recall_score(y_true, y_pre, average='macro'))
print("F1 Score", f1_score(y_true, y_pre, average='macro'))

cm = confusion_matrix(y_true, y_pre)
print(cm)

Normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(Normalized_cm)


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.3f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('cm.png', format='png')


plot_confusion_matrix(cm, classes=labels_name)
