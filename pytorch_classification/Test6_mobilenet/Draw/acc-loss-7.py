# 模型大小四个实验的对比
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

fr1 = open('/home/tx-lab/city-planning/degree/B/history/VGG16.txt', 'r')
dic1 = eval(fr1.read())  # 读取的str转换为字典
print(dic1)
fr1.close()

fr3 = open('/home/tx-lab/city-planning/degree/B/history/FGR-AM.txt', 'r')
dic3 = eval(fr3.read())  # 读取的str转换为字典
print(dic3)
fr3.close()

fr5 = open('/home/tx-lab/city-planning/degree/B/history/LW-CNNV1.txt', 'r')
dic5 = eval(fr5.read())  # 读取的str转换为字典
print(dic5)
fr5.close()

# fr7 = open('/home/tx-lab/city-planning/A-1/history/VGG16.txt', 'r')
# dic7 = eval(fr7.read())   #读取的str转换为字典
# print(dic7)
# fr7.close()
#
# fr2 = open('/home/tx-lab/city-planning/A-1/history/Goolenet.txt', 'r')
# dic2 = eval(fr2.read())   #读取的str转换为字典
# print(dic2)
# fr2.close()
#
# fr4 = open('/home/tx-lab/city-planning/A-1/history/Inception-resnet_v1.txt', 'r')
# dic4 = eval(fr4.read())   #读取的str转换为字典
# print(dic4)
# fr4.close()
#
#
# fr6 = open('/home/tx-lab/city-planning/A/history/Mobilenet.txt', 'r')
# dic6 = eval(fr6.read())   #读取的str转换为字典
# print(dic6)
# fr6.close()
#
# fr8 = open('/home/tx-lab/city-planning/A/history/Xception.txt', 'r')
# dic8 = eval(fr8.read())   #读取的str转换为字典
# print(dic8)
# fr8.close()
#
# fr9 = open('/home/tx-lab/city-planning/A-1/history/LW-CNNV1.txt', 'r')
# dic9 = eval(fr9.read())   #读取的str转换为字典
# print(dic9)
# fr9.close()


acc1 = dic1['val_accuracy']
acc3 = dic3['val_accuracy']
acc5 = dic5['val_accuracy']
# acc7 = dic7['val_accuracy']
# acc2 = dic2['val_accuracy']
# acc4 = dic4['val_accuracy']
# acc6= dic6['val_accuracy']
# acc8 = dic8['val_accuracy']
# acc9 = dic9['val_accuracy']

loss1 = dic1['val_loss']
loss3 = dic3['val_loss']
loss5 = dic5['val_loss']
# loss7 = dic7['val_loss']
# loss2 = dic2['val_loss']
# loss4 = dic4['val_loss']
# loss6 = dic6['val_loss']
# loss8 = dic8['val_loss']
# loss9 = dic9['val_loss']

epochs = range(1, len(acc1) + 1)

plt.plot(epochs, acc1, '#9bbb59', label='VGG16')
plt.plot(epochs, acc3, '#c0504d', label='FGR-AM')
plt.plot(epochs, acc5, '#8064a2', label='LW-CNN')
# plt.plot(epochs, acc7, '#4bacc6',label='VGG16')
# plt.plot(epochs, acc2, '#4f81bd',label='GooLeNet')
# plt.plot(epochs, acc4,'#9bbb59', label='Inception-ResNet-V1')
# plt.plot(epochs, acc6, '#8064a2',label='MobileNet')
# plt.plot(epochs, acc8, '#4bacc6',label='Xception')
# plt.plot(epochs, acc9, '#c0504d',label='LW-CNN(ours)')
# 颜色选择   #c0504d   #4f81bd #00ECFF  #FFC800 c
plt.ylim(ymin=0)
plt.ylim(ymax=1.0)
plt.title('Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss1, '#9bbb59', label='VGG16')
plt.plot(epochs, loss3, '#c0504d', label='FGR-AM')
plt.plot(epochs, loss5, '#8064a2', label='LW-CNN')
# plt.plot(epochs, loss7, '#0080FF',label='VGG16')
# plt.plot(epochs, loss2,'#FF4D00', label='GooLeNet')
# plt.plot(epochs, loss4,'#B3FF00',label='Inception-ResNet-V1')
# plt.plot(epochs, loss6, '#FFA500',label='MobileNet')
# plt.plot(epochs, loss8, '#4169E1',label='Xception')
# plt.plot(epochs, loss9, 'r',label='LW-CNN(ours)')

plt.title('Loss')
plt.legend()
plt.show()
