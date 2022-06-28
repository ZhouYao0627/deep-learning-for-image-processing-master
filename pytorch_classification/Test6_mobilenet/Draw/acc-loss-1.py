# 模型大小四个实验的对比
import matplotlib.pyplot as plt

fr0 = open('/home/tx-lab/city-planning/degree/A/history/Multichannel1.txt', 'r')
dic0 = eval(fr0.read())  # 读取的str转换为字典
print(dic0)
fr0.close()

fr1 = open('/home/tx-lab/city-planning/degree/A/history/Multichannel2.txt', 'r')
dic1 = eval(fr1.read())  # 读取的str转换为字典
print(dic1)
fr1.close()

acc0 = dic0['val_accuracy']
acc1 = dic1['val_accuracy']

loss0 = dic0['val_loss']
loss1 = dic1['val_loss']

epochs = range(1, len(acc0) + 1)

plt.plot(epochs, acc0, 'b:', label='Multichannel1')
plt.plot(epochs, acc1, 'r', label='Multichannel2')

plt.ylim(ymin=0)
plt.ylim(ymax=1.0)

plt.title('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss0, 'b:', label='Multichannel1')
plt.plot(epochs, loss1, 'r', label='Multichannel2')

plt.title('Loss')
plt.legend()
plt.show()
