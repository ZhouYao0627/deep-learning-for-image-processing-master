from keras.preprocessing.image import ImageDataGenerator
from keras import layers, Input
from keras import models
from keras.layers import Dense, Flatten, Dropout, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
import datetime

from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

starttime = datetime.datetime.now()

import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL

from keras.layers import Input
from keras.layers import Activation, Dense, AveragePooling2D, GlobalAvgPool2D
from keras.models import Model

# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3


# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal',
                         use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)

    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])


# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])


seed = 7
np.random.seed(seed)
model = models.Sequential()

INPUT = Input(shape=(200, 200, 3))

x0 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(INPUT)
x0 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x0)
x0 = MaxPooling2D(pool_size=(2, 2))(x0)

# 第一通道开始
x1 = cbam_module(x0)
x1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)
x1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x1)
x1 = MaxPooling2D(pool_size=(2, 2))(x1)

# 第二通道开始
x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x0)
x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = MaxPooling2D(pool_size=(2, 2))(x2)
x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = MaxPooling2D(pool_size=(2, 2))(x2)
x2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = MaxPooling2D(pool_size=(2, 2))(x2)

# 第3通道开始
x3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x3)
x3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x3)
x3 = MaxPooling2D(pool_size=(2, 2))(x2)
x3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x3)
x3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x3)
x3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x3)
x3 = MaxPooling2D(pool_size=(2, 2))(x3)

x2 = cbam_module(x2)
x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = MaxPooling2D(pool_size=(2, 2))(x2)
x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x2)
x2 = MaxPooling2D(pool_size=(2, 2))(x2)

# 三个通道融合
x = concatenate([x1, x2, x3], axis=-1)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(12, activation='softmax')(x)
model = Model(INPUT, x)
model.summary()  # 打印出模型概述信息。它是utils.print_summary的简捷调用。

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.sgd(lr=0.001, momentum=0.9, decay=0.0002, nesterov=True), metrics=['accuracy'])
# 数据增强
# ImageDataGenerator（
#     rescale=所有数据集将乘以该数值,
#     rotation_range=随即旋转角度数范围,
#     width_shift_range=随即宽度偏移量,
#     height_shift_range=随即高度偏移量,
#     horizontal_flip=是否随机水平翻转,
#     zoom_range=随机缩放的范围 -> [1-n,1+n]）
#     该函数可以增强图片数据，需要fit函数来对指定的数据进行增强，这里要求是四维数据（图片张数，图片长度，图片宽度，灰度），先reshape为四维数据然后调用fit函数e
train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, )  # 图片增强器
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = '/home/tx-lab/dingyue/B/train'
validation_dir = '/home/tx-lab/dingyue/B/validation'

# 类别次序根据文件名称的字母顺序来排列，对于二分类问题，cat文件夹和dog文件夹里面的图片会分别归为第一类和第二类。
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(200, 200), batch_size=36,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(200, 200), batch_size=36,
                                                        class_mode='categorical')

# 官方给出该函数的作用是以一定的频率保存keras模型或参数，通常是和model.compile()、model.fit()结合使用的，可以在训练过程中保存模型，
# 也可以再加载出来训练一般的模型接着训练。具体的讲，可以理解为在每一个epoch训练完成后，可以根据参数指定保存一个效果最好的模型。
MC = keras.callbacks.ModelCheckpoint(filepath='/home/tx-lab/city-planning/degree/B/models/Multichannel1M.h5',
                                     monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False,
                                     mode='auto', period=1)
# 定义学习率之后，经过一定epoch迭代之后，模型效果不再提升，该学习率可能已经不再适应该模型。需要在训练过程中缩小学习率，进而提升模型。
# 使用keras中的回调函keras.callbacks.ReduceLROnPlateau可以实现此效果，
RL = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto',
                                       min_delta=0.000001, cooldown=0, min_lr=0)
# 首先，利用keras，搭建顺序模型，具体搭建步骤省略。完成搭建后，我们需要将数据送入模型进行训练，送入数据的方式有很多种，
# models.fit_generator()是其中一种方式。具体说，model.fit_generator()是利用生成器，分批次向模型送入数据的方式，可以有效节省单次内存的消耗。
history = model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=100,
                              validation_data=validation_generator, validation_steps=None, callbacks=[MC, RL])
model.save('/home/tx-lab/city-planning/degree/B/models/Multichannel1.h5')
with open('/home/tx-lab/city-planning/degree/B/history/Multichannel1.txt', 'w') as f:
    f.write(str(history.history))

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
# 保存图片
# plt.savefig('./result-pdf/FGR-AM1_accuracy.pdf', format='pdf')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.savefig('./result-pdf/FGR-AM1_loss.pdf', format='pdf')
plt.legend()
plt.show()
