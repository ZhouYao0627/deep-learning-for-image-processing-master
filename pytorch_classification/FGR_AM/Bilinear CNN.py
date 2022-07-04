# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 00:28:01 2018
@author: Administrator
"""

import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, Input, Reshape, Lambda
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau


def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)


def l2_norm(x):
    return K.l2_normalize(x, axis=-1)


def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])


def bilinearnet():
    input_tensor = Input(shape=(384, 512, 3))
    vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    #    conv2048 = Convolution2D(filters=2048,kernel_size=(3,3),)
    #    vgg16_add_conv_to_2048 = Model(inputs=input_tensor,outputs=)
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    model_vgg16 = Model(inputs=input_tensor, outputs=vgg16.output)
    model_resnet50 = Model(inputs=input_tensor, outputs=resnet50.output)
    model_vgg16.compile(loss='categorical_crossentropy', optimizer='adam')
    model_resnet50.compile(loss='categorical_crossentropy', optimizer='adam')

    resnet50_x = Reshape([model_resnet50.layers[-6].output_shape[1] * model_resnet50.layers[-6].output_shape[2],
                          model_resnet50.layers[-6].output_shape[3]])(model_resnet50.layers[-6].output)
    vgg16_x = Reshape([model_vgg16.layers[-1].output_shape[1] * model_vgg16.layers[-1].output_shape[2],
                       model_vgg16.layers[-1].output_shape[3]])(model_vgg16.layers[-1].output)

    cnn_dot_out = Lambda(batch_dot)([vgg16_x, resnet50_x])

    sign_sqrt_out = Lambda(sign_sqrt)(cnn_dot_out)
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)
    flatten = Flatten()(l2_norm_out)
    dropout = Dropout(0.5)(flatten)
    output = Dense(12, activation='softmax')(dropout)

    model = Model(input_tensor, output)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-6),
                  metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='vgg_resnet_bilinear_model.png')
    return model
    #    print(vgg16_x.shape)
    #    print(resnet50_x.shape)
    #    print(cnn_dot_out.shape)
    #    print(model_vgg16.summary())
    '''
    vgg16

    block5_conv1 (Conv2D)        (None, 24, 32, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 24, 32, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 24, 32, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 12, 16, 512)       0         

    resnet50

    bn5c_branch2c (BatchNormalizati (None, 12, 16, 2048) 8192        res5c_branch2c[0][0]             
    __________________________________________________________________________________________________
    add_112 (Add)                   (None, 12, 16, 2048) 0           bn5c_branch2c[0][0]              
                                                                     activation_340[0][0]             
    __________________________________________________________________________________________________
    activation_343 (Activation)     (None, 12, 16, 2048) 0           add_112[0][0]                    
    __________________________________________________________________________________________________
    '''

#    print(model_resnet50.summary())
#    vgg16.layers[]


#
# def categorical_crossentropy(y_true, y_pred):
#    return K.categorical_crossentropy(y_true, y_pred)
#
#
# model = VGG16(weights='imagenet', include_top=False,input_shape=(384, 512,3))
##print('youdianmeng')
# top_model = Sequential()
# top_model.add(Flatten(input_shape=model.output_shape[1:]))  # model.output_shape[1:])
# top_model.add(Dropout(0.5))
# top_model.add(Dense(12, activation='softmax'))
## 载入上一模型的权重
#
# ftvggmodel = Model(inputs=model.input, outputs=top_model(model.output))
##for layer in ftvggmodel.layers[:25]:
##    layer.trainable=True
#
# ftvggmodel.compile(loss=categorical_crossentropy,
#              optimizer=optimizers.SGD(lr=1e-4, momentum=0.90,decay=1e-5),metrics=['accuracy'])
##
# train_data_gen = ImageDataGenerator(rescale=1 / 255.,
#                                     samplewise_center=True,
#                                     samplewise_std_normalization=True,
#                                     #                 zca_whitening=True,
#                                     #                 zca_epsilon=1e-6,
#
#                                     width_shift_range=0.05,
#                                     height_shift_range=0.05,
#                                     fill_mode='reflect',
#                                     horizontal_flip=True,
#                                     vertical_flip=True)
#
# test_data_gen = ImageDataGenerator(rescale=1 / 255.)
# #
# train_gen = train_data_gen.flow_from_directory(directory='D:\\xkkAI\\ZZN\\guangdong\\train',
#                                                target_size=(384, 512), color_mode='rgb',
#                                                class_mode='categorical',
#                                                batch_size=5, shuffle=True, seed=222
#                                                )
#
# val_gen = test_data_gen.flow_from_directory(directory='D:\\xkkAI\\ZZN\\guangdong\\val',
#                                             target_size=(384, 512), color_mode='rgb',
#                                             class_mode='categorical',
#                                             batch_size=5, shuffle=True, seed=222
#                                             )
# test_gen = test_data_gen.flow_from_directory(directory='D:\\xkkAI\\ZZN\\guangdong\\test',
#                                              target_size=(384, 512), color_mode='rgb',
#                                              class_mode='categorical',
#                                              batch_size=5
#                                              )
# cp = ModelCheckpoint('guangdong_best_vgg16.h5', monitor='val_loss', verbose=1,
#                      save_best_only=True, save_weights_only=False,
#                      mode='auto', period=1)
# es = EarlyStopping(monitor='val_loss',
#                    patience=8, verbose=1, mode='auto')
# lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=1e-5, patience=2, verbose=1, min_lr=0.00000001)
# callbackslist = [cp, es, lr_reduce]
#
# ftvggmodel = bilinearnet()
# ftvggmodel.fit_generator(train_gen,
#                          epochs=1111,
#                          verbose=1,
#                          callbacks=callbackslist,
#                          validation_data=val_gen,
#                          shuffle=True)
#
# # ftvggmodel.load_weights('guangdong_best_vgg16.h5')
# pred = ftvggmodel.predict_generator(test_gen)
#
# defectlist = ['norm', 'defect1', 'defect2', 'defect3', 'defect4', 'defect5', 'defect6', 'defect7', 'defect8', 'defect9',
#               'defect10', 'defect11']
# import csv
#
# with open('lvcai_result.csv', 'w') as f:
#     w = csv.writer(f)
#     for i in range(len(pred)):
#         w.writerow([str(i) + '.jpg', defectlist[np.argmax(pred[i])]])
