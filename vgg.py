import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.layers import LeakyReLU, MaxPooling2D, Input, Dense, Conv2DTranspose, Conv2D, Dropout, Cropping2D, Reshape, Permute, Activation, Flatten, BatchNormalization
from keras.models import Model, Sequential
import tensorflow as tf


def CreateVGG16Model(input_shape, numClasses):
    H, W, C = input_shape
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = Conv2D(filters=1024, kernel_size=7, activation='relu', padding='same', name='fc6')(base_model.output)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=1024, kernel_size=1, activation='relu', padding='valid', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=numClasses, kernel_size=1, padding='valid', name='score')(x)
    x = Conv2DTranspose(numClasses, kernel_size=6, strides=4, use_bias=False, name='upscore')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(numClasses, kernel_size=16, strides=8, name="finalscore")(x)
    model = Model(inputs=base_model.input, outputs=x)
    currShape = model.output_shape
    x = CropToTarget(currShape, input_shape)(x)
    x = Permute((2,1))(Reshape((-1, H*W))(x))
    x = Activation('softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    print(model.summary())
    return model

def CropToTarget(currShape, targetShape):
    diffHeight = int((currShape[1] - targetShape[0])/2)
    diffWidth = int((currShape[2] - targetShape[1])/2)
    print(diffWidth, diffHeight)
    return Cropping2D((diffHeight, diffWidth))

def CreateConvBlock(filters, num_convs, index):
    layers = []
    for i in range(num_convs):
        layers.append(Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same', name='cnn%s_%s' % (i, index)))
    layers.append(MaxPooling2D(pool_size=2, strides=2, padding='same', name="pool_%s" % index))
    return layers
    
def CreateSimpleFCN(input_shape):
    print(input_shape)
    H, W, C = input_shape
    layers = []
    layers += CreateConvBlock(64, 2, 1)
    layers += CreateConvBlock(128, 2, 2)
    layers += CreateConvBlock(256, 3, 3)
    layers += CreateConvBlock(512, 3, 4)
    layers += CreateConvBlock(512, 3, 5)
    layers += [
        Flatten(),
        Dense(1024, use_bias=True, activation='relu')
    
    model = Sequential([
        Conv2D(32, kernel_size=5, strides=1, padding='same', use_bias=True, activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2, padding="same"),
        Conv2D(64, kernel_size=5, strides=1, padding='same', use_bias=True, activation='relu'),
        MaxPooling2D(4,4, padding="same"),
        Flatten(),
        Dense(2048, use_bias=True, activation='relu'),
#         BatchNormalization(),
#         Dense(2016, use_bias=True, activation='relu'),
#         BatchNormalization(),
#         Reshape((8,8,32)),
        Conv2DTranspose(64, kernel_size=4, strides=2, activation='relu', use_bias=True),
        BatchNormalization(),
        Conv2DTranspose(32, kernel_size=16, strides=2, activation='relu', use_bias=True),
        BatchNormalization(),
        Conv2DTranspose(1, kernel_size=1, strides=4, activation='sigmoid', use_bias=True),
    ])
    model.add(CropToTarget(model.output_shape, input_shape))
    model.add(Flatten())
#     model.add(Reshape((-1, H*W)))
#     model.add( Permute((2,1)))
#     model.add( Activation('softmax'))
    print(model.summary())
    return model


# CreateVGG16Model((240, 320, 3), 8)