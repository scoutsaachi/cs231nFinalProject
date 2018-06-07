import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.layers import LeakyReLU, MaxPooling2D, Input, Dense, Conv2DTranspose, Conv2D, Dropout, Cropping2D, Reshape, Permute, Activation, Flatten, BatchNormalization
from keras.models import Model, Sequential
import tensorflow as tf
from keras import regularizers
import sys
sys.path.insert(1, "crfasrnn_keras/src/")
from crfasrnn_keras.src.crfrnn_layer import CrfRnnLayer

def AddCRF_RNNToModel(model, dims, num_classes, freeze_model, outputLayer): # dims are just height and weight
    inputs = model.input
    outputs = model.get_layer(outputLayer).output
    H, W = dims
    if freeze_model:
        for layer in model.layers:
            layer.trainable = False
    x = CrfRnnLayer(image_dims=(H, W),
                         num_classes=num_classes,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([outputs, inputs])
    x = Permute((2,1))(Reshape((-1, H*W))(x))
    x = Activation('softmax')(x)
    newmodel = Model(inputs=inputs, outputs=x)
    print(newmodel.output_shape)
    return newmodel
    

def CreateVGG16Model(input_shape, numClasses, regParam):
    H, W, C = input_shape
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = Conv2D(filters=1024, kernel_size=7, activation='relu', padding='same',
               name='fc6', use_bias=True, kernel_regularizer=regularizers.l2(regParam))(base_model.output)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=1024, kernel_size=1, activation='relu', padding='valid',
               name='fc7', use_bias=True, kernel_regularizer=regularizers.l2(regParam))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(filters=numClasses, kernel_size=1, padding='valid', name='score',
               kernel_regularizer=regularizers.l2(regParam))(x)
    x = Conv2DTranspose(numClasses, kernel_size=6, strides=4, use_bias=False,
                        name='upscore', kernel_regularizer=regularizers.l2(regParam))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(numClasses, kernel_size=16, strides=8, name="finalscore",
                        kernel_regularizer=regularizers.l2(regParam))(x)
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
    return Cropping2D((diffHeight, diffWidth), name="Cropping")

def CreateConvBlock(filters, num_convs, index, input_shape=None):
    layers = []
    for i in range(num_convs):
        if i==0 and input_shape is not None:
            layers.append(Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same', name='cnn%s_%s' % (i, index), input_shape=input_shape))
        else:
            layers.append(Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same', name='cnn%s_%s' % (i, index)))
    layers.append(MaxPooling2D(pool_size=4, strides=2, padding='same', name="pool_%s" % index))
    return layers
    
def CreateSimpleFCN(input_shape, numClasses, regParam):
    print(input_shape)
    H, W, C = input_shape
    layers = []
    layers += CreateConvBlock(64, 1, 1, input_shape)
    layers += CreateConvBlock(128, 1, 2)
    layers += CreateConvBlock(256, 1, 3)
    layers += CreateConvBlock(256, 1, 5)
    layers += [
        Conv2D(filters=1024, kernel_size=7, activation='relu', padding='same', name='fc6'),
        Dropout(0.5),
        Conv2DTranspose(numClasses, kernel_size=4, strides=4, padding='same', activation='relu', use_bias=True),
        BatchNormalization(),
        Conv2DTranspose(numClasses, kernel_size=4, strides=4, use_bias=True),
    ]
    model = Sequential(layers)
    model.add(CropToTarget(model.output_shape, input_shape))
    model.add(Reshape((-1, H*W)))
    model.add( Permute((2,1)))
    model.add( Activation('softmax'))
    print(model.summary())
    return model



m = CreateVGG16Model((240, 320, 3), 8, 0.001)
newm = AddCRF_RNNToModel(m, (240, 320,), 8, False, "Cropping")
print(newm.summary())