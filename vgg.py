import numpy as np
import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Conv2DTranspose, Conv2D, Dropout, Cropping2D, Reshape, Permute, Activation
from keras.models import Model
import tensorflow as tf

def CreateVGG16Model(input_shape, numClasses):
    H, W, C = input_shape
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.get_layer('block4_pool').output
    x = (Conv2D(512 , 3, activation='relu' , padding='same'))(base_model.output)
#     x = Dropout(0.5)(x)
    x = Conv2DTranspose(numClasses, kernel_size=4,  strides=8, use_bias=True, activation='relu')(x)
#     x = Conv2DTranspose(numClasses, kernel_size=4, strides=2, use_bias=True, activation='relu' )(x)
    x = Conv2DTranspose(numClasses, kernel_size=1, strides=2, use_bias=True, activation='relu')(x)
    x = CropToTarget(x, base_model.input, input_shape)(x)
    x = Permute((2,1))(Reshape((-1, H*W))(x))
    x = Activation('softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    print(model.summary())
    return model

def CropToTarget(x, inputs, targetShape):
    model = Model(inputs=inputs, outputs=x)
    currShape = model.output_shape
    diffHeight = int((currShape[1] - targetShape[0])/2)
    diffWidth = int((currShape[2] - targetShape[1])/2)
    print(diffWidth, diffHeight)
    return Cropping2D((diffHeight, diffWidth))
    
   
CreateVGG16Model((240, 320, 3), 8)