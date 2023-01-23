# -*- coding: utf-8 -*-

from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
import utils

K.set_image_data_format('channels_first')


import warnings
warnings.filterwarnings("ignore")

def VGG16(weights_path = None, input_shape = (3, 224, 224)):

    # Determine proper input shape
    img_input = Input(shape=input_shape)

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', padding='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', padding='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', padding='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x)

    # Load pre-trained weights if available
    if weights_path:
        model.load_weights(weights_path)

    return model


if __name__ == '__main__':

    weights_path = utils.DATA_DIR + utils.WEIGHTS_FILE
    model = VGG16(weights_path)
    #model = tf.keras.applications.vgg16.VGG16()
    print(model.summary())
