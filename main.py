import numpy as np
import h5py
import os
import matplotlib as mpl
print(mpl.__version__)
import matplotlib.pyplot as plt
#plt.style.use('classic')
#import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as plt2
from scipy.stats import gennorm, poisson, norm
import scipy.io
from scipy.io import savemat
from scipy.io import loadmat

from keras.layers import Activation, BatchNormalization, Conv2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.layers import MaxPooling2D, Dropout, Input, AveragePooling2D, Reshape, Permute, UpSampling2D, Conv1D, LeakyReLU
from keras.layers import SimpleRNN, Bidirectional, LSTM
from keras.layers import Lambda
from keras.models import load_model, Model
from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf

from keras.optimizers import *
import keras.backend as K
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
K.set_image_data_format('channels_last')


#from tensorflow.keras.optimizers import SGD
from keras.layers import Bidirectional, LSTM, Reshape, Permute, Conv2D, concatenate, Input, Conv2DTranspose, BatchNormalization, Activation, UpSampling2D, Conv1D
from keras.models import load_model, Model

from cbamModel import cbam

print(keras.__version__)


# QGPU imports

#from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
import pandas as pd
import skimage.measure as sk
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def model():
    input_shape = (256,256,1)

    input = Input(input_shape)
    densenet_model = tf.keras.applications.DenseNet121(include_top=False,
                                                           weights=None,
                                                           input_tensor=input,
                                                           input_shape=input_shape,
                                                           pooling=None,
                                                           classes=1000)

        # SQD-LSTM Block
    x_hor_1 = Reshape((8 * 8, 1024))(densenet_model.layers[-1].output)
    x_ver_1 = Reshape((8 * 8, 1024))(Permute((2, 1, 3))(densenet_model.layers[-1].output))

    h_hor_1 = Bidirectional(LSTM(units=256, activation='tanh', return_sequences=True, go_backwards=False))(x_hor_1)
    h_ver_1 = Bidirectional(LSTM(units=256, activation='tanh', return_sequences=True, go_backwards=False))(x_ver_1)

    H_hor_1 = Reshape((8, 8, 512))(h_hor_1)
    H_ver_1 = Permute((2, 1, 3))(Reshape((8, 8, 512))(h_ver_1))

    c_hor_1 = Conv2D(filters=256, kernel_size=(3, 3),
                         kernel_initializer='he_normal', padding='same')(H_hor_1)
    c_ver_1 = Conv2D(filters=256, kernel_size=(3, 3),
                         kernel_initializer='he_normal', padding='same')(H_ver_1)

    H = concatenate([c_hor_1, c_ver_1])

    y = cbam(H)

        # Decoder Network
        # 8,8,512
    u1 = UpSampling2D(size=(2,2))(y) # 16,16,512
    c1 = Conv2D(256, (3, 3), padding='same', activation='relu')(u1) # 16,16,64
    b1 = BatchNormalization()(c1)
    c2 = Conv2D(256, (3, 3), padding='same', activation='relu')(b1)
    b2 = BatchNormalization()(c2)

    u2 = UpSampling2D(size=(2, 2))(b2)  # 32,32,128
    c3 = Conv2D(128, (3, 3), padding='same', activation='relu')(u2)  # 16,16,64
    b3 = BatchNormalization()(c3)
    c4 = Conv2D(128, (3, 3), padding='same', activation='relu')(b3)
    b4 = BatchNormalization()(c4)
        #
    u3 = UpSampling2D(size=(2, 2))(b4)  # 16,16,128
    c5 = Conv2D(32, (3, 3), padding='same', activation='relu')(u3)  # 16,16,64
    b5 = BatchNormalization()(c5)
    c6 = Conv2D(32, (3, 3), padding='same', activation='relu')(b5)
    b6 = BatchNormalization()(c6)
        #
    u4 = UpSampling2D(size=(2, 2))(b6)  # 16,16,128
    c7 = Conv2D(8, (3, 3), padding='same', activation='relu')(u4)  # 16,16,64
    b7 = BatchNormalization()(c7)
    c8 = Conv2D(8, (1, 1), padding='same', activation='relu')(b7)
    b8 = BatchNormalization()(c8)
        #
    u5 = UpSampling2D(size=(2, 2))(b8)  # 16,16,128
    c9 = Conv2D(1, (1, 1), padding='same', activation='relu')(u5)  # 16,16,64
        # b9 = BatchNormalization()(c9)
    c10 = Conv2D(1, (1, 1), padding='same', activation='leaky_relu')(c9)
        # b10 = BatchNormalization()(c10)
    a10 = LeakyReLU(alpha=0.3)(c10)

    model = Model(inputs=[input], outputs=[a10])
    print(model.summary())
    return model


model = model()
