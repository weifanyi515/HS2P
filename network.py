import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, Concatenate, Activation, Lambda, Add, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, multiply, Permute
from keras.models import Model, Input
import numpy as np

K.set_image_data_format('channels_first')


def Get_gradient(input_l, layername):

    def filter_v(shape, dtype=None):
        ker = np.zeros(shape, dtype=dtype) 
        ker[0, 1] = -1 
        ker[2, 1] = 1
        return ker
    
    def filter_h(shape, dtype=None):
        ker = np.zeros(shape, dtype=dtype) 
        ker[1, 0] = -1 
        ker[1, 2] = 1
        return ker

    tmp_v = Conv2D(13, kernel_size = 3, kernel_initializer=filter_v, strides=1, padding='same', name='get_gradv' + layername)(input_l)
    tmp_h = Conv2D(13, kernel_size = 3, kernel_initializer=filter_h, strides=1, padding='same', name='get_gradh' + layername)(input_l)
    tmp_v = Lambda(lambda x: tf.square(x))(tmp_v)
    tmp_h = Lambda(lambda x: tf.square(x))(tmp_h)
    tmpvh = Add()([tmp_v, tmp_h])
    tmpvh = Lambda(lambda x: tf.sqrt(x + 1e-6))(tmpvh)

    return tmpvh


def Attention(input_l, feature_size, reduction=256):
    '''Attention module'''

    layer_one = Dense(feature_size // reduction,
                        kernel_initializer='he_normal',
                        activation='relu',
                        use_bias=True,
                        bias_initializer='zeros')

    layer_two = Dense(feature_size,
                        kernel_initializer='he_normal',
                        use_bias=True,
                        bias_initializer='zeros')
    x = input_l
    avg_pool = GlobalAveragePooling2D()(x)
    avg_pool = Reshape((1, 1, feature_size))(avg_pool)
    avg_pool = layer_one(avg_pool)
    avg_pool = layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(x)
    max_pool = Reshape((1, 1, feature_size))(max_pool)
    max_pool = layer_one(max_pool)
    max_pool = layer_two(max_pool)

    weight = Add()([max_pool, avg_pool])
    weight = Activation('hard_sigmoid')(weight)
    weight = Permute((3, 1, 2))(weight)

    x = multiply([input_l, weight])

    return x


def resBlock(input_l, feature_size, reduction=256): 
    """ResBlock"""

    x = input_l
    x = Conv2D(feature_size, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(feature_size, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)

    x = Attention(x, feature_size)

    return Add()([input_l, x])


def resGroup(input_l, feature_size, block_num=4):
    """ResGroup"""

    x = input_l
    for i in range(block_num):
        x = resBlock(x, feature_size)  
    x = Conv2D(feature_size, kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)

    return Add()([input_l, x])


def data_fusion(input_l, feature_size):
    '''Data fusion module'''
    x = Conv2D(feature_size, kernel_size=3, kernel_initializer='he_uniform', padding='same')(input_l)
    x = Activation('relu')(x)
    x = Attention(x, feature_size)

    return x


def HS2P_model(input_shape, N, feature_size=256):
    """Architecture of HS2P """

    input_opt = Input(shape=input_shape[0])    #input data
    input_sar = Input(shape=input_shape[1])

    ref = Input(shape=input_shape[2])    #caculate the gradient map of target
    ref_grad = Get_gradient(ref, 'tar')


    x = Concatenate(axis=1)([input_opt, input_sar])
    x = data_fusion(x, feature_size=256)

    for i in range (N):    #the stacked resgroups
        x = resGroup(x, feature_size)

        feature_tmp = Conv2D(input_shape[0][0], kernel_size=3, kernel_initializer='he_uniform', padding='same')(x)
        feature_grad_tmp = Get_gradient(feature_tmp, 'fea'+str(i+1))

        if(i == 0):
            feature = feature_tmp
            feature_grad = feature_grad_tmp
        else:
            feature = Concatenate(axis=1)([feature, feature_tmp])
            feature_grad = Concatenate(axis=1)([feature_grad, feature_grad_tmp])


    X = Concatenate(axis=1)([feature, feature_grad, ref_grad])

    model = Model(inputs=[input_opt, input_sar, ref], outputs=X)

    return model
