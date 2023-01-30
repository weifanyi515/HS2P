import keras.backend as K
import tensorflow as tf
import numpy as np

# Normalize your data to a range of [0, 1] before the quantative evaluation
def MAE(y_true, y_pred):
    return K.mean(K.abs(y_pred[:, 39:52, :, :] - y_true[:, 0:13, :, :]))


def RMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred[:, 39:52, :, :] - y_true[:, 0:13, :, :])))


def PSNR(y_true, y_pred):
    y_true *= 2000
    y_pred *= 2000
    rmse = K.sqrt(K.mean(K.square(y_pred[:, 39:52, :, :] - y_true[:, 0:13, :, :])))

    return 20.0 * (K.log(10000.0 / rmse) / K.log(10.0))


def get_sam(y_true, y_pred):
    mat = tf.multiply(y_true, y_pred)
    mat = tf.reduce_sum(mat, 1)
    mat = tf.div(mat, K.sqrt(tf.reduce_sum(tf.multiply(y_true, y_true), 1)))
    mat = tf.div(mat, K.sqrt(tf.reduce_sum(tf.multiply(y_pred, y_pred), 1)))
    mat = tf.acos(K.clip(mat, -1, 1))

    return mat


def SAM(y_true, y_pred):
    mat = get_sam(y_true[:, 0:13, :, :], y_pred[:, 39:52, :, :])

    return tf.reduce_mean(mat)


def SSIM(y_true, y_pred):
    y_true = y_true[:, 0:13, :, :]
    y_pred = y_pred[:, 39:52, :, :]

    y_true *= 2000
    y_pred *= 2000

    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])

    ssim = tf.image.ssim(y_true, y_pred, max_val=10000.0)
    ssim = tf.reduce_mean(ssim)

    return ssim


def validation_metric(y_true, y_pred):

    PSNR_tmp = PSNR(y_true, y_pred)
    SSIM_tmp = SSIM(y_true, y_pred)

    return PSNR_tmp + 10.0 * SSIM_tmp


def loss(y_true, y_pred):
    b = 0.50
    n = 4

    target = y_true[:, 0:13, :, :]
    tar_grad = y_pred[:, 104:117, :, :]

    loss = 0

    w = []
    sum_w = 0
    for i in range (n):
        tmp_w = 1 / (1 + np.exp(2 - (i + 1)))   #1/(1+e^(2-x)) np.exp(1)
        w.append(tmp_w)
        sum_w  += tmp_w
    
    for i in range (n):
        w_i = round(w[i] / sum_w, 2) 

        s = i * 13
        e = (i + 1) * 13
        predicted_tmp = y_pred[:, s:e, :, :]

        s1 = (i + n) * 13
        e1 = (i + n + 1) * 13
        predicted_grad_tmp = y_pred[:, s1:e1, :, :]

        loss += w_i * (K.mean(K.abs(predicted_tmp - target)) + b * K.mean(K.abs(predicted_grad_tmp - tar_grad)))

    return loss

