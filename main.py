from __future__ import division

import random

import keras.backend as K
import numpy as np
import tensorflow as tf
from configparser import ConfigParser
from distutils.util import strtobool
import image_metrics as img_met
from network import HS2P_model
from tools import train, predict
from keras.optimizers import Nadam
from keras.utils import multi_gpu_model
from utils import get_filelists
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
K.set_image_data_format('channels_first')


def run_hs2p(cfg):

    # set model parameters
    model_name = str(cfg.get('Base', 'model_name'))  
    feature_size = int(cfg.get('Base', 'feature_size'))
    start_epoch = int(cfg.get('Base', 'start_epoch'))
    total_epoch = int(cfg.get('Base', 'total_epoch'))
    batch_size = int(cfg.get('Base', 'batch_size'))
    N = int(cfg.get('Base', 'ResGroupNum'))
    lr = float(cfg.get('Base', 'lr'))
    optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, schedule_decay=0.004)

    include_target = strtobool(cfg.get('Data', 'include_target'))  #set False only in the testing phase
    shuffle_train = strtobool(cfg.get('Data', 'shuffle_train'))
    data_augmentation = strtobool(cfg.get('Data', 'data_augmentation'))
    random_crop = strtobool(cfg.get('Data', 'random_crop'))
    crop_size = int(cfg.get('Data', 'crop_size'))
    full_size = int(cfg.get('Data', 'full_size')) #keep 256 if you use the SEN12MS-CR dataset (256*256 px)
    scale = int(cfg.get('Data', 'scale'))
    max_val_sar = int(cfg.get('Data', 'max_val_sar'))
    sar_channels = int(cfg.get('Data', 'sar_channels'))  #keep 2
    opt_channels = int(cfg.get('Data', 'opt_channels'))   #keep 13
    input_shape = ((opt_channels, crop_size, crop_size), (sar_channels, crop_size, crop_size), (opt_channels, crop_size, crop_size))
    clip_min = [[-25.0, -32.5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    clip_max = [[0, 0], [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
                [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]]


    dataset_list_filepath = str(cfg.get('Path', 'dataset_list_filepath'))  #set your csv file path
    input_data_folder = str(cfg.get('Path', 'input_data_folder'))  #set your data root path
    base_out_path = str(cfg.get('Path', 'base_out_path'))  #set your root path of the output
    

    loss = img_met.loss
    metrics = [img_met.MAE, img_met.RMSE, 
               img_met.PSNR, img_met.SAM, 
               img_met.SSIM, img_met.validation_metric]


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))
    random_seed_general = 42
    random.seed(random_seed_general)
    np.random.seed(random_seed_general)
    tf.set_random_seed(random_seed_general)


    model = HS2P_model(input_shape, N, feature_size=feature_size)
    for layer in model.layers:
        if('get_gradv' in layer.name or 'get_gradh' in layer.name):
            layer.trainable = False
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print('Model compiled successfully!')


    print("Getting file lists")
    train_filelist, val_filelist, test_filelist = get_filelists(dataset_list_filepath)
    print("Number of train files found: ", len(train_filelist))
    print("Number of validation files found: ", len(val_filelist))
    print("Number of test files found: ", len(test_filelist))


    predict_file = str(cfg.get('Path', 'predict_file'))
    resume_file = str(cfg.get('Path', 'resume_file'))

    if predict_file != 'None':
        predict_filelist = test_filelist
        predict(predict_file, model, base_out_path, input_data_folder, predict_filelist,
                    batch_size, clip_min, clip_max, crop_size, full_size, include_target, input_shape, max_val_sar, scale)

    else:
        train(model, model_name, base_out_path, resume_file, train_filelist, val_filelist,
                shuffle_train, data_augmentation, random_crop, batch_size, scale, clip_max, clip_min, max_val_sar,
                crop_size, full_size, total_epoch, start_epoch, input_data_folder, input_shape)




if __name__ == '__main__':

    cfg_path = "./config.ini"  #set your config.ini file path first!
    cfg = ConfigParser()
    cfg.read(cfg_path, encoding='gbk')

    run_hs2p(cfg)
