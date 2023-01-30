import csv
import os
import os.path

import rasterio
import keras
import matplotlib
import numpy as np
import scipy.signal as scisig
from matplotlib import pyplot as plt
import cv2


def make_dir(dir_path):
    if os.path.isdir(dir_path):
        print("WARNING: Folder {} exists and content may be overwritten!")
    else:
        os.makedirs(dir_path)

    return dir_path


def get_filelists(listpath):
    '''Get the number of samples'''
    
    with open(listpath) as f:
        reader = csv.reader(f, delimiter=',')
        filelist = list(reader)
    
    train_filelist = [f[:] for f in filelist if f[0] == '1']
    val_filelist = [f[:] for f in filelist if f[0] == '2']
    test_filelist = [f[:] for f in filelist if f[0] == '3']

    return train_filelist, val_filelist, test_filelist


# Process output data
def get_info_quartet(ID, input_data_folder):

    file_name = ['train', 'val', 'test']
    I = ID[4] + '.tif'

    if (I[9] == 'f'):
        scene_name = I[:14] + I[17:]
    else:
        scene_name = I[:16] + I[19:]
    

    if (I[9] == 'f'):
        filepath_sar = os.path.join(input_data_folder, ID[1], file_name[int(ID[0]) - 1], I[:14] + 's1_' + I[17:]).lstrip()
    else:
        filepath_sar = os.path.join(input_data_folder, ID[1], file_name[int(ID[0]) - 1], I[:16] + 's1_' + I[19:]).lstrip()
    

    filepath_cloudFree = os.path.join(input_data_folder, ID[2], file_name[int(ID[0]) - 1], I).lstrip()


    if (I[9] == 'f'):
        filepath_cloudy = os.path.join(input_data_folder, ID[3], file_name[int(ID[0]) - 1], I[:14] + 's2_cloudy_' + I[17:]).lstrip()
    else:
        filepath_cloudy = os.path.join(input_data_folder, ID[3], file_name[int(ID[0]) - 1], I[:16] + 's2_cloudy_' + I[19:]).lstrip()
    

    return scene_name[:-4], filepath_sar, filepath_cloudFree, filepath_cloudy


def get_rgb_preview(r, g, b, sar_composite=False):

    if not sar_composite:

        # stack and move to zero
        rgb = np.dstack((r, g, b))
        rgb = rgb - np.nanmin(rgb)

        # treat saturated images, scale values
        if np.nanmax(rgb) == 0:
            rgb = 255 * np.ones_like(rgb)
        else:
            rgb = 255 * (rgb / np.nanmax(rgb))

        # replace nan values before final conversion
        rgb[np.isnan(rgb)] = np.nanmean(rgb)

        return rgb.astype(np.uint8)

    else:
        # generate SAR composite
        HH = r
        HV = g

        HH = np.clip(HH, -25.0, 0)
        HH = (HH + 25.1) * 255 / 25.1
        HV = np.clip(HV, -32.5, 0)
        HV = (HV + 32.6) * 255 / 32.6

        rgb = np.dstack((np.zeros_like(HH), HH, HV))

        return rgb.astype(np.uint8)


def get_raw_data(path):
    with rasterio.open(path, driver='GTiff') as src:
        image = src.read()

    # checkimage for nans
    image[np.isnan(image)] = np.nanmean(image)

    return image.astype('float32')


def get_preview(file, predicted_file, bands, brighten_limit=None, sar_composite=False):
    if not predicted_file:
        with rasterio.open(file) as src:
            r, g, b = src.read(bands)
    else:
        # file is actually the predicted array
        r = file[bands[0] - 1]
        g = file[bands[1] - 1]
        b = file[bands[2] - 1]

    if brighten_limit is None:
        return get_rgb_preview(r, g, b, sar_composite)
    else:
        r = np.clip(r, 0, brighten_limit)
        g = np.clip(g, 0, brighten_limit)
        b = np.clip(b, 0, brighten_limit)
        return get_rgb_preview(r, g, b, sar_composite)


def generate_output_images(predicted, ID, predicted_images_path, input_data_folder):

    scene_name, filepath_sar, filepath_cloudFree, filepath_cloudy = get_info_quartet(ID, input_data_folder)

    print("Generating quartet for ", scene_name)

    predicted = predicted[39:52, :, :]

    sar_preview = get_preview(filepath_sar, False, [1, 2, 2], sar_composite=True)

    opt_bands = [4, 3, 2]  # R, G, B bands (S2 channel numbers)
    cloudFree_preview = get_preview(filepath_cloudFree, False, opt_bands, brighten_limit=2000)
    cloudy_preview = get_preview(filepath_cloudy, False, opt_bands, brighten_limit=2000)
    predicted_preview = get_preview(predicted, True, opt_bands, brighten_limit=2000)

    save_single_images(predicted_images_path, scene_name, sar_preview, cloudy_preview, cloudFree_preview, predicted_preview)

    return


def save_single_image(image, out_path, name):
    plt.figure(frameon=False)
    plt.imshow(image)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.axis('off')
    plt.savefig(os.path.join(out_path, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    return

def save_graymap(image, out_path, name):
    cmap="gray"
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.savefig(os.path.join(out_path, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    return


def save_single_images(predicted_images_path, scene_name, sar_preview, cloudy_preview, cloudFree_preview, predicted_preview):

    out_path = make_dir(os.path.join(predicted_images_path, scene_name))

    save_graymap(sar_preview[:, :, 1], out_path, "inputsar")
    save_single_image(cloudy_preview, out_path, "input")
    save_single_image(cloudFree_preview, out_path, "inputtarg")
    save_single_image(predicted_preview, out_path, "inputpred")

    return


def process_predicted(predicted, ID, predicted_images_path, scale, input_data_folder):
    for i, data_image in enumerate(predicted):
        data_image *= scale
        generate_output_images(data_image, ID[i], predicted_images_path, input_data_folder)

    return


# Process input data
class DataGenerator(keras.utils.Sequence):
    """DataGenerator for Keras routines."""

    def __init__(self,
                 list_IDs,
                 input_dim,
                 batch_size=1,
                 scale=2000,
                 shuffle=False,
                 include_target=True,
                 data_augmentation=False,
                 random_crop=False,
                 crop_size=256,
                 full_size=256,
                 clip_min=None,
                 clip_max=None,
                 input_data_folder='./',
                 max_val_sar=5
                 ):

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.nr_images = len(self.list_IDs)
        self.indexes = np.arange(self.nr_images)
        self.scale = scale
        self.shuffle = shuffle
        self.include_target = include_target
        self.data_augmentation = data_augmentation
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.full_size = full_size
        self.max_val = max_val_sar

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.input_data_folder = input_data_folder

        self.augment_rotation_param = np.repeat(0, self.nr_images)
        self.augment_flip_param = np.repeat(0, self.nr_images)
        self.random_crop_paramx = np.repeat(0, self.nr_images)
        self.random_crop_paramy = np.repeat(0, self.nr_images)

        self.on_epoch_end()

        print("Generator initialized")


    def __len__(self):
        """Gets the number of batches per epoch"""
        return int(np.floor(self.nr_images / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if self.include_target:
            # Generate data
            X, y = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                          self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                          self.random_crop_paramy[indexes])
            return X, y
        else:
            X = self.__data_generation(list_IDs_temp, self.augment_rotation_param[indexes],
                                       self.augment_flip_param[indexes], self.random_crop_paramx[indexes],
                                       self.random_crop_paramy[indexes])
            return X


    def on_epoch_end(self):
        """Update indexes after each epoch."""

        if self.shuffle:
            np.random.shuffle(self.indexes)

        if self.data_augmentation:
            self.augment_rotation_param = np.random.randint(0, 4, self.nr_images)
            self.augment_flip_param = np.random.randint(0, 3, self.nr_images)

        if self.random_crop:
            self.random_crop_paramx = np.random.randint(0, int(self.full_size-self.crop_size), self.nr_images)
            self.random_crop_paramy = np.random.randint(0, int(self.full_size-self.crop_size), self.nr_images)
        return


    def __data_generation(self, list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp, random_crop_paramx_temp, random_crop_paramy_temp):


        input_sar_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                         random_crop_paramx_temp, random_crop_paramy_temp, data_type=1)


        output_opt_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp,
                                          random_crop_paramx_temp, random_crop_paramy_temp, data_type=2)
        
 
        input_opt_batch = self.get_batch(list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp, 
                                         random_crop_paramx_temp, random_crop_paramy_temp, data_type=3)
        
        empty_opt_batch = np.empty((self.batch_size, *self.input_dim[0])).astype('float32')

        if self.include_target:
            return ([input_opt_batch, input_sar_batch, output_opt_batch], [output_opt_batch])
        elif not self.include_target:
            return ([input_opt_batch, input_sar_batch, empty_opt_batch])


    def get_image_data(self, paramx, paramy, path):
        # with block not working with window kw
        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read(window=((paramx, paramx + self.crop_size), (paramy, paramy + self.crop_size)))
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts
        return image


    def get_opt_image(self, path, paramx, paramy):

        image = self.get_image_data(paramx, paramy, path)

        return image.astype('float32')


    def get_sar_image(self, path, paramx, paramy):

        image = self.get_image_data(paramx, paramy, path)

        return image.astype('float32')


    def get_data_image(self, ID, data_type, paramx, paramy):

        file_name = ['train', 'val', 'test']

        if(data_type == 1):
            if (ID[4][9] == 'f'):
                pathname = ID[4][:14] + 's1_' + ID[4][17:] + '.tif'
            else:
                pathname = ID[4][:16] + 's1_' + ID[4][19:] + '.tif'
        elif (data_type == 3):
            if (ID[4][9] == 'f'):
                pathname = ID[4][:14] + 's2_cloudy_' + ID[4][17:] + '.tif'
            else:
                pathname = ID[4][:16] + 's2_cloudy_' + ID[4][19:] + '.tif'
        else:
            pathname = ID[4] + '.tif'
        
        data_path = os.path.join(self.input_data_folder, ID[data_type], file_name[int(ID[0])-1], pathname).lstrip()

        if data_type == 2 or data_type == 3:
            data_image = self.get_opt_image(data_path, paramx, paramy)
        elif data_type == 1:
            data_image = self.get_sar_image(data_path, paramx, paramy)
        else:
            print('Error! Data type invalid')

        return data_image


    def get_normalized_data(self, data_image, data_type):
        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (
                        self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))

        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel],
                                              self.clip_max[data_type - 1][channel])

            data_image /= self.scale

        return data_image

 
    def get_batch(self, list_IDs_temp, augment_rotation_param_temp, augment_flip_param_temp, random_crop_paramx_temp,
                  random_crop_paramy_temp, data_type):

        if data_type == 1:
            dim = self.input_dim[1]
        else:
            dim = self.input_dim[0]

        batch = np.empty((self.batch_size, *dim)).astype('float32')

        for i, ID in enumerate(list_IDs_temp):
            data_image = self.get_data_image(ID, data_type, random_crop_paramx_temp[i], random_crop_paramy_temp[i])
            if self.data_augmentation:
                if not augment_flip_param_temp[i] == 0:
                    data_image = np.flip(data_image, augment_flip_param_temp[i])
                if not augment_rotation_param_temp[i] == 0:
                    data_image = np.rot90(data_image, augment_rotation_param_temp[i], (1, 2))

            data_image = self.get_normalized_data(data_image, data_type)

            batch[i,] = data_image


        return batch
