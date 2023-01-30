import csv
import os
from random import shuffle
import numpy as np

import image_metrics as img_met
from utils import make_dir, DataGenerator, process_predicted
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.utils import plot_model
from keras.models import load_model


def train(model, model_name, base_out_path, resume_file, train_filelist, val_filelist,
                  shuffle_train, data_augmentation, random_crop, batch_size, scale, clip_max, clip_min, max_val_sar,
                  crop_size, full_size, total_epochs, start_epoch, input_data_folder, input_shape):

    print('Training model name: {}'.format(model_name))

    out_path_train = make_dir(os.path.join(base_out_path, model_name))

    plot_model(model, to_file=os.path.join(out_path_train, 'HS2P_architecture.png'), show_shapes=True, show_layer_names=True)

    model_path = make_dir(os.path.join(out_path_train, 'Checkpoint'))
    model_filepath = os.path.join(model_path, '{epoch:02d}-{val_validation_metric:.4f}.hdf5')
    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor='val_validation_metric',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='max')

    # instantiate csv logging callback
    csv_name = 'csvlog.csv'
    csv_path = make_dir(os.path.join(out_path_train, 'Logs'))
    csv_filepath = os.path.join(csv_path, csv_name)
    csv_logger = CSVLogger(csv_filepath, append=True, separator=",")
    callbacks_list = [checkpoint, csv_logger]


    params = {'input_dim': input_shape,
              'batch_size': batch_size,
              'shuffle': shuffle_train,
              'scale': scale,
              'include_target': True,  # keep true
              'data_augmentation': data_augmentation,
              'random_crop': random_crop,
              'crop_size': crop_size,
              'full_size': full_size,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'input_data_folder': input_data_folder,
              'max_val_sar': max_val_sar}
    training_generator = DataGenerator(train_filelist, **params)

    params = {'input_dim': input_shape,
              'batch_size': batch_size,
              'shuffle': shuffle_train,
              'scale': scale,                                                                                                                
              'include_target': True,  # keep true
              'data_augmentation': False,  # keep false
              'random_crop': False,  # keep false
              'crop_size': crop_size,
              'full_size': full_size,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'input_data_folder': input_data_folder,
              'max_val_sar': max_val_sar
              }

    validation_generator = DataGenerator(val_filelist, **params)


    print('Training starts...')

    if resume_file != 'None':
        print("Will resume from the weights in file {}".format(resume_file))
        model.load_weights(resume_file)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=total_epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        shuffle=False,
                        initial_epoch=start_epoch)



def predict(predict_file, model, base_out_path, input_data_folder, predict_filelist, batch_size,
                    clip_min, clip_max, crop_size, full_size, include_target, input_shape, max_val_sar, scale):
    print("Predicting using file: {}".format(predict_file))

    # load the model weights at checkpoint
    model.load_weights(predict_file)

    out_path_predict = base_out_path
    predicted_images_path = make_dir(os.path.join(out_path_predict, 'Predicted'))

    print("Initializing generator for prediction and evaluation")
    params = {'input_dim': input_shape,
              'batch_size': batch_size,
              'shuffle': False,
              'scale': scale,
              'include_target': include_target,
              'data_augmentation': False,
              'random_crop': False,
              'crop_size': crop_size,
              'full_size': full_size,
              'clip_min': clip_min,
              'clip_max': clip_max,
              'input_data_folder': input_data_folder,
              'max_val_sar': max_val_sar}
    predict_generator = DataGenerator(predict_filelist, **params)


    if include_target:
        eval_csv_name = out_path_predict + '/evaluation.csv'
        print("Quantative Evaluation Results at ", eval_csv_name)

        for i, (data, y) in enumerate(predict_generator):
            print("Processing file number ", i+1)        
            # get evaluation metrics
            eval_results = model.test_on_batch(data, y)
            # write evaluation metrics
            with open(eval_csv_name, 'a') as eval_csv_fh:
                eval_writer = csv.writer(eval_csv_fh, dialect='excel')
                if(i == 0):
                    eval_writer.writerow(model.metrics_names)
                eval_writer.writerow(eval_results)

            #predict output image
            predicted = model.predict_on_batch(data)

            # # process predicted image
            process_predicted(predicted, predict_filelist[i * batch_size:i * batch_size + batch_size],
                                predicted_images_path, scale, input_data_folder)
    
    else:
        for i, data in enumerate(predict_generator):
            print("Processing file number ", i+1)        
            #predict output image
            predicted = model.predict_on_batch(data)

            # # process predicted image
            process_predicted(predicted, predict_filelist[i * batch_size:i * batch_size + batch_size],
                                predicted_images_path, scale, input_data_folder)

    print("Prediction finished with success!")
