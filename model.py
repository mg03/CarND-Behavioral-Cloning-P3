import pandas as pd
import numpy as np
import time
import shutil
import os
import random
import cv2
import math
import json
import sklearn
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, MaxPooling2D, Lambda, ELU
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def change_brightness(image):
    # Randomly select a percent change
    change_pct = random.uniform(0.4, 1.2)
    
    # Change to HSV to change the brightness V
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * change_pct
    
    #Convert back to RGB 
    img_brightness = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img_brightness

def flip_image(image, angle):
    img_flip = cv2.flip(image,1)
    angle = -angle
        
    return img_flip, angle

def resizeImage(image):
    # Proportionally get lower half portion of the image
    nrow, ncol, nchannel = image.shape
    
    start_row = int(nrow * 0.35)
    end_row = int(nrow * 0.875)   
    
    # This removes most of the sky and small amount below including the hood
    new_image = image[start_row:end_row, :]
    
    # This resizes to 66 x 220 for NVIDIA's model
    new_image = cv2.resize(new_image, (220,66), interpolation=cv2.INTER_AREA)
    
    return new_image


def augmented_row(data_row_df):
    angle = data_row_df['steering_angle']
    
    # random camera choice
    
    camera = np.random.choice(['center_image', 'left_image', 'right_image'])
    
    # adjust angle for left and right camera
    if camera == 'left_image':
        angle += 0.25
    elif camera == 'right_image':
        angle -+ 0.25
    else:
        angle = angle
        
    image = load_img(data_row_df[camera].strip())
    image = img_to_array(image).astype(np.uint8)
    
    # decide by coin flip whether to flip image or not
    
    img_flip = np.random.random()
    
    if img_flip > 0.5:
        # flip the image and reverse the steering angle
        image, angle = flip_image(image, angle)
        
    # change brightness
    image = change_brightness(image)
    
    # crop, resize and normalize image
    
    image = resizeImage(image)
    
    return image, angle

def data_generator(data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 66, 220, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch

def nvidia_model():
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape = (66, 220, 3)))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv1'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv2'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal', name='conv3'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal', name='conv4'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal', name='conv5'))
    model.add(Flatten(name='flatten1'))
    model.add(ELU())
    model.add(Dense(1164, init='he_normal', name='dense1'))
    model.add(ELU())
    model.add(Dense(100, init='he_normal', name='dense2'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal', name='dense3'))
    model.add(ELU())
    model.add(Dense(10, init='he_normal', name='dense4'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal', name='dense5'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss='mse')
    
    return model

if __name__ == "__main__":

    df1 = pd.read_csv('./data/driving_log.csv', header=0)
    df.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]

    df2 = pd.read_csv('./recovery_data/driving_log.csv', header=0)
    df2.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]
    df = pd.concat([df1, df2], ignore_index=True)
    df.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]

    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    training_split = 0.8

    num_rows_training = int(df.shape[0]*training_split)

    training_data = df.loc[0:num_rows_training-1]
    validation_data = df.loc[num_rows_training:]

    # release the main data_frame from memory
    df = None

    BATCH_SIZE = 32

    training_generator = data_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = data_generator(validation_data, batch_size=BATCH_SIZE)
    model = nvidia_model()
    samples_per_epoch = (20000//BATCH_SIZE)*BATCH_SIZE
    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=3000)
    
    print("Saving model weights and configuration file.")

    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())
