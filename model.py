import numpy as np
import cv2
import random
from random import randrange

import keras
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Cropping2D

import sklearn
from sklearn.model_selection import train_test_split
import csv

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os
import ntpath
import random

lines = []
training_folder = '../data/'
training_csv = training_folder + 'driving_log.csv'
training_img_folder = training_folder + '/IMG/'

with open(training_csv) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
 


#random.shuffle(lines)

def generator(linesgiven, batchsize):
    while True:
        count = 0
        images = []
        measurements = []
        is_reverse = False
        random.shuffle(linesgiven)
        for line in linesgiven:
            if count == 0:
                images = []
                measurements = []
                is_reverse = False
            img_source = line[0]
            file_name = img_source.split('/')[-1]
            current_path = training_img_folder + file_name
            image = cv2.imread(current_path)
            measurement = float(line[3])
            if is_reverse:
                images.append(image)
                measurements.append(measurement)
                is_reverse = False
            else:
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1)
                is_reverse = True
            if count + 1 >= batchsize:
                count = 0
                yield np.array(images), np.array(measurements)
            else:
                count = count + 1


model = Sequential()


model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((60,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))
model.add(Convolution2D(64, 3, 3, activation='elu'))

model.add(Flatten())
model.add(Dense(100, activation = 'elu'))
model.add(Dense(50, activation = 'elu'))
model.add(Dense(10, activation = 'elu'))
model.add(Dense(1))
optimizer = Adam(lr=1e-3)
model.compile(loss='mse', optimizer=optimizer)

random.shuffle(lines)
train_lines, validation_lines = sklearn.model_selection.train_test_split(lines, test_size=0.2)
train_generator = generator(train_lines, 32)
validation_generator = generator(validation_lines, 32)

history_object = model.fit_generator(train_generator, samples_per_epoch = 600, validation_data = validation_generator, nb_val_samples = 120, nb_epoch=10, verbose=1)

model.save('model.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


exit()
