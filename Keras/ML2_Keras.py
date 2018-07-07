#!/usr/bin/env python
# coding: utf-8

# In[7]:

import matplotlib.pyplot as plt
import numpy as np
import datetime
import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


# fix random seed for reproducibility
np.random.seed(7)


# In[8]:

def pre_process_data(x_train, x_test, y_train, y_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    mean_train = np.mean(x_train,axis=0)
    mean_test = np.mean(x_test,axis=0)

    x_train -= mean_train
    x_test -= mean_test

    x_train  /= np.std(x_train,axis=0)
    x_test /= np.std(x_test,axis=0)
    
    return x_train, x_test, y_train, y_test


# In[9]:

def create_model(x_train):
    
    model = Sequential()
  
    model.add(Flatten(input_shape=x_train.shape[1:]))
    # 1st Layer
    
    model.add(Dense(1500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # 2nd Layer
    
    model.add(Dense(750))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    # 3rd Layer
    
    model.add(Dense(300))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.125))
    
    # Output Layer
    
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    adam_opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.000001)

    model.compile(loss='categorical_crossentropy', optimizer = adam_opt, metrics=['accuracy'])
    
    return model


# In[10]:

def fit_model(x_train, y_train, model, batch_size, epochs):
    datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,  
        zca_whitening=False, 
        rotation_range=5,  
        width_shift_range=0.16,  
        height_shift_range=0.16, 
        horizontal_flip=True, 
        vertical_flip=False) 
    
    datagen.fit(x_train[:40000])
    checkpointer = ModelCheckpoint(filepath="./Ass3.hdf5", verbose=1, save_best_only=True, monitor='val_acc')
    model.fit_generator(datagen.flow(x_train[:40000], y_train[:40000], batch_size=batch_size), epochs=epochs,
                        validation_data=(x_train[40000:], y_train[40000:]),workers=8, callbacks=[checkpointer])
    return model


# In[11]:

batch_size = 200
num_classes = 10
epochs = 3000


# In[12]:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[13]:

x_train, x_test, y_train, y_test = pre_process_data(x_train, x_test, y_train, y_test)


# In[14]:

model = create_model(x_train)


# In[15]:

model = fit_model(x_train, y_train, model,batch_size, epochs)

model.save('currentmodel.h5')