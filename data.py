# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:55:37 2023

@author: anadjj
"""

import cv2
import numpy as np
from imshowtools import imshow
import os
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

from imshowtools import imshow

def visualize(images, masks):
    
    counter = 0
    for x, y in zip(images, masks):
        print(x, y)
        counter += 1
        if counter == 10:
            break
    
    cat = []
    for x, y in zip(images[0:6], masks[:6]):
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        
        z = np.concatenate([x, y], axis=1)
        cat.append(z)
    
    imshow(*cat, size=(20, 10), columns=3)
    
    

def load_data(images, masks, split=0.1):
    ## 80 - 10 - 10
    
    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    #Dataset split
    
    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
    

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x/255.0 #normalization, normalization vs standardization
    ## (256, 256, 3)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x/255.0
    ## (256, 256)
    x = np.expand_dims(x, axis=-1)
    ## (256, 256, 1)
    return x

#building dataset pipeline
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    #print(tf_parse(x, y))
    #z, y = tf_parse
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset