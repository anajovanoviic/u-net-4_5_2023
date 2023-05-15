# -*- coding: utf-8 -*-
"""
Created on Mon May 15 03:26:27 2023

@author: anadjj
"""
from two_dataset_comb import combine
from data import visualize
import numpy as np
from model3 import build_model


from sklearn.model_selection import GridSearchCV

from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasClassifier

import cv2
from PIL import Image #PIL library to resize images


from sklearn.model_selection import train_test_split

np.random.seed(42)



if __name__ == "__main__":
    
    path1 = "dataset1"
    path2 = "dataset2"
    
    all_images, all_masks = combine(path1, path2)
    
    print(f"Images: {len(all_images)} - Masks: {len(all_masks)}")
    
    visualize(all_images, all_masks)
    
    
    SIZE = 256
    image_dataset = []
    mask_dataset = []
    
    for i, image_name in enumerate(all_images):    #Remember enumerate method adds a counter and returns the enumerate object
        #print(image_directory+image_name)
        image = cv2.imread(all_images[i], cv2.IMREAD_COLOR) # 0-ucitavanje slike u crno belom modu
        #image = image/255.0
        image = Image.fromarray(image)
        #image.show()
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image).reshape(256, 256, 3))
        
    for i, image_name in enumerate(all_masks):
        image = cv2.imread(all_masks[i], cv2.IMREAD_GRAYSCALE)
        #image = image/255.0
        #image = np.expand_dims(image, axis=-1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image).reshape(256, 256, 1))
        
    
    image_dataset = np.array(image_dataset)
    mask_dataset = np.array(mask_dataset)
    

    
    x_grid, x_not_use, y_grid, y_not_use = train_test_split(image_dataset, mask_dataset, test_size=0.9, random_state=42)
    
    
    input_dim = x_grid.shape[1]
    
    learning_rate=0.01
    momentum=0.1
    
    def define_model(learning_rate, momentum):
        model = build_model(learning_rate, momentum)
        
        optimizer = SGD(lr=learning_rate, momentum=momentum)
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,      
                      metrics=['acc'])
        return model
    
    
    batch_size = 100
    epochs = 10
    
    model = KerasClassifier(build_fn=define_model, 
                            epochs=epochs, 
                            batch_size = batch_size, 
                            verbose=1)
    
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    momentum = [0.3, 0.5, 0.7, 0.9]
    

    param_grid = dict(learning_rate=learning_rate, momentum=momentum)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    
    grid_result = grid.fit(x_grid, y_grid)
    

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("Mean = %f (std=%f) with: %r" % (mean, stdev, param))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    