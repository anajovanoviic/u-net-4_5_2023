# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 18:44:07 2023

@author: anadjj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:47:47 2023

@author: anadjj
"""
from two_dataset_comb import combine
from data import visualize
from data import load_data
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, tf_dataset
from model import build_model

from matplotlib import pyplot as plt

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


if __name__ == "__main__":
    
    path1 = "dataset1"
    path2 = "dataset2"
    
    all_images, all_masks = combine(path1, path2)
    
    print(f"Images: {len(all_images)} - Masks: {len(all_masks)}")
    
    visualize(all_images, all_masks)
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(all_images, all_masks)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")
    
    print(test_x)
    
    '''
    ## Hyperparameters
    batch = 8
    lr = 1e-4
    #epochs = 20
    epochs = 5
    '''
    
    batches = [8, 32, 256]
    learning_rates = [1e-4, 1e-3, 1e-2]
    epochs = [10, 50, 200]
    
    best_iou = 0.0
    best_params = {}
    
    for batch in batches:
        for lr in learning_rates:
            for num_epochs in epochs:
                # Train your model with current hyperparameters
                # Evaluate the model and calculate the IOU score
                
                train_dataset = tf_dataset(train_x, train_y, batch=batch)
                valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

                model = build_model()

                opt = tf.keras.optimizers.Adam(lr)
                metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
                model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

                callbacks = [
                    ModelCheckpoint("files/model.h5"),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
                    CSVLogger("files/data.csv"),
                    TensorBoard(),
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
                ]

                train_steps = len(train_x)//batch
                valid_steps = len(valid_x)//batch

                if len(train_x) % batch != 0:
                    train_steps += 1
                if len(valid_x) % batch != 0:
                    valid_steps += 1

                    

                history = model.fit(train_dataset,
                    validation_data=valid_dataset,
                    epochs=num_epochs,
                    steps_per_epoch=train_steps,
                    validation_steps=valid_steps,
                    callbacks=callbacks)
                
                val_iou_values = history.history['val_iou']
                
                iou_score = max(val_iou_values)  #this param is actually highest value of iou recorded in one training
                
                print(iou_score)
                
                # Save the results if it's the best IOU score so far
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_params = {'batch_size': batch, 'lr': lr, 'epochs': num_epochs}
    
    print("Best IOU score:", best_iou)
    print("Best parameters:", best_params)
        
    
    
    
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    #acc = history.history['acc']
    acc = history.history['acc']
    #val_acc = history.history['val_acc']
    val_acc = history.history['val_acc']
    
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    '''
