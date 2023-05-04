# -*- coding: utf-8 -*-
"""
Created on Thu May  4 20:39:28 2023

@author: anadjj
"""

from glob import glob
import os

def combine(path1, path2):
    
    images1 = sorted(glob(os.path.join(path1, "images/*")))
    masks1 = sorted(glob(os.path.join(path1, "masks/*")))
    
    images2 = sorted(glob(os.path.join(path2, "images/*")))
    masks2 = sorted(glob(os.path.join(path2, "masks/*")))
    
    
    all_images = images1 + images2
    all_masks = masks1 + masks2
    
    return (all_images, all_masks)
    
    
    
    
