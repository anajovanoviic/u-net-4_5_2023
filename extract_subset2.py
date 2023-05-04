# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:54:27 2023

@author: anadjj
"""

import os
import random
import shutil
from glob import glob

src_dir = 'C:/Users/anadjj/OneDrive - Comtrade Group/Neural networks/projekat/kvasir-seg/Kvasir-SEG/images'
dest_dir = 'C:/Users/anadjj/OneDrive - Comtrade Group/Neural networks/u net - Final/u-net-4_5_2023/dataset2/images'

dest_dir2 = 'C:/Users/anadjj/OneDrive - Comtrade Group/Neural networks/u net - Final/u-net-4_5_2023/dataset2/masks'

# Set the number of images to extract
num_images = 50

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)



image_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.jpg')]
# Randomly shuffle the list of image files


#shuffle in below line you should run only once
#random.shuffle(image_files)
# Extract the first num_images from the shuffled list
selected_images = image_files[:num_images]


# Copy the selected images to the destination directory
for img_file in selected_images:
    shutil.copy(img_file, dest_dir)
    
selected_images.sort()
# path = 'C:/Users/anadjj/OneDrive - Comtrade Group/Neural networks/u net - Final/dataset1/'
# images_subset = sorted(glob(os.path.join(path, "images/*")))
# name = ''

for i, image_name in enumerate(selected_images):
    name = image_name.split("\\")[-1]
    src2 = 'C:/Users/anadjj/OneDrive - Comtrade Group/Neural networks/projekat/kvasir-seg/Kvasir-SEG/masks/' + name
    shutil.copy(src2, dest_dir2)
    
final_images = os.listdir('C:/Users/anadjj/OneDrive - Comtrade Group/Neural networks/u net - Final/u-net-4_5_2023/dataset2/images')
final_masks = os.listdir('C:/Users/anadjj/OneDrive - Comtrade Group/Neural networks/u net - Final/u-net-4_5_2023/dataset2/masks')

final_images.sort()
final_masks.sort()
    
