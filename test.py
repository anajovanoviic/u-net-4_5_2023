# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:50:24 2023

@author: anadjj

"""

import os
from glob import glob

path = 'C:/Users/anadjj/OneDrive - Comtrade Group/Neural networks/u net - Final/u-net-4_5_2023/dataset1/images'

entries = os.scandir(path)

# Get a list of filenames in the same order as they appear in the directory
file_names = [entry.name for entry in sorted(entries, key=lambda x: x.stat().st_ctime)]

# Print the filenames in the same order as they appear in the directory
print(file_names)


file_list = os.listdir(path)

sorted_file_list = sorted(file_list, key=lambda x: os.path.join(path, x))

print(sorted_file_list)

