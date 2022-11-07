import pickle as pkl
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

MASK_DIR_PATH = 'Semantic_Segmentation_Dataset/train/labels/'

# image_paths = [os.path.join(IMAGE_DIR_PATH, x) for x in os.listdir(IMAGE_DIR_PATH) if x.endswith('.png')]
mask_paths = [os.path.join(MASK_DIR_PATH, x) for x in os.listdir(MASK_DIR_PATH) if x.endswith('.npy')]

mask_arrays = [np.load(i) for i in mask_paths]

mask_arrays_np = np.asarray(mask_arrays)

mask_ds = tf.data.Dataset.from_tensor_slices(mask_arrays_np)

print(mask_ds)