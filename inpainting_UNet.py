
# coding: utf-8

# # Implementing the U-Net Architecture
# The paper uses a U-Net architecture for doing the image inpainting.
# We've implemented this in `libs/unet.py` module, and we will review it here.


import gc
import os
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imsave

from libs.pconv_model import PConvUnet
from libs.util import random_mask

# Settings
start_time = time.time()
MAX_BATCH_SIZE = 128
PConvUnet().summary()


# # Testing out on single image
# Load image
img = cv2.imread('./data/building.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512)) / 255
shape = img.shape
print(f"Shape of image is: {shape}")

# Load mask
mask = random_mask(shape[0], shape[1])

# Image + mask
masked_img = deepcopy(img)
masked_img[mask==0] = 1

# Show side by side
# _, axes = plt.subplots(1, 3, figsize=(20, 5))
# axes[0].imshow(img)
# axes[1].imshow(mask*255)
# axes[2].imshow(masked_img)
# plt.show()


# ## Creating data generator
# In this simple testing case we'll only be testing the architecture on a single image to see how it performs.
# We create a generator that will infinitely yield the same image and masked_image for us.
# The generator is based off the ImageDataGenerator from keras.processing, which allows us to do all kinds of autmentation more easily.


class DataGenerator(ImageDataGenerator):
    def flow(self, x, *args, **kwargs):
        while True:

            # Get augmentend image samples
            ori = next(super().flow(x, *args, **kwargs))

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask==0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori


# Create datagen
datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Create generator from numpy arrays
batch = np.stack([img for _ in range(MAX_BATCH_SIZE)], axis=0)
generator = datagen.flow(x=batch, batch_size=4)

# Get samples & Display them
(masked, mask), ori = next(generator)

# Show side by side
_, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].imshow(masked[0,:,:,:])
axes[1].imshow(mask[0,:,:,:]*255)
axes[2].imshow(ori[0,:,:,:])


# ## Training classifier on single image
def plot_callback(model):
    """Called at the end of each epoch, displaying our previous test images,
    as well as their masked predictions and saving them to disk"""

    # Get samples & Display them
    pred_img = model.predict([masked, mask])

    # Clear current output and display test images
    for i in range(len(ori)):
        # _, axes = plt.subplots(1, 3, figsize=(20, 5))
        # axes[0].imshow(masked[i,:,:,:])
        # axes[1].imshow(pred_img[i,:,:,:] * 1.)
        # axes[2].imshow(ori[i,:,:,:])
        # axes[0].set_title('Masked Image')
        # axes[1].set_title('Predicted Image')
        # axes[2].set_title('Original Image')
        # plt.show()
        imsave('result/{}_orginal.png'.format(i), ori[i,:,:,:])
        imsave('result/{}_masked.png'.format(i), masked[i,:,:,:])
        imsave('result/{}_pred.png'.format(i), pred_img[i,:,:,:])


if not os.path.exists('result/logs'):
    os.mkdir('result/logs')

model = PConvUnet(weight_filepath='result/logs/')
model.fit(
    generator,
    steps_per_epoch=1000,
    epochs=1,
    plot_callback=plot_callback,
)

e = int(time.time() - start_time)
print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
print('finish')
