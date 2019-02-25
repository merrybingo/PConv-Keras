
# coding: utf-8

# Predictions for images of arbitrary size

from copy import deepcopy

import cv2
from skimage.io import imsave

# Import modules from libs/ directory
from libs.pconv_model import PConvUnet
from libs.util import random_mask, plot_images

# Load image
img = cv2.imread('./data/building.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255
shape = img.shape
print(f"Shape of image is: {shape}")

# Load mask
mask = random_mask(shape[0], shape[1])

# Image + mask
masked_img = deepcopy(img)
masked_img[mask==0] = 1

model = PConvUnet(weight_filepath='result/logs/')
model.load(r"result/logs/1_weights_2019-02-21-04-59-53.h5", train_bn=False)

# Run prediction quickly
pred = model.scan_predict((img, mask))

# Show result
plot_images([img, masked_img, pred])
imsave('result/test_orginal.png', img)
imsave('result/test_masked.png', masked_img)
imsave('result/test_pred.png', pred)

print("finish")
