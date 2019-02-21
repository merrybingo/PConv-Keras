
# coding: utf-8

# # Mask Generation with OpenCV
# In the paper they generate irregular masks by using occlusion/dis-occlusion between two consecutive frames of videos, as described in [this paper](https://lmb.informatik.uni-freiburg.de/Publications/2010/Bro10e/sundaram_eccv10.pdf).
#
# Instead we'll simply be using OpenCV to generate some irregular masks, which will hopefully perform just as well.
# We've implemented this in the function `random_mask`, which is located in the `util.py` file int he libs directory


import itertools

import matplotlib.pyplot as plt

from libs.util import random_mask

# Plot the results
_, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = list(itertools.chain.from_iterable(axes))

for i in range(len(axes)):
    
    # Generate image
    img = random_mask(500, 500)
    
    # Plot image on axis
    axes[i].imshow(img*255)

print("finish")