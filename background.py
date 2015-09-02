from itertools import repeat
import logging
import os

import matplotlib.pyplot as plt

from skimage import io, img_as_float, img_as_ubyte
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.color import rgb2gray, gray2rgb
from skimage.exposure import rescale_intensity
import numpy as np

__author__ = 'vvvkamper'

EPS = 0.03

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def erase_background(img):
    gray_img = rgb2gray(img)
    gray_img = rescale_intensity(gray_img)

    markers = np.zeros_like(gray_img)
    markers[gray_img < 0.7] = 1
    markers[:30, :30] = 2  # small top left corner
    markers[-30:, :30] = 2  # small top right corner
    markers[-30:, -30:] = 2  # small bottom right corner
    markers[:30, -30:] = 2  # small bottom left corner
    markers[gray_img == 1] = 2

    elevation_map = sobel(gray_img)
    segmentation = watershed(elevation_map, markers)

    alpha = (segmentation == 1)
    alpha = img_as_ubyte(alpha)
    alpha_img = np.dstack((img, img_as_ubyte(alpha)))
    return alpha_img, markers


def clean_background(img):
    img = img_as_float(img)
    if len(img.shape) == 3:
        pass
    elif len(img.shape) == 2:
        img = gray2rgb(img)
    else:
        raise ValueError('img shape must be (w, h, d) or (w, h)')
    bg = repeat(1, img.shape[2])
    alpha = np.linalg.norm(img - tuple(bg), axis=2)
    alpha[alpha > EPS] = 1
    alpha_img = np.dstack((img, alpha))
    return alpha_img, alpha


if __name__ == '__main__':

    filename = os.path.realpath("data/597832.jpg")
    image = io.imread(filename)

    img_with_alpha, img_markers = clean_background(image)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(img_markers, cmap=plt.cm.spectral, interpolation='nearest')
    ax.axis('off')
    ax.set_title('markers')

    io.show()

    io.imsave('out/589768_in_xl_clean.png', img_with_alpha)
