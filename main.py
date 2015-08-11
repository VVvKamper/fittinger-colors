from background import erase_background, clean_background
from get_colors import get_image_colors
from skimage.color import gray2rgb

__author__ = 'vvvkamper'
import os
import json

from skimage import io

for root, subdirs, files in os.walk('data/out'):
    for f in files:
        filename = os.path.join(root, f)
        if 'jpg' in filename and '_' not in filename:
            image = io.imread(filename)
            if len(image.shape) == 2:  # 1 channel greyscale image
                image = gray2rgb(image)

            alpha_image, markers = clean_background(image)
            io.imsave(os.path.join(root, f[:-4] + '_clean.png'), alpha_image)
            # io.imsave(os.path.join(root, f[:-4] + '_markers.jpg'), markers)
            # (color, color_code_naive), (color_by_distance, color_code_by_distance), debug_img = get_image_colors(image)
            # io.imsave(os.path.join(root, f[:-4] + '_color_debug.jpg'), debug_img)
            # with open(os.path.join(root, 'colors.json'), 'w') as outfile:
            #     json.dump({
            #         'naive': '.'.join(map(str, color_code_naive)),
            #         'distance': '.'.join(map(str, color_code_by_distance))
            #     }, outfile)
