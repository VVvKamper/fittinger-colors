from colorsys import rgb_to_hsv
from skimage import io, img_as_float
from sklearn.cluster import KMeans
from color_wheel import find_color_code_by_color_distance
from color_wheel import find_color_code_naive

__author__ = 'vvvkamper'

SIZE = 32


def detect_image_main_color(img):
    img = img_as_float(img)
    r_img = img.reshape((-1, 3))
    est = KMeans(n_clusters=2)
    ans = est.fit(r_img)
    bg_index = ans.labels_[0]
    main_color = ans.cluster_centers_[(bg_index + 1) % 2]
    bg_color = ans.cluster_centers_[bg_index]
    return main_color, bg_color


def get_image_colors(img):
    img = img_as_float(img, True)  # TODO: deal with signed color type

    main_color, bg_color = detect_image_main_color(img)

    for i, color in enumerate([main_color, bg_color]):
        img[:SIZE, i * SIZE:(i + 1) * SIZE] = color

    img[SIZE:2*SIZE, :SIZE] = main_color
    img[2*SIZE:3*SIZE, :SIZE] = main_color

    color_by_distance, color_code_by_distance = find_color_code_by_color_distance(*main_color)

    color_hsv = rgb_to_hsv(*main_color)

    color, color_code_naive = find_color_code_naive(color_hsv[0] * 360.0, color_hsv[1] * 100.0, color_hsv[2] * 100.0)

    img[SIZE:2*SIZE, SIZE:2*SIZE] = color
    img[2*SIZE:3*SIZE, SIZE:2*SIZE] = color_by_distance

    return (color, color_code_naive), (color_by_distance, color_code_by_distance), img

if __name__ == '__main__':
    import os
    filename = os.path.realpath("data/4.6.jpg")
    image = io.imread(filename)
    get_image_colors(image)