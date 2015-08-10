import logging

logger = logging.getLogger()

from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import HSVColor, sRGBColor, LabColor
import numpy as np
from itertools import product, izip
from config import (BLACK_MAX_BRIGHTNESS, WHITE_MIN_BRIGHTNESS, GRAY_MAX_SATURATION, WHITE_MAX_SATURATION, HUE_TABLE,
                    BRIGHTNESS_IMPORTANCE_LEVEL, SATURATION_TABLE, BRIGHTNESS_TABLE)

from utils import find_nearest_idx

__author__ = 'vvvkamper'


color_wheel = product(enumerate(HUE_TABLE.values()), enumerate(izip(SATURATION_TABLE, BRIGHTNESS_TABLE)))
color_wheel = [(HSVColor(h, s / 100.0, v / 100.0), ih, iy) for (ih, h), (iy, (s, v)) in color_wheel]

algorithms = {
    'delta_e_cie2000': delta_e_cie2000
}


def find_color_code_by_color_distance(r, g, b, algorithm='delta_e_cie2000'):
    """

    :param r:
    :param g:
    :param b:
    :param algorithm:
    :return: tuple(sRGBColor, color_code)
    """
    color = sRGBColor(r, g, b)
    lab_color = convert_color(color, LabColor)
    distances = [algorithms[algorithm](lab_color, convert_color(x[0], LabColor)) for x in color_wheel]
    np_distances = np.array(distances)
    idx = np.argmin(np_distances)
    ans_color = color_wheel[idx][0]
    return convert_color(ans_color, sRGBColor).get_value_tuple(), (color_wheel[idx][1], color_wheel[idx][2])


def check_gray(s, b):
    if b < BLACK_MAX_BRIGHTNESS:
        return False
    elif b > WHITE_MIN_BRIGHTNESS:
        return False
    elif s > GRAY_MAX_SATURATION:
        return False
    elif (GRAY_MAX_SATURATION * s + (WHITE_MIN_BRIGHTNESS - BLACK_MAX_BRIGHTNESS) * b > (
        WHITE_MIN_BRIGHTNESS - BLACK_MAX_BRIGHTNESS) * GRAY_MAX_SATURATION):
        return False
    else:
        return True


def check_black(s, b):
    return b <= BLACK_MAX_BRIGHTNESS


def check_white(s, b):
    return b >= WHITE_MIN_BRIGHTNESS and s <= WHITE_MAX_SATURATION


def find_color_code_naive(h, s, b):
    if check_gray(s, b):
        return 'gray'
    if check_black(s, b):
        return 'black'
    if check_white(s, b):
        return 'white'
    first = find_nearest_idx(HUE_TABLE.values(), h)
    if b > BRIGHTNESS_IMPORTANCE_LEVEL:
        logger.info('Brightness is important')
        second = find_nearest_idx(SATURATION_TABLE, s)
    else:
        logger.info('Saturation is important')
        second = find_nearest_idx(BRIGHTNESS_TABLE, b)
    color = HSVColor(HUE_TABLE[first], SATURATION_TABLE[second]/100.0, BRIGHTNESS_TABLE[second]/100.0)
    color = convert_color(color, sRGBColor)
    return color.get_value_tuple(), (first, second)
