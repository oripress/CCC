# original code copied from https://github.com/hendrycks/robustness
# -*- coding: utf-8 -*-

import os
import os.path

import numba
import numpy as np
from scipy import interpolate

interpolation_function_dict = dict()

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"]


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


# /////////////// Distortion Helpers ///////////////

import ctypes
import warnings
from io import BytesIO

import cv2
import skimage as sk
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

warnings.simplefilter("ignore", UserWarning)


def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,  # wand
    ctypes.c_double,  # radius
    ctypes.c_double,  # sigma
    ctypes.c_double,
)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(
        img[top : top + ch, top : top + ch], (zoom_factor, zoom_factor, 1), order=1
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top : trim_top + h, trim_top : trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////
def gaussian_noise(x, severity=1):
    if "gaussian noise" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [0.0, 0.08, 0.12, 0.18, 0.26, 0.38],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["gaussian noise"] = f

    f = interpolation_function_dict["gaussian noise"]

    c = f(severity)

    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    if "shot noise" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                0,
                float(1) / 60,
                float(1) / 25,
                float(1) / 12,
                float(1) / 5,
                float(1) / 3,
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["shot noise"] = f

    f = interpolation_function_dict["shot noise"]

    c = f(severity)
    if c != 0:
        c = float(1) / c
    else:
        c = 9999

    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    if "impulse noise" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5], [0, 0.03, 0.06, 0.09, 0.17, 0.27], axis=0, kind="linear"
        )
        interpolation_function_dict["impulse noise"] = f

    f = interpolation_function_dict["impulse noise"]

    c = f(severity)

    x = sk.util.random_noise(np.array(x) / 255.0, mode="s&p", amount=c)
    return np.clip(x, 0, 1) * 255


def glass_blur_slow(x, severity=1):
    if "glass blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (0.0, 0.0, 0.0),
                (0.7, 1, 2),
                (0.9, 2, 1),
                (1, 2, 3),
                (1.1, 3, 2),
                (1.5, 4, 2),
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["glass blur"] = f

    f = interpolation_function_dict["glass blur"]

    c = f(severity)

    if c[1] < 1:
        c[1] = 1

    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(round(c[2])):
        for h in range(224 - round(c[1]), round(c[1]), -1):
            for w in range(224 - round(c[1]), round(c[1]), -1):
                dx, dy = np.random.randint(-round(c[1]), round(c[1]), size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255.0, sigma=c[0], multichannel=True), 0, 1) * 255


@numba.jit()
def shuffle_pixels(x, c):
    # locally shuffle pixels
    for i in range(round(c[2])):
        for h in range(224 - round(c[1]), round(c[1]), -1):
            for w in range(224 - round(c[1]), round(c[1]), -1):
                dx, dy = np.random.randint(-round(c[1]), round(c[1]), size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x


def glass_blur(x, severity=1):
    if "glass blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (0.0, 0.0, 0.0),
                (0.7, 1, 2),
                (0.9, 2, 1),
                (1, 2, 3),
                (1.1, 3, 2),
                (1.5, 4, 2),
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["glass blur"] = f

    f = interpolation_function_dict["glass blur"]

    c = f(severity)

    if c[1] < 1:
        c[1] = 1

    x = np.uint8(gaussian(np.array(x) / 255.0, sigma=c[0], multichannel=True) * 255)

    shuffle_pixels(x, c)

    return np.clip(gaussian(x / 255.0, sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1):
    if "defocus blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [(0.0, 0.0), (3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["defocus blur"] = f

    f = interpolation_function_dict["defocus blur"]

    c = f(severity)

    x = np.array(x) / 255.0
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1):
    if "motion blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [(0.0, 0.0), (10, 3), (15, 5), (15, 8), (15, 12), (20, 15)],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["motion blur"] = f

    f = interpolation_function_dict["motion blur"]

    c = f(severity)

    output = BytesIO()
    x.save(output, format="PNG")
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    if "zoom blur" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (1.0, 1.0, 0.01),
                (1, 1.11, 0.01),
                (1, 1.16, 0.01),
                (1, 1.21, 0.02),
                (1, 1.26, 0.02),
                (1, 1.31, 0.03),
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["zoom blur"] = f

    f = interpolation_function_dict["zoom blur"]

    c = f(severity)
    c = np.arange(c[0], c[1], c[2])

    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1):
    if "fog" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [(0.0, 2.0), (1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["fog"] = f

    f = interpolation_function_dict["fog"]

    c = f(severity)
    x = np.array(x) / 255.0
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    if "frost" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [(1.0, 0.0), (1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)],
            axis=0,
            kind="linear",
        )

        interpolation_function_dict["frost"] = f

    f = interpolation_function_dict["frost"]

    c = f(severity)

    idx = np.random.randint(5)
    filename = [
        "./frost/frost1.png",
        "./frost/frost2.png",
        "./frost/frost3.png",
        "./frost/frost4.jpg",
        "./frost/frost5.jpg",
        "./frost/frost6.jpg",
    ][idx]
    filename = os.path.abspath(filename)
    frost = cv2.imread(os.path.abspath(filename))
    # frost = cv2.cv.LoadImage(os.path.abspath(filename), CV_LOAD_IMAGE_COLOR)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(
        0, frost.shape[1] - 224
    )
    frost = frost[x_start : x_start + 224, y_start : y_start + 224][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1):
    if "snow" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (0.1, 0.3, 3, 1.0, 10, 4, 1.0),
                (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
                (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
                (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
                (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
                (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55),
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["snow"] = f

    f = interpolation_function_dict["snow"]

    c = f(severity)

    x = np.array(x, dtype=np.float32) / 255.0
    snow_layer = np.random.normal(
        size=x.shape[:2], loc=c[0], scale=c[1]
    )  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray(
        (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode="L"
    )
    output = BytesIO()
    snow_layer.save(output, format="PNG")
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = (
        cv2.imdecode(
            np.fromstring(snow_layer.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        / 255.0
    )
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(
        x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5
    )
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def contrast(x, severity=1):
    if "contrast" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5], [1.0, 0.4, 0.3, 0.2, 0.1, 0.05], axis=0, kind="linear"
        )
        interpolation_function_dict["contrast"] = f

    f = interpolation_function_dict["contrast"]
    c = f(severity)

    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, c):
    if "brightness" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], kind="linear"
        )
        interpolation_function_dict["brightness"] = f

    f = interpolation_function_dict["brightness"]
    c = f(c)

    x = np.array(x) / 255.0
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    if "jpeg" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5], [85, 25, 18, 15, 10, 7], axis=0, kind="linear"
        )
        interpolation_function_dict["jpeg"] = f

    f = interpolation_function_dict["jpeg"]
    c = f(severity)

    c = round(c.item())
    output = BytesIO()

    x.save(output, "JPEG", quality=c)
    x = PILImage.open(output)
    return x


def pixelate(x, severity=1):
    if "pixelate" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5], [1.0, 0.6, 0.5, 0.4, 0.3, 0.25], axis=0, kind="linear"
        )
        interpolation_function_dict["pixelate"] = f

    f = interpolation_function_dict["pixelate"]
    c = f(severity)

    x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x = x.resize((224, 224), PILImage.BOX)
    return x


def elastic_transform(image, severity=1):
    if "elastic_transform" not in interpolation_function_dict.keys():
        f = interpolate.interp1d(
            [0, 1, 2, 3, 4, 5],
            [
                (0, 999, 0),
                (244 * 2, 244 * 0.7, 244 * 0.1),
                (244 * 2, 244 * 0.08, 244 * 0.2),
                (244 * 0.05, 244 * 0.01, 244 * 0.02),
                (244 * 0.07, 244 * 0.01, 244 * 0.02),
                (244 * 0.12, 244 * 0.01, 244 * 0.02),
            ],
            axis=0,
            kind="linear",
        )
        interpolation_function_dict["elastic_transform"] = f

    f = interpolation_function_dict["elastic_transform"]
    c = f(severity)

    image = np.array(image, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
    )

    dx = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dy = (
        gaussian(
            np.random.uniform(-1, 1, size=shape[:2]), c[1], mode="reflect", truncate=3
        )
        * c[0]
    ).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = (
        np.reshape(y + dy, (-1, 1)),
        np.reshape(x + dx, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    return (
        np.clip(
            map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
            0,
            1,
        )
        * 255
    )


# /////////////// End Distortions ///////////////


def noise_dict():
    import collections

    d = collections.OrderedDict()
    d["gaussian_noise"] = gaussian_noise
    d["shot_noise"] = shot_noise
    d["impulse_noise"] = impulse_noise
    d["defocus_blur"] = defocus_blur
    d["glass_blur"] = glass_blur
    d["motion_blur"] = motion_blur
    d["zoom_blur"] = zoom_blur
    d["snow"] = snow
    d["frost"] = frost
    d["fog"] = fog
    d["brightness"] = brightness
    d["contrast"] = contrast
    d["elastic"] = elastic_transform
    d["pixelate"] = pixelate
    d["jpeg"] = jpeg_compression
    return d


if __name__ == "__main__":
    pass
