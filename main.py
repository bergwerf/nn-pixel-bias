#!/usr/bin/env python3

'''
This scripts implement a network that squeezes 100x100 into 18x18, then scales
up to 60x60. To analyze the influence of each pixel at a given output pixel
(x, y), a 1 channel image where one pixel is set to 1 is feeded into the network
where all parameters are initialized to 1.
'''


import sys
import argparse
import numpy as np

from keras.models import Sequential
from keras.initializers import Ones
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from skimage.io import imsave


def conv3x3():
    return Conv2D(1, (3, 3), use_bias=False, kernel_initializer='ones')


def conv2transpose():
    return Conv2DTranspose(1, (2, 2), use_bias=False, strides=(2, 2),
                           kernel_initializer='ones')


def generate_map(ox, oy):
    # Build model.
    model = Sequential()  # 100
    model.add(Conv2D(1, (3, 3), use_bias=False, kernel_initializer='ones',
                     input_shape=(100, 100, 1)))
    model.add(conv3x3())
    model.add(MaxPooling2D((2, 2)))  # 96 => 48
    model.add(conv3x3())
    model.add(conv3x3())
    model.add(MaxPooling2D((2, 2)))  # 44 => 22
    model.add(conv3x3())
    model.add(conv3x3())  # 18x18x1
    model.add(conv2transpose())  # 18 => 36
    model.add(conv3x3())
    model.add(conv3x3())
    model.add(conv2transpose())  # 32 => 64
    model.add(conv3x3())
    model.add(conv3x3())  # 60x60x1

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    # Output heatmap.
    map = np.zeros((100, 100))

    # Loop through all possible positions.
    for x in range(100):
        print('Processing x={}'.format(x))
        # Create batch to speed up processing.
        input = np.zeros((100, 100, 100, 1))
        for y in range(100):
            input[y][x][y] = 1

        spread = model.predict(input, batch_size=100)

        # Take (ox, oy) from spread and add it to map.
        for y in range(100):
            map[x][y] = spread[y][ox][oy]

    return map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=int, dest='x', default=0)
    parser.add_argument('-y', type=int, dest='y', default=0)
    parser.add_argument('-o', type=str, dest='out', default='out.png')
    args = parser.parse_args()

    # Should generate NumPy array with floats (float is standard in Keras).
    pmap = generate_map(args.x, args.y)

    # Normalize and save as image file.
    pmap = pmap / np.max(pmap)
    imsave(args.out, pmap)


if __name__ == '__main__':
    main()
