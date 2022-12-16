import numpy as np
import cv2
import math
import random


class Coloring:

    def intensity_slicing(self, image, n_slices):
        # Convert greyscale image to color image using color slicing technique.
        # takes as input:
        # image: the grayscale input image
        # n_slices: number of slices

        # Steps:

        # 1. Split the exising dynamic range (0, k-1) using n slices (creates n+1 intervals)
        # 2. Randomly assign a color to each interval
        # 3. Create and output color image
        # 4. Iterate through the image and assign colors to the color image based on which interval the intensity belongs to

        # returns color image

        rows, cols = np.shape(image)
        color = np.zeros((rows, cols, 3), dtype=np.uint8)

        length = n_slices + 1
        interval = 256 / length

        rgb = np.random.randint(255, size=(length, 3), dtype=np.uint8)

        rgb[0] = [0] * 3

        for r in range(0, rows):
            for c in range(0, cols):
                intensity = image[r][c]
                idx = int(intensity / interval)
                color[r][c] = rgb[idx]

        return color

    def color_transformation(self, image, n_slices, theta):
        # Convert greyscale image to color image using color transformation technique.
        # takes as input:
        # image:  grayscale input image
        # colors: color array containing RGB values

        # Steps:
        # 1. Split the exising dynamic range (0, k-1) using n slices (creates n+1 intervals)
        # 2. create red values for each slice using 255*sin(slice + theta[0])
        #    similarly create green and blue using 255*sin(slice + theta[1]), 255*sin(slice + theta[2])
        # 3. Create and output color image
        # 4. Iterate through the image and assign colors to the color image based on which interval the intensity belongs to

        # returns color image
        rows, cols = np.shape(image)
        color = np.zeros((rows, cols, 3), dtype=np.uint8)

        length = n_slices + 1
        interval = 256 / length

        rgb = np.zeros((length, 3), dtype=np.uint8)

        for r in range(1, length):
            rgb[r][0] = 255 * np.sin(r + (theta[0] * math.pi / 180))
            rgb[r][1] = 255 * np.sin(r + (theta[1] * math.pi / 180))
            rgb[r][2] = 255 * np.sin(r + (theta[2] * math.pi / 180))

        for r in range(0, rows):
            for c in range(0, cols):
                intensity = image[r][c]
                idx = int(intensity / interval)
                color[r][c] = rgb[idx]

        return color
