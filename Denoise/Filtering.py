import numpy as np
import math
import cv2


class Filtering:

    def __init__(self, image, filter_name, filter_size, var=None):
        """initializes the variables of spatial filtering on an input image
        takes as input:
        image: the noisy input image
        filter_name: the name of the filter to use
        filter_size: integer value of the size of the fitler
        global_var: noise variance to be used in the Local noise reduction filter
        S_max: Maximum allowed size of the window that is used in adaptive median filter
        """

        self.image = image

        if filter_name == 'arithmetic_mean':
            self.filter = self.get_arithmetic_mean
        elif filter_name == 'geometric_mean':
            self.filter = self.get_geometric_mean
        if filter_name == 'local_noise':
            self.filter = self.get_local_noise
        elif filter_name == 'median':
            self.filter = self.get_median
        elif filter_name == 'adaptive_median':
            self.filter = self.get_adaptive_median

        self.filter_size = filter_size
        self.global_var = var
        self.S_max = 15

    def mean(self, roi):
        return sum(roi) / len(roi)

    def get_arithmetic_mean(self, roi):
        """Computes the arithmetic mean of the input roi
        takes as input:
        roi: region of interest (a list/array of intensity values)
        returns the arithmetic mean value of the roi"""

        return self.mean(roi)

    def get_geometric_mean(self, roi):
        """Computes the geometric mean for the input roi
        takes as input:
        roi: region of interest (a list/array of intensity values)
        returns the geometric mean value of the roi"""

        prod = 1
        for i in range(0, len(roi)):
            prod *= roi[i]

        return math.pow(prod, (1 / len(roi)))

    def get_local_noise(self, roi):
        """Computes the local noise reduction value
        takes as input:
        roi: region of interest (a list/array of intensity values)
        returns the local noise reduction value of the roi"""
        lMean = self.mean(roi)
        lVariance = self.mean(np.array(roi) ** 2) - lMean ** 2
        nVariance = self.global_var
        center = roi[len(roi) // 2]

        return center - ((nVariance / lVariance) * (center - lMean))

    def get_median(self, roi):
        """Computes the median for the input roi
        takes as input:
        roi: region of interest (a list/array of intensity values)
        returns the median value of the roi"""

        roi.sort()

        return roi[len(roi) // 2]

    def get_adaptive_median(self, roi):
        """Computes the harmonic filter
                        takes as input:
        kernel: a list/array of intensity values
        order: order paramter for the
        returns the harmonic mean value in the current kernel"""

        wind = self.filter_size

        x = y = (self.S_max - 1) // 2

        Roi2D = np.reshape(roi, (self.S_max, self.S_max))
        yxz = Roi2D[y][x]

        while True:
            adaptive_roi = [0] * (wind * wind)
            padding = (wind - 1) // 2
            counter = 0
            for m in range(-padding, padding + 1):
                for n in range(-padding, padding + 1):
                    adaptive_roi[counter] = Roi2D[y + m][x + n]
                    counter += 1

            adaptive_roi.sort()
            ZMin = adaptive_roi[0]
            ZMax = adaptive_roi[-1]

            ZMid = adaptive_roi[len(adaptive_roi) // 2]

            a1 = ZMid - ZMin
            a2 = ZMid - ZMax

            if (a1 > 0) and (a2 < 0):
                b1 = yxz - ZMin
                b2 = yxz - ZMax

                if (b1 > 0) and (b2 < 0):
                    return yxz
                else:
                    return ZMid
            else:
                wind += 2
                if wind > self.S_max:
                    return ZMid

    def adaptive_filtering(self):

        rows, cols = np.shape(self.image)
        filteredImage = np.zeros((rows, cols))

        padding = (self.S_max - 1) // 2
        paddingImage = np.zeros((rows + padding * 2, cols + padding * 2))

        roi = [0] * (self.S_max * self.S_max)

        for r in range(0, rows):
            for c in range(0, cols):
                paddingImage[r + padding][c + padding] = self.image[r][c]

        for r in range(0, rows):
            for c in range(0, cols):
                y = r + padding
                x = c + padding
                c = 0
                for m in range(-padding, padding + 1):
                    for n in range(-padding, padding + 1):
                        roi[c] = paddingImage[y + m][x + n]
                        c += 1
                intensity = self.filter(roi)
                filteredImage[r][c] = intensity

        return filteredImage

    def filtering(self):
        """performs filtering on an image containing gaussian or salt & pepper noise
        returns the denoised image
        ----------------------------------------------------------
        Note: Here when we perform filtering we are not doing convolution.
        For every pixel in the image, we select a neighborhood of values defined by the kernal and apply a mathematical
        operation for all the elements with in the kernel. For example, mean, median and etc.

        Steps:
        1. add the necesssary zero padding to the noisy image, that way we have sufficient values to perform the operati
        ons on the pixels at the image corners. The number of rows and columns of zero padding is defined by the kernel size
        2. Iterate through the image and every pixel (i,j) gather the neighbors defined by the kernel into a list (or any data structure)
        3. Pass these values to one of the filters that will compute the necessary mathematical operations (mean, median, etc.)
        4. Save the results at (i,j) in the ouput image.
        5. return the output image

        Note: You can create extra functions as needed. For example if you feel that it is easier to create a new function for
        the adaptive median filter as it has two stages, you are welcome to do that.
        For the adaptive median filter assume that S_max (maximum allowed size of the window) is 15


        """

        if self.filter == self.get_adaptive_median:
            return self.adaptive_filtering()

        roi = [0] * (self.filter_size * self.filter_size)
        rows, cols = np.shape(self.image)
        padding = (self.filter_size - 1) // 2
        paddingImage = np.zeros((rows + padding * 2, cols + padding * 2))
        filterImage = np.zeros((rows, cols), dtype=np.uint8)

        for r in range(0, rows):
            for c in range(0, cols):
                paddingImage[r + padding][c + padding] = self.image[r][c]

        for r in range(0, rows):
            for c in range(0, cols):
                y = r + padding
                x = c + padding
                counter = 0
                for m in range(-padding, padding + 1):
                    for n in range(-padding, padding + 1):
                        roi[counter] = paddingImage[y + m][x + n]
                        counter += 1
                intensity = self.filter(roi)
                filterImage[r][c] = intensity

        return filterImage
