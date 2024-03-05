# ana.rebeka.kamsek@ki.si, 2022

import numpy as np
import cv2


class Particle(object):
    """A class to represent particle information in a binary mask.

    The input paths to images containing binary masks can be used to read the particle
    masks. Alternatively, masks can be passed in the form of lists of OpenCV contours.
    For each mask, its real-space area, perimeter, circularity, equivalent sphere diameter
    and volume can be determined. The last two parameters assume spherical particles.
    Volumes can be sorted to determine characteristic equivalent sphere diameters with
    50 % or 90 % of the sample mass having smaller diameters, assuming a constant density.

    Attributes
    ----------
    mask_paths : list
        A list of paths to images of binary particle masks.

    mask_data : list
        A list with contours representing particles.

    calib : float
        Image calibration in real-space distance units per pixel.

    areas : ndarray of shape (number of masks,)
        Calibrated areas of the blobs on binary masks (particles).

    perimeters : ndarray of shape (number of masks,)
        Calibrated particle perimeters.

    circularities : ndarray of shape (number of masks,)
        Calculated particle circularities.

    diameters : ndarray of shape (number of masks,)
        Calculated calibrated particle diameters, assuming spherical particles.

    volumes : ndarray of shape (number of masks,)
        Calculated calibrated particle volumes, assuming spherical particles.
    """

    def __init__(self, mask_paths=None, mask_data=None, calib=1.0):
        """Constructs the attributes for the particle object.

        :param mask_paths: a list of paths to binary masks
        :param mask_data: a list of contours
        :param calib: image calibration in real-space distance units per pixel
        """

        if mask_data is None:
            mask_data = []
        if mask_paths is None:
            mask_paths = []
        self.mask_paths = mask_paths
        self.mask_data = mask_data
        self.calib = calib

        # attributes that will be determined later are set to None upon creating an instance of the Particle class
        self.areas = None
        self.perimeters = None
        self.circularities = None
        self.diameters = None
        self.volumes = None

    def read_masks(self):
        """Considers paths to binary masks and reads the files. Returns the masks or 0 if the paths are unsuitable.

        :return: 0 if the mask paths are not suitable, a list of masks otherwise
        """

        if self.mask_paths is None:
            print("The mask paths are not supplied as a list of strings.")
            return 0
        else:
            masks = []
            # read every mask using its path
            for mask_path in self.mask_paths:
                mask_temp = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_temp is None:
                    print("It was not possible to read the file at:", mask_path)

                # extract the blob (contour) from the mask and store it
                max_value = np.amax(mask_temp).astype("uint8")
                ret, thresh = cv2.threshold(mask_temp, max_value / 2, max_value, 0)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                masks.append(contours[0])

            self.mask_data = masks
            return masks

    def calculate_quantities(self):
        """Determines physical parameters for particles, read from binary masks.

        Considers the data either directly as contours or by reading files with specified paths.
        For each contour, it determines the calibrated areas, perimeters, circularities,
        equivalent sphere diameters and equivalent sphere volumes.

        :return: arrays with areas, perimeters, circularities, diameters, volumes
        """

        masks = []
        if len(self.mask_data) == 0:
            if self.read_masks() == 0:
                print("No masks supplied.")
            else:
                masks = self.read_masks()
        else:
            masks = self.mask_data

        areas, perimeters, circularities, diameters, volumes = [], [], [], [], []

        # determine calibrated physical parameters for each particle
        for i in range(len(masks)):
            area = cv2.contourArea(masks[i]) * (self.calib ** 2)
            perimeter = cv2.arcLength(masks[i], True) * self.calib
            circularity = 4 * np.pi * area / np.square(perimeter)
            diameter = perimeter / np.pi
            volume = 4 * np.pi * ((diameter * 0.5) ** 3 / 3)

            areas.append(area)
            perimeters.append(perimeter)
            circularities.append(circularity)
            diameters.append(diameter)
            volumes.append(volume)

        self.areas = areas
        self.perimeters = perimeters
        self.circularities = circularities
        self.diameters = diameters
        self.volumes = volumes

        return areas, perimeters, circularities, diameters, volumes

    def volume_parameters(self):
        """Determines d_50 and d_90 values to characterize the volume distribution of particles.

        Particle volumes are be sorted to determine characteristic equivalent sphere diameters
        with 50 % or 90 % of the sample mass having smaller diameters, assuming a constant density.
        This provides an approximate value and requires a large number of particles to be reliable.

        :return: d_50 and d_90 floats
        """

        # sort the equivalent sphere volumes and cumulative volumes in ascending order
        sorted_volumes = np.sort(self.volumes)
        entire_volume = np.sum(sorted_volumes)
        volume_sums = np.cumsum(sorted_volumes)

        # determine how many particles are below the threshold
        index_d50 = np.argwhere(volume_sums > entire_volume * 0.5)[0]
        index_d90 = np.argwhere(volume_sums > entire_volume * 0.9)[0]

        # sort the diameters and calculate the characteristic values
        sorted_diameters = np.sort(np.asarray(self.diameters))

        d_50 = np.average((sorted_diameters[index_d50 - 1], sorted_diameters[index_d50]))
        d_90 = np.average((sorted_diameters[index_d90], sorted_diameters[index_d90 - 1]))

        return d_50, d_90
