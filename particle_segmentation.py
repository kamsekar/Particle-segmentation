# ana.rebeka.kamsek@ki.si, 2022

import numpy as np
import cv2
from scipy.signal import argrelextrema
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


def extract_histogram(image):
    """Extracts the intensity histogram from a grayscale image.

    :param image: a grayscale image
    :return: a flattened intensity histogram
    """

    max_value = np.iinfo(image.dtype).max
    hist = cv2.calcHist([image], [0], None, [max_value], [0, max_value])
    cv2.normalize(hist, hist)

    return hist.flatten()


def support_removal(image, show_hist=False):
    """Sets the pixels outside of particle regions to zero.

    Uses an intensity histogram to determine the threshold value. Works when the atomic number of
    the support is significantly lower than that of the particles and when there are no significant
    specimen thickness variations, i.e., when the support is uniform.
    :param image: input image
    :param show_hist: a Boolean value to determine whether the image histogram should be displayed
    :return: a thresholded image with particle areas retaining their original values
    """

    hist = extract_histogram(image)

    # cut everything with a lower intensity than the second minimum
    # zero values are common when the image includes a vacuum region
    local_min = argrelextrema(hist, np.less)
    second_min = int(str(local_min)[14:16])

    # optional display of a histogram and key values used for thresholding
    ax = plt.subplot(111)
    ax.plot(hist)
    plt.title("Histogram")
    plt.xlabel("Intensity (pixel value)")
    plt.ylabel("Frequency (share of pixels)")
    plt.vlines(second_min, 0, np.amax(hist) * 1.05, colors='red')
    if show_hist:
        plt.show()
    else:
        plt.close()

    # setting pixels with too low or too high values to zero
    _, thresholded = cv2.threshold(image, 0, second_min, cv2.THRESH_TOZERO)

    return thresholded


def particle_thresholding(image):
    """Constructs a binary mask containing all particles in an image and the corresponding background area.

    :param image: input image
    :return: a mask with all particles, an image with the background area
    """

    # preparation for segmentation and contouring
    image_2 = support_removal(image)
    _, thresh = cv2.threshold(image_2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # determine background area, dilation increases object boundaries towards the background
    background = cv2.dilate(opening, kernel, iterations=1)
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = cv2.normalize(background, background, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return mask, background


def watershed_segmentation(image):
    """Performs watershed segmentation on a thresholded image containing supported particles.

    Determines the foreground and the background for a given image, assigns the labels to objects
    and performs the watershed algorithm. Works well for appropriately preprocessed images, where
    the particle signal is significantly more prominent than the rest.
    :param image: input image
    :return: two images representing the distances and particle markers
    """

    mask_particles, background = particle_thresholding(image)

    # determining the foreground
    dist_transform = cv2.distanceTransform(mask_particles, cv2.DIST_L2, 5)
    _, foreground = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), np.iinfo(image.dtype).max, 0)

    # determine the border between the foreground and the background
    foreground = np.uint8(foreground)
    unknown = cv2.subtract(background, foreground)

    # label the background as 1, other objects with larger ints, and unknown as 0
    ret3, markers = cv2.connectedComponents(foreground)
    markers = markers + 1
    markers[unknown == 255] = 0

    # mark the boundary region with -1, apply watershed
    image_3 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image_3, markers)

    return dist_transform, markers


def make_individual_masks(mask_particles, signal_type="uint8"):
    """Takes an image, finds contours in it, and converts them to masks of individual particles.

    :param mask_particles: an image with all segmented particles
    :param signal_type: data type of the original image
    :return: individual particle masks
    """

    # find contours in the image and make masks from them
    contours = cv2.findContours(mask_particles, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    # loop over the contours and convert them to instance masks
    masks = []
    for i, contour in enumerate(contours):
        blank = np.ones(mask_particles.shape[:2], dtype=signal_type) * np.iinfo(signal_type).max
        cv2.drawContours(blank, [contour], -1, 0, -1)
        blank = ~blank
        masks.append(blank)
    masks = np.asarray(masks)

    return masks


def blur_subtract(image, kernel_size=10):
    """Blurs grayscale image with a box blur and subtracts the blurred image from the original.

    :param image: input image to be filtered
    :param kernel_size: kernel size for blurring
    :return: filtered image
    """

    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    dst = cv2.filter2D(image, -1, kernel)

    diff = cv2.subtract(image, dst)
    return diff


def adaptive_segmentation(image):
    """Segments particles on an image with an uneven background.

    Uses a combination of unsharp masking, blurring, morphological operations and thresholding
    to remove the background. The result is used to determine particle regions with smoothened edges.
    It returns a binary mask containing all particle regions. Works well on moderate atomic number
    differences between the support and the particles, including when the support thickness is not
    uniform. Occasional discrepancies can arise when considering very small particles.

    :param image: original image
    :return: a binary mask containing the segmented particles
    """

    # unsharp masking as preprocessing, kernel size sufficiently large to retain particles
    preprocessed = blur_subtract(image, kernel_size=80)

    # noise removal with Gaussian blurring
    blurred = cv2.GaussianBlur(preprocessed, ksize=(7, 7), sigmaX=0)

    # erosion with a small kernel to remove only random pixels
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.morphologyEx(blurred, cv2.MORPH_ERODE, kernel, iterations=1)

    # set the dark parts to black, the rest remains unchanged
    _, tozero_first = cv2.threshold(eroded, 8, 255, cv2.THRESH_TOZERO)
    tozero_first = cv2.convertScaleAbs(tozero_first, alpha=1, beta=30)
    _, tozero_second = cv2.threshold(tozero_first, 30, 255, cv2.THRESH_TOZERO)

    opened = cv2.morphologyEx(tozero_second, cv2.MORPH_OPEN, kernel, iterations=5)
    dilated = cv2.morphologyEx(opened, cv2.MORPH_DILATE, kernel, iterations=1)

    # adaptive thresholding
    adapted = cv2.adaptiveThreshold(dilated, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 0)

    # opening and closing to eliminate trailing pixels and small holes
    opened_2 = cv2.morphologyEx(adapted, cv2.MORPH_OPEN, kernel, iterations=3)
    closed = cv2.morphologyEx(opened_2, cv2.MORPH_CLOSE, kernel, iterations=1)

    # find contours in the thresholded image
    contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # smoothening contours
    smoothened = []
    for contour in contours:
        x, y = contour.T
        x = x.tolist()[0]
        y = y.tolist()[0]

        # approximate the pixelated line shape with a curve
        tck, u = splprep((x, y), s=1.1, per=1)
        u_new = np.linspace(u.min(), u.max(), 25)
        x_new, y_new = splev(u_new, tck, der=0)

        # convert it back to the numpy format
        converted = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        smoothened.append(np.asarray(converted, dtype=np.int32))

    # overlay the smoothed contours on the original image
    binary_mask = cv2.drawContours(np.zeros(image.shape), smoothened, -1, (255, 255, 255), -1)
    cv2.imshow("Smoothened edges", binary_mask)
    cv2.waitKey(0)

    return binary_mask
