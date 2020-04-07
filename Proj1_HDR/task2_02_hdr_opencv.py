import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def countTonemap(hdr, min_fraction=0.0005):
    counts, ranges = np.histogram(hdr, 256)
    min_count = min_fraction * hdr.size
    delta_range = ranges[1] - ranges[0]

    image = hdr.copy()
    for i in range(len(counts)):
        if counts[i] < min_count:
            image[image >= ranges[i + 1]] -= delta_range
            ranges -= delta_range

    return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)


def plot_responses(g_red, g_green, g_blue):
    """
    plot_responses() is a helper function which plots response to intensity.
    """
    pixel_range = [i for i in range(256)]
    plt.plot(g_red, pixel_range, label='red')
    plt.plot(g_green, pixel_range, label='green')
    plt.plot(g_blue, pixel_range, label='blue')

    plt.title('line chart')
    plt.xlabel('Log Exposure X')
    plt.ylabel('Pixel Value Z')

    plt.show()


if __name__ == '__main__':
    folder = './src_images'

    # Read all the files with OpenCV
    files = list([os.path.join(folder, f) for f in os.listdir(folder) if f[-3:] == 'png'])
    files = sorted(files)
    images = list([cv2.imread(f) for f in files])

    # read the exposure times in seconds from txt file
    exposures = list()
    with open(os.path.join(folder, 'list.txt'), 'r') as f:
        for line in f.readlines():
            fname, exposure = line.strip('\n').split(' ')
            exposures.append(1. / float(exposure))

    exposures = np.float32(exposures)
    # Compute the response curve
    calibration = cv2.createCalibrateDebevec()
    response = calibration.process(images, exposures)
    plot_responses(response[:, :, 0], response[:, :, 1], response[:, :, 2])

    # Compute the HDR image
    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, exposures, response)

    # Save it to disk
    cv2.imwrite('hdr_image.hdr', hdr)

    # tone-mapping using Durand operator
    durand = cv2.createTonemapDurand(gamma=2.5)
    ldr = durand.process(hdr)

    # or else:
    # ldr = countTonemap(hdr)

    # Tonemap operators create floating point images with values in the 0..1 range
    # This is why we multiply the image with 255 before saving
    cv2.imwrite('durand_image.png', ldr * 255)
