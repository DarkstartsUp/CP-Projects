import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_images(directory, extension='png'):
    # Read all the files with OpenCV
    files = list([os.path.join(directory, f) for f in os.listdir(directory) if f[-3:] == extension])
    files = sorted(files)
    images = list([cv2.imread(f) for f in files])

    # read the exposure times in seconds from txt file
    exposures = list()
    with open(os.path.join(directory, 'list.txt'), 'r') as f:
        for line in f.readlines():
            fname, exposure = line.strip('\n').split(' ')
            exposures.append(1. / float(exposure))
    exposures = np.float32(exposures)

    return images, exposures


def compute_weights():
    """
    A helper function which computes the weights needed in the algorithm for each
    of the intensity values.  As of now, we've implemented the triangle function specified in the paper.
    :return: weights: a 1x256 vector with the corresponding weight for the intensity.
    """
    print('== Computing weights ==')
    weights = [i for i in range(256)]
    weights = [min(i, 255 - i) for i in weights]
    return weights


def sample_rgb_images(images):
    """
    A helper function which samples images to construct the Z matrix needed in
    gsolve.  Z is an NxP matrix, where N is the number of sampled pixels and P is the number of input
    images (exposures).  This is the RGB version, and returns one Z matrix per channel.
    :param images: a array of images to use, where each item in the array is one exposure.
    :return: z_red is the Z matrix for the red channel
             z_green is the Z matrix for the green channel
             z_blue is the Z matrix for the blue channel
    """
    print('== Sampling images to construct the Z matrices for each channel == ')
    num_exposures = len(images)    # Value of P.
    '''
    Number of samples should satisfy:     N(P-1) > Z_max - Z_min
    This means that we should have:       N > (Z_max - Z_min) / (P-1).
    We will use:                          N = 255 / (P-1) * 2, which satisfies the equation
    '''
    num_samples = round(255 / (num_exposures - 1) * 2)     # Value of N.

    # Calculate the indices we are going to sample from.  Assumes that all images are same size.
    img_pixels = images[0].shape[0] * images[0].shape[1]
    step = img_pixels / num_samples
    sample_indices = np.arange(0, img_pixels - 1, step).astype(np.int)

    # Preallocate space for results.
    z_red = np.zeros((num_samples, num_exposures))
    z_green = np.zeros((num_samples, num_exposures))
    z_blue = np.zeros((num_samples, num_exposures))

    # Sample the images.
    for i in range(num_exposures):
        sampled_red, sampled_green, sampled_blue = sample_exposure(images[i], sample_indices)
        z_red[:, i] = sampled_red
        z_green[:, i] = sampled_green
        z_blue[:, i] = sampled_blue

    return z_red, z_green, z_blue


def sample_exposure(image, sample_indices):
    """
    A helper function which samples the given image at the specified indices.
    :param image: a single RGB image to be sampled from
    :param sample_indices:  an array of the length N with the indices to sample at.  N is the number of pixels
    :return: sampled_red is an array of the length N with the sample from the red channel
             sampled_green is an array of the length N with the sample from the green channel
             sampled_blue is an array of the length N with the sample from the blue channel
    """
    # Get the constituent channels.
    red_img = image[:, :, 0].flatten()
    green_img = image[:, :, 1].flatten()
    blue_img = image[:, :, 2].flatten()

    # Construct the samples.
    sampled_red = [red_img[indice] for indice in sample_indices]
    sampled_green = [green_img[indice] for indice in sample_indices]
    sampled_blue = [blue_img[indice] for indice in sample_indices]

    return sampled_red, sampled_green, sampled_blue


def gsolve(Z,B,l,w):
    """
    Solve for imaging system response function.
    Given a set of pixel values observed for several pixels in several images with different exposure times,
    this function returns the imaging system's response function g as well as the log film irradiance
    values for the observed pixels.
    :param Z: Z(i,j) is the pixel values of pixel location number i in image j
    :param B: B(j) is the log delta t, or log shutter speed, for image j
    :param l: l is lamdba, the constant that determines the amount of smoothness
    :param w: w(z) is the weighting function value for pixel value z
    :return: g(z) is the log exposure corresponding to pixel value z
             lE(i) is the log film irradiance at pixel location i
    """
    n = 256
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros((A.shape[0], 1))

    # Include the data-fitting equations
    k = 1
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = w[int(Z[i][j])]
            A[k][int(Z[i][j]) + 1] = wij
            A[k][n + i] = -wij
            b[k][0] = wij * B[j]
            k = k + 1

    # Fix the curve by setting its middle value to 0
    A[k][129] = 1
    k = k + 1

    # Include the smoothness equations
    for i in range(n-2):
        A[k][i] = l * w[i + 1]
        A[k][i + 1] = -2 * l * w[i + 1]
        A[k][i + 2] = l * w[i + 1]
        k = k + 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b)[0]
    g = x.flatten()[:n]

    return g


def nesting(weight, array):
    height, width = array.shape[:2]
    if len(array.shape) > 2:
        num_channels = array.shape[2]
        nested = np.zeros((height, width, num_channels), dtype=float)
    else:
        num_channels = 1
        nested = np.zeros((height, width), dtype=float)

    for i in range(height):
        for j in range(width):
            if num_channels > 1:
                for k in range(num_channels):
                    nested[i][j][k] = weight[int(array[i][j][k])]
            else:
                nested[i][j] = weight[int(array[i][j])]

    return nested


def plot_radiance_map(rmap):
    height, width, num_channels = rmap.shape
    # thres = (np.max(rmap) - np.min(rmap)) * 0.001
    # rmap = (rmap + np.abs(np.min(rmap))) / thres * 255
    # rmap = (rmap - np.min(rmap)) / (np.max(rmap) - np.min(rmap)) * 255
    rmap = rmap / np.max(rmap) * 255
    for i in range(height):
        for j in range(width):
            for k in range(num_channels):
                rmap[i][j][k] = 0 if rmap[i][j][k] < 0 else rmap[i][j][k]
                rmap[i][j][k] = 255 if rmap[i][j][k] > 255 else rmap[i][j][k]

    # intensity = []
    # for i in range(height):
    #     for j in range(width):
    #         intensity.append(np.mean(rmap[i][j]))
    # plt.hist(intensity, bins=400, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.show()

    rmap = rmap.astype(np.uint8)
    cv2.imshow('radiance map', rmap)
    cv2.imwrite('radiance_map.jpg', rmap)
    cv2.waitKey(0)


def compute_hdr_map(images, g_red, g_green, g_blue, weights, ln_dt):
    """
    A helper function which creates the HDR radiance map.
    :param images: a cell array of the input images
    :param g_red: the camera response for the red channel
    :param g_green: the camera response for the green channel
    :param g_blue: the camera response for the blue channel
    :param weights: the weight vector to use
    :param ln_dt: the log of the exposure times
    :return: hdr_map: the HDR radiance map we are trying to compute
    """
    print("Computing HDR map")
    num_exposures = len(images)
    height, width, num_channels = images[0].shape   # Assume all images are the same size.

    numerator = np.zeros((height, width, num_channels), dtype=float)
    denominator = np.zeros((height, width, num_channels), dtype=float)
    curr_num = np.zeros((height, width, num_channels), dtype=float)

    for i in range(num_exposures):
        # Grab the current image we are processing and split into channels.
        curr_image = images[i].astype(np.float) + 1e-5   # Grab the current image.  Add 1e-5 to get rid of zeros.
        curr_red = curr_image[:, :, 0]
        curr_green = curr_image[:, :, 1]
        curr_blue = curr_image[:, :, 2]

        """
        Compute the numerator and denominator for this exposure.  Add to cumulative total.
                 sum_{j=1}^{P} (w(Z_ij)[g(Z_ij) - ln dt_j])
        ln E_i = ------------------------------------------
                         sum_{j=1}^{P} (w(Z_ij))
        """
        curr_weight = nesting(weights, curr_image)
        curr_num[:, :, 0] = curr_weight[:, :, 0] * (nesting(g_red, curr_red) - ln_dt[i])
        curr_num[:, :, 1] = curr_weight[:, :, 1] * (nesting(g_green, curr_green) - ln_dt[i])
        curr_num[:, :, 2] = curr_weight[:, :, 2] * (nesting(g_blue, curr_blue) - ln_dt[i])

        # Sum into the numerator and denominator.
        numerator = numerator + curr_num
        denominator = denominator + curr_weight

    ln_hdr_map = numerator / denominator
    hdr_map = np.exp(ln_hdr_map)

    # Plot radiance map.
    plot_radiance_map(ln_hdr_map)

    return hdr_map


def plot_responses(g_red, g_green, g_blue):
    """
    plot_responses() is a helper function which plots response to intensity.
    """
    pixel_range = [i for i in range(256)]
    plt.plot(g_red, pixel_range, label='red')
    plt.plot(g_green, pixel_range, label='green')
    plt.plot(g_blue, pixel_range, label='blue')

    plt.title('response to intensity')
    plt.xlabel('Log Exposure X')
    plt.ylabel('Pixel Value Z')

    plt.show()


def create_hdr_map(lamb, directory, extension):
    """
    A helper function which creates the HDR image from the provided directory of images.
    :param lamb: the smoothing factor to use
    :param directory: the path relative to input/ which contains the images.
    :param extension: the file extension of the images. Default is 'jpg'
    :return:
    """
    print('## Creating HDR image: ', directory)
    # Read in images and exposure times from directory.  Take the log of exposure time.
    images, exposure_times = read_images(directory, extension)
    ln_dt = np.log(exposure_times)

    # Sample the images appropriately, per color channel.
    z_red, z_green, z_blue = sample_rgb_images(images)

    # Compute the weighting function needed.
    weights = compute_weights()

    # Solve for the camera response for each color channel.
    print('== Computing camera response for each channel ==')
    g_red = gsolve(z_red, ln_dt, lamb, weights)
    g_green = gsolve(z_green, ln_dt, lamb, weights)
    g_blue = gsolve(z_blue, ln_dt, lamb, weights)

    # Plot response.
    plot_responses(g_red, g_green, g_blue)

    # Compute the HDR radiance map.
    hdr_map = compute_hdr_map(images, g_red, g_green, g_blue, weights, ln_dt)

    return hdr_map


if __name__ == '__main__':
    directory = './src_images'
    radiance_map = create_hdr_map(50, directory, 'png')
