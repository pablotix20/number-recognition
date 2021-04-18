import numpy as np
from numpy.lib.stride_tricks import as_strided
import math
import matplotlib.pyplot as plt
import cv2 as cv2


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride*A.strides[0],
                              stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def show_images(images, titles):
    n = math.ceil(math.sqrt(len(images)))
    for i in range(len(images)):
        plt.subplot(n, n, i+1), plt.imshow(images[i], 'gray',)
        plt.title(titles[i])
        plt.axis(False)
    plt.show()


def show_inference(input, output):
    images = [input]
    titles = ['Input']
    for i in range(11):
        channel = np.array(output[:, :, i]*255, dtype='uint8')
        images.append(channel)
        if i == 0:
            titles.append('-')
        else:
            titles.append(f'{i-1}')
    show_images(images, titles)


def classify_numbers(input, output):
    numbers = []
    for i in range(11):
        channel = np.array(output[:, :, i]*255, dtype='uint8')
        images.append(cv2.Canny(channel, 10, 50))
