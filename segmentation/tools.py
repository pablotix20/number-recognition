import numpy as np
from numpy.lib.stride_tricks import as_strided
import math
import matplotlib.pyplot as plt
import cv2 as cv2
from tensorflow.python.keras.backend import dtype


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


def show_inference(input, inference):
    output = classify_numbers(cv2.merge([input, input, input]), inference)[1]
    images = [input, output]
    titles = ['Input', 'Output']
    for i in range(11):
        channel = np.array(inference[:, :, i]*255, dtype='uint8')
        images.append(channel)
        if i == 0:
            titles.append('-')
        else:
            titles.append(f'{i-1}')

    show_images(images, titles)


def drawBoundingBox(image, label, box):
    (x, y, w, h) = box
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255))

    (text_width, text_height) = cv2.getTextSize(
        label, 0, fontScale=0.5, thickness=1)[0]
    # make the coords of the box with a small padding of two pixels
    box_coords = ((x, y), (x + text_width + 2, y - text_height - 2))
    cv2.rectangle(image, box_coords[0],
                  box_coords[1], (0, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (x+1, y-1), 0, fontScale=0.5,
                color=(255, 0, 0), thickness=1)


NUM_THRESHOLD = 0.4
MIN_SIZE = 20


def classify_numbers(input, inference):
    numbers = []
    binary = np.array(
        np.array(inference[:, :, :]) > NUM_THRESHOLD, dtype='uint8')
    output = np.array(input)

    for i in range(10):
        numbers.append([])
        num_labels, img_labels, stats, cg = cv2.connectedComponentsWithStats(
            binary[:, :, i+1])
        for region in stats:
            if region[4] < MIN_SIZE:
                continue
            numbers[i].append(region)

            drawBoundingBox(output, f'{i}', region[:4]*2)
    return numbers, output
