from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from tqdm import tqdm
from compress_pickle import dump
import cv2
from max_pool import pool2d
from background import get_random_background

TRAIN_LEN = 12
VAL_LEN = 12
WIDTH = 192
HEIGHT = 192
OUT_DOWNSCALING = 1

BATCHES_PER_IMAGE = 10
NUMBERS_PER_BATCH = 6

dataset = mnist.load_data()
(mnist_train_x, mnist_train_y), (mnist_val_x, mnist_val_y) = dataset


def gen_images(TRAIN_LEN, HEIGHT, WIDTH, OUT_DOWNSCALING, MAX_NUMBER_BATCHES, MAX_NUMBERS_PER_BATCH, src_x, src_y):
    train_x = np.zeros((TRAIN_LEN, WIDTH, HEIGHT))
    mask_y = np.zeros(
        (TRAIN_LEN, int(HEIGHT/OUT_DOWNSCALING), int(WIDTH/OUT_DOWNSCALING)))
    # train_y = np.zeros((TRAIN_LEN, OUT_HEIGHT, OUT_WIDTH))

    for i in tqdm(range(TRAIN_LEN)):
        mask = np.zeros((HEIGHT, WIDTH))
        min_y = 0

        for n in range(MAX_NUMBER_BATCHES):
            y = rnd.randint(min_y, min_y + HEIGHT//2)
            x = rnd.randint(0, WIDTH//2)
            resized_h = resized_w = int(28*(rnd.random()*2+.7))

            for m in range(MAX_NUMBERS_PER_BATCH):
                resized_w = int(resized_w*(rnd.random()*.25+.875))
                resized_h = int(resized_h*(rnd.random()*.25+.875))

                x += rnd.randint(int(resized_w*.5), int(resized_w*1.25))
                y += rnd.randint(-int(resized_h*.2), int(resized_h*.2))
                if y < 0:
                    y = 0

                # Choose random number
                img = rnd.randint(0, src_x.shape[0]-1)

                # Make numbers thinner
                kernel = np.ones((2, 2), np.uint8)
                src = cv2.erode(src_x[img], kernel, iterations=1)
                src = cv2.resize(src, (resized_w, resized_h))

                # Check position inside bounds
                if (x+resized_w+1) > train_x.shape[1] or (y+resized_h+1) > train_x.shape[2]:
                    continue

                # print(train_x.shape)
                # print(x+resized_w, y+resized_h)
                if np.all(train_x[i, y:y+resized_h, x:x+resized_w] == 0) or True:
                    train_x[i, y:y+resized_h, x:x+resized_w] -= src * \
                        (rnd.random()/2+.5)

                    # print((src > 0) & (mask[y:y + resized_h, x:x+resized_w, ] == 0))
                    # print(src > 0 and mask[y:y + resized_h, x:x+resized_w, ] == 0)
                    # mask[y:y + resized_h, x:x+resized_w, ] = np.where(
                    #     (src > 0) &
                    #     (mask[y:y + resized_h, x:x+resized_w, ] == 0),
                    #     (src_y[img]+1), 0)

                    mask_zone = mask[y:y + resized_h, x:x+resized_w, ]
                    mask_zone[src > 10] = src_y[img]+1
                    # mask_zone += np.where((src > 0),(src_y[img]+1), 0)

                    # mask[mask[y:y + resized_h, x:x+resized_w, ] == 0] = np.where(
                    #     (src > 0),(src_y[img]+1), 0)

                    if min_y < y+resized_h:
                        min_y = y+resized_h

        # TODO: Replaced by MaxPool
        # mask_y[i] = cv2.resize(mask, (OUT_WIDTH, OUT_HEIGHT),
        #                        interpolation=cv2.INTER_NEAREST)
        mask_y[i] = pool2d(mask, stride=OUT_DOWNSCALING, padding=0,
                           kernel_size=OUT_DOWNSCALING, pool_mode='max')
        # train_y[i] = pool2d(mask, stride=4, padding=0,
        #                    kernel_size=4, pool_mode='max')
        # train_x[i] += np.array(np.random.normal(0, 25, (WIDTH, HEIGHT)))
        train_x[i] += get_random_background(WIDTH, HEIGHT)
        # if rnd.randint(0, 1) == 0:
        #     # train_x[i] = 255-train_x[i]
        #     train_x[i] = cv2.GaussianBlur(train_x[i], (3, 3), 0)
        train_x[i] = np.clip(train_x[i], 0, 255).astype('uint8')

    # train_y = np.zeros(
    #     (TRAIN_LEN, int(HEIGHT/OUT_DOWNSCALING), int(WIDTH/OUT_DOWNSCALING), 11), dtype=np.uint8)
    # for i in tqdm(range(11)):
    #     submask = np.zeros((11), dtype=np.uint8)
    #     submask[i] = 1
    #     train_y[mask_y == i] = submask
    train_y = tf.keras.utils.to_categorical(
        mask_y, num_classes=11, dtype='uint8')

    # train_y = np.zeros(
    #     (TRAIN_LEN, int(HEIGHT/OUT_DOWNSCALING), int(WIDTH/OUT_DOWNSCALING)))
    # train_y = mask_y

    # train_y[mask_y==0]=[1,0,0]
    # train_y[mask_y==1]=[0,1,0]
    # train_y[mask_y==2]=[0,0,1]

    return (train_x/255, train_y)


if __name__ == "__main__":
    (train_x, train_y) = gen_images(TRAIN_LEN, HEIGHT, WIDTH,
                                    OUT_DOWNSCALING, BATCHES_PER_IMAGE, NUMBERS_PER_BATCH, mnist_train_x, mnist_train_y)
    (val_x, val_y) = gen_images(VAL_LEN, HEIGHT, WIDTH,
                                OUT_DOWNSCALING, BATCHES_PER_IMAGE, NUMBERS_PER_BATCH, mnist_val_x, mnist_val_y)
    # with open('./gen/mask_data.pickle', 'wb') as handle:
    #     pickle.dump((train_x, train_y), handle)
    dump((train_x, train_y, val_x, val_y),
         './gen/mask_data.pickle', compression="lz4")

    rows = 3
    cols = 3
    axes = []
    fig = plt.figure()
    for a in range(rows*cols):
        axes.append(fig.add_subplot(rows, cols*2, a+1))
        plt.imshow(train_x[a])
        plt.axis('off')

    for a in range(rows*cols):
        axes.append(fig.add_subplot(rows, cols*2, rows*cols + a+1))
        # plt.imshow(train_y[a, :, :])
        plt.imshow(train_y[a, :, :, :3]*255)
        plt.axis('off')
    fig.tight_layout()
    plt.show()
