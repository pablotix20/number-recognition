from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from tqdm import tqdm
from compress_pickle import dump
import cv2
from max_pool import pool2d

TRAIN_LEN = 20000
WIDTH = 128
HEIGHT = 128
OUT_DOWNSCALING = 2

NUMBERS_PER_IMAGE = 6

(mnist_train_x, mnist_train_y), (mnist_val_x, mnist_val_y) = mnist.load_data()


def gen_images(TRAIN_LEN, HEIGHT, WIDTH, OUT_DOWNSCALING, MAX_NUMBERS, src_x, src_y):
    train_x = np.zeros((TRAIN_LEN, HEIGHT, WIDTH))
    mask_y = np.zeros(
        (TRAIN_LEN, int(HEIGHT/OUT_DOWNSCALING), int(WIDTH/OUT_DOWNSCALING)))
    # train_y = np.zeros((TRAIN_LEN, OUT_HEIGHT, OUT_WIDTH))

    for i in tqdm(range(TRAIN_LEN)):
        mask = np.zeros((HEIGHT, WIDTH))

        for n in range(MAX_NUMBERS):
            x = rnd.randint(0, WIDTH-28)
            y = rnd.randint(0, HEIGHT-28)
            img = rnd.randint(0, src_x.shape[0]-1)
            if np.count_nonzero(train_x[i, x:x+28, y:y+28]) == 0:
                train_x[i, x:x+28, y:y+28] = src_x[img]
                # submask = np.zeros((3), dtype=np.uint8)
                # submask[mnist_train_y[img] % 2] = 1
                # mask[x:x+28, y:y+28] = np.where(mnist_train_x[img]
                #                                 > 0, 1, 0) * (mnist_train_y[img] % 2+1)
                mask[x:x+28, y:y+28] = np.where(src_x[img]
                                                > 0, 1, 0) * (src_y[img]+1)
                # mask[x:x+28, y:y+28] = np.where(mnist_train_x[img] > 0, 1, 0)
                # mask[x:x+28, y:y+28] = (mnist_train_y[img] % 2+1)

                # scaled_x = int(x*OUT_WIDTH/WIDTH)
                # scaled_y = int(y*OUT_HEIGHT/HEIGHT)
                # scaled_size = int(28*OUT_WIDTH/WIDTH)
                # train_y[i, scaled_x:scaled_x+scaled_size, scaled_y:scaled_y +
                #         scaled_size] = mask
        # TODO: Replaced by MaxPool
        # mask_y[i] = cv2.resize(mask, (OUT_WIDTH, OUT_HEIGHT),
        #                        interpolation=cv2.INTER_NEAREST)
        mask_y[i] = pool2d(mask, stride=OUT_DOWNSCALING, padding=0,
                           kernel_size=OUT_DOWNSCALING, pool_mode='max')
        # train_y[i] = pool2d(mask, stride=4, padding=0,
        #                    kernel_size=4, pool_mode='max')

    train_y = np.zeros(
        (TRAIN_LEN, int(HEIGHT/OUT_DOWNSCALING), int(WIDTH/OUT_DOWNSCALING), 11))
    for i in range(11):
        submask = np.zeros((11), dtype=np.uint8)
        submask[i] = 1
        train_y[mask_y == i] = submask

    # train_y[mask_y==0]=[1,0,0]
    # train_y[mask_y==1]=[0,1,0]
    # train_y[mask_y==2]=[0,0,1]

    return (train_x/255, train_y)


(train_x, train_y) = gen_images(TRAIN_LEN, HEIGHT, WIDTH,
                                OUT_DOWNSCALING, NUMBERS_PER_IMAGE, mnist_train_x, mnist_train_y)
(val_x, val_y) = gen_images(int(TRAIN_LEN*.2), HEIGHT, WIDTH,
                            OUT_DOWNSCALING, NUMBERS_PER_IMAGE, mnist_val_x, mnist_val_y)
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
    axes[-1].set_title(f'asdf')
    plt.imshow(train_x[a])

for a in range(rows*cols):
    axes.append(fig.add_subplot(rows, cols*2, rows*cols + a+1))
    axes[-1].set_title(f'asdf')
    plt.imshow(train_y[a, :, :, :3])
fig.tight_layout()
plt.show()
