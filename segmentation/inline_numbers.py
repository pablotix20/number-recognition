import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pickle
from max_pool import pool2d

"""
Uing mnist dataset to create composite images with numbers in multiple lines
"""

MAX_LINES = 4
MAX_NUMS_LINE = 6

IN_SIZE = 28
OUT_SIZE_W = 192
OUT_SIZE_H = 192

OUT_DOWNSCALING = 2
TRAIN_LEN = 100

MAX_SCALE_FACTOR = 1.25
MIN_SCALE_FACTOR = 0.75

KERNEL_SIZE = 2

MIN_OFFSET = 10 #min offset between lines
OFF_Y_MU = 0
OFF_Y_SIGMA = 3 
JUMP_P = 0.2 #probability of jumping within a line and between lines

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def inline_img_gen(src_x, src_y):
    """
    inputs: 
    src_x: an image dataset with IN_SIZE x IN_SIZE 
    src_y: tags of src_x

    outputs:
    composite_imgs: image dataset with shape: [TRAIN_LEN, OUT_SIZE_H, OUT_SIZE_W, 1]
    composite_img_tags: image of the tags of the dataset with shape: [TRAIN_LEN, OUT_SIZE_H//OUT_DOWNSCALING, OUT_SIZE_W//OUT_DOWNSCALING, 1]
    composite_tags: numerical tags of the image as binary row vectors: [TRAIN_LEN, OUT_SIZE_H//OUT_DOWNSCALING, OUT_SIZE_W//OUT_DOWNSCALING, 11]

    """
    composite_imgs = np.zeros((TRAIN_LEN, OUT_SIZE_H, OUT_SIZE_W), dtype = 'uint8') # 128 x 128 x 1 imgs
    composite_img_tags = np.zeros((TRAIN_LEN, OUT_SIZE_H//OUT_DOWNSCALING, OUT_SIZE_W//OUT_DOWNSCALING), dtype = 'uint8')
    composite_tags = np.zeros((TRAIN_LEN, OUT_SIZE_H//OUT_DOWNSCALING, OUT_SIZE_W//OUT_DOWNSCALING, 11), dtype = 'uint8')

    for p in range(TRAIN_LEN):
        new_img_x = np.full((OUT_SIZE_H, OUT_SIZE_W), 255, dtype = 'uint8') 
        temp_img_y = np.zeros((OUT_SIZE_H, OUT_SIZE_W), dtype = 'uint8')
        #new_img_y = np.zeros((OUT_SIZE_W, OUT_SIZE_H)//OUT_DOWNSCALING, dtype = 'uint8')

        lines = random.randint(1, MAX_LINES)
        #range = (0, OUT_SIZE_H - IN_SIZE * MAX_SCALE_FACTOR)

        #we need lines lines contained in range and with a minimum separation of IN_SIZE * MAX_SCALE_FACTOR
        play = OUT_SIZE_H - lines * IN_SIZE * MAX_SCALE_FACTOR
        min_off = MIN_OFFSET #lim offsets per line
        max_off = play//(lines)
        mode_off = (min_off+max_off) / MAX_SCALE_FACTOR #can be float

        for l in range(lines):
            y = int(random.triangular(min_off, mode_off, max_off) + l * IN_SIZE * MAX_SCALE_FACTOR 
                        + (random.random() < JUMP_P) * (random.random() * OUT_SIZE_H)) #coordinates of the start of the line
            x = random.randint(MIN_OFFSET, OUT_SIZE_W - IN_SIZE)
            nums = random.randint(1, MAX_NUMS_LINE)
            for _ in range(nums):

                i = random.randint(0, src_x.shape[0]-1)
                img_x = src_x[i]
                img_y = src_y[i]

                scale_x = random.uniform(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
                scale_y = random.uniform(MIN_SCALE_FACTOR, MAX_SCALE_FACTOR)
                dx = int(scale_x * IN_SIZE)
                dy = int(scale_y * IN_SIZE)
                img_res = cv2.resize(img_x, (dx, dy))

                if x+dx > OUT_SIZE_W or y+dy > OUT_SIZE_H:
                    break

                if np.all(new_img_x[y:y+dy, x:x+dx] == 255):
                    new_img_x[y:y+dy, x:x+dx] = cv2.bitwise_not(img_res)
                    temp_img_y[y:y+dy, x:x+dx] = np.where(img_res != 0, img_y + 1 , 0) #set num pixels to num+1

                # x and y for new iter
                x += int(dx + random.randint(MIN_OFFSET, 2 * MIN_OFFSET) + (random.random() < JUMP_P) * (random.random() * OUT_SIZE_W))
                y += int(random.normalvariate(OFF_Y_MU, OFF_Y_SIGMA))

                
        new_img_y = pool2d(temp_img_y, stride=OUT_DOWNSCALING, padding=0, kernel_size=KERNEL_SIZE, pool_mode='max')
        composite_imgs[p] = new_img_x
        composite_img_tags[p] = new_img_y

        for i in range(11):
            submask = np.zeros((11), dtype=np.uint8)
            submask[i] = 1
            composite_tags[composite_img_tags == i] = submask
    return (composite_imgs/255, composite_img_tags, composite_tags)


(composite_imgs, composite_img_tags, composite_tags) = inline_img_gen(x_train, y_train)


fig, ax = plt.subplots(4,5)
for i in range(2):
    for j in range(5):
        cur = ax[2*i, j]
        cur.imshow(composite_imgs[i*5+j], cmap = 'gray')
        cur.set_title(i*5 + j)
        cur.axis(False)
        cur2 = ax[1+2*i, j]
        cur2.imshow(composite_img_tags[i*5+j], cmap = 'gray')
        cur2.set_title(i*5 + j)
        cur2.axis(False)
plt.show()


filename = ".\composite_inline_dataset.pkl"
with open(filename, "wb") as f:
    pickle.dump((composite_imgs, composite_tags), f)

print('Done')
