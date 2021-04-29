import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm
from tools import pool2d
import os
from background import get_random_background

"""
Uing mnist dataset to create composite images with numbers in multiple lines
"""

MAX_LINES = 5
MAX_NUMS_LINE = 9

#img sizes
IN_SIZE = 28 #square input images
OUT_SIZE_W = 192
OUT_SIZE_H = 192

OUT_DOWNSCALING = 2 #downscaling of output tags
TRAIN_LEN = 10

MAX_SCALE_FACTOR = 1.25 #rescale number factor
MIN_SCALE_FACTOR = 0.75

KERNEL_SIZE = OUT_DOWNSCALING #convolution kernel size

MIN_OFFSET = 10 #min offset between lines
MIN_OFFSET_X = 10 #min offset of the start of a line

OFF_X_MU = 5 #offset between numbers in a line
OFF_X_SIGMA = 5 

OFF_Y_MU = 0 #variation around the start y position of a line
OFF_Y_SIGMA = 3 

JUMP_P = 0.2 #probability of jumping within a line and between lines

NOISE_MAX_HALFWIDTH = 3 #size of the gaussian mask is at most 2*NOISE_MAX_HALFWIDTH +1
NOISE_P = 0.5 #probability of adding gaussian noise 

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()



def inline_img_gen(src_x, src_y):
    """
    inputs: 
    src_x: an image dataset of sizes IN_SIZE x IN_SIZE 
    src_y: tags of src_x

    outputs:
    composite_imgs: image dataset with shape: [TRAIN_LEN, OUT_SIZE_H, OUT_SIZE_W, 1]
    composite_img_tags: image of the tags of the dataset with shape: [TRAIN_LEN, OUT_SIZE_H//OUT_DOWNSCALING, OUT_SIZE_W//OUT_DOWNSCALING, 1]
    composite_tags: numerical tags of the image as binary row vectors: [TRAIN_LEN, OUT_SIZE_H//OUT_DOWNSCALING, OUT_SIZE_W//OUT_DOWNSCALING, 11]

    """
    composite_imgs = np.zeros((TRAIN_LEN, OUT_SIZE_H, OUT_SIZE_W), dtype = 'uint8') # 128 x 128 x 1 imgs
    composite_img_tags = np.zeros((TRAIN_LEN, OUT_SIZE_H//OUT_DOWNSCALING, OUT_SIZE_W//OUT_DOWNSCALING), dtype = 'uint8')
    composite_tags = np.zeros((TRAIN_LEN, OUT_SIZE_H//OUT_DOWNSCALING, OUT_SIZE_W//OUT_DOWNSCALING, 11), dtype = 'uint8')

    for p in tqdm(range(TRAIN_LEN)):
        new_img_x = get_random_background(OUT_SIZE_H, OUT_SIZE_W).astype(np.int16)
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
                        + (random.random() < JUMP_P) * (random.random() * OUT_SIZE_H)/2) #coordinates of the start of the line
            x = random.randint(MIN_OFFSET_X, OUT_SIZE_W - IN_SIZE)
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

                if np.all(temp_img_y[y:y+dy, x:x+dx] == 0):
                    new_img_x[y:y+dy, x:x+dx] -= (img_res * 0.7).astype('uint8')
                    temp_img_y[y:y+dy, x:x+dx] = np.where(img_res != 0, img_y + 1 , 0) #set num pixels to num+1

                # x and y for new iter
                x += int(dx + random.normalvariate(OFF_X_MU, OFF_X_SIGMA) + (random.random() < JUMP_P) * (random.random() * OUT_SIZE_W))
                y += int(random.normalvariate(OFF_Y_MU, OFF_Y_SIGMA) + (random.random() < JUMP_P) * (random.random() * OUT_SIZE_H)/2)

                
        new_img_y = pool2d(temp_img_y, stride=OUT_DOWNSCALING, padding=0, kernel_size=KERNEL_SIZE, pool_mode='max')

        #add gaussian noise with probability NOISE_P
        if random.uniform(0, 1) < NOISE_P:
            new_img_x = cv2.GaussianBlur(new_img_x, (random.randint(1,NOISE_MAX_HALFWIDTH)*2+1, random.randint(0,NOISE_MAX_HALFWIDTH)*2+1), 0)
        
        new_img_x = np.clip(new_img_x, 0, 255).astype('uint8')

        composite_imgs[p] = new_img_x
        composite_img_tags[p] = new_img_y

    for i in tqdm(range(11)):
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

exit()

filename = ".\composite_inline_dataset.pkl"
with open(filename, "wb") as f:
    pickle.dump((composite_imgs, composite_tags), f)

print('Done')
