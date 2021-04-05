import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random 
import pickle

MAX_NUMS = 6
N = 10000

IN_SIZE = 28
OUT_SIZE = 128

IN_OFFSET = IN_SIZE - 1

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

composite_img = np.zeros((N, OUT_SIZE, OUT_SIZE), dtype = 'uint8') # 128 x 128 x 1 imgs
composite_tags = np.zeros((N, OUT_SIZE, OUT_SIZE), dtype = 'uint8')

for p in range(N):
    new_in_img = np.zeros((OUT_SIZE,OUT_SIZE), dtype = 'uint8')
    new_out_img = np.full((OUT_SIZE,OUT_SIZE), 255, dtype = 'uint8') 

    for i in range(MAX_NUMS):
        pos = random.randint(0,x_train.shape[0])
        n = y_train[pos]
        in_img = x_train[pos]
        (x, y) = (random.randint(0, OUT_SIZE - IN_SIZE), random.randint(0, OUT_SIZE - IN_SIZE))

        if(new_out_img[x, y] == 255 and new_out_img[x + IN_OFFSET, y] == 255 and
           new_out_img[x, y + IN_OFFSET] == 255 and new_out_img[x + IN_OFFSET, y + IN_OFFSET] == 255):
            new_out_img[x:x+IN_SIZE, y:y+IN_SIZE] = n
            new_in_img[x:x+IN_SIZE, y:y+IN_SIZE] = in_img

    composite_img[p] = new_in_img
    composite_tags[p] = new_out_img
        
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.imshow(new_out_img, cmap = 'gray')
    #plt.subplot(1,2,2)
    #plt.imshow(new_in_img, cmap = 'gray')
    #plt.show()

fig, ax = plt.subplots(4,5)
for i in range(2):
    for j in range(5):
        cur = ax[2*i, j]
        cur.imshow(composite_img[i*5+j], cmap = 'gray')
        cur.set_title(i*5 + j)
        cur.axis(False)
        cur2 = ax[1+2*i, j]
        cur2.imshow(composite_tags[i*5+j], cmap = 'gray')
        cur2.set_title(i*5 + j)
        cur2.axis(False)
plt.show()

filename = "composite_dataset.pkl"
with open(filename, "wb") as f:
    pickle.dump((composite_img, composite_tags), f)

print("Done")