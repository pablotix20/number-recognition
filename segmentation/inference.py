import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from read_numbers import read_numbers, binarize
from compress_pickle import load
from tools import show_inference, classify_numbers
from image_generation import gen_images, dataset


model = tf.keras.models.load_model('./gen/model')

img = cv2.imread('./gen/test5.jpg', 0)
img = cv2.resize(img, (288, 288))/255

# (train_x, train_y, val_x, val_y) = load(
#     './gen/mask_data.pickle', compression='lz4')

input = np.array([img]).reshape((1, 288, 288, 1))

# (x, y) = dataset[1]
# input = gen_images(1, 192, 192, 1, 4, 8, x, y)[0]

# input = val_x[:10]
# cv2.imwrite('./gen/test.png', val_x[0]*255)

inference = model(input)

processed = np.argmax(inference[0], axis=2)

# processed = processed.reshape(-1, 192, 192, 1)
# cv2.imshow('output', processed)

# fig = plt.figure(figsize=(8, 8))
# fig.add_subplot(3, 2, 1)
# plt.imshow(input[0])
# fig.add_subplot(3, 2, 2)
# plt.imshow(processed[0])
# for i in range(3):
#     fig.add_subplot(3, 2, 3+i)
#     plt.imshow(inference[0, :, :, i*3: (i+1)*3])
# # plt.savefig('./gen/foo.png')
# plt.show()

# classify_numbers(inference[0])

show_inference(input[0], inference[0])

# bin_labels = binarize(inference[0])
# read_numbers(img, bin_labels)

exit()

rows = 4
axes = []
fig = plt.figure()
for a in range(rows):
    axes.append(fig.add_subplot(rows, 4, a*4+1))
    plt.imshow(val_x[a])
    plt.axis('off')

for a in range(rows):
    axes.append(fig.add_subplot(rows, 4, a*4+2))
    plt.imshow(val_y[a, :, :, :3]*255)
    plt.axis('off')

for a in range(rows):
    axes.append(fig.add_subplot(rows, 4, a*4+3))
    plt.imshow(inference[a, :, :, :3])
    plt.axis('off')

for a in range(rows):
    axes.append(fig.add_subplot(rows, 4, a*4+4))
    plt.imshow(processed[a, :, :])
    plt.axis('off')
fig.tight_layout()
# plt.savefig('./gen/foo.png')
plt.show()
