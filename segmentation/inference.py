import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from compress_pickle import load

model = tf.keras.models.load_model('./gen/model')

img = cv2.imread('./gen/test_img.png',0)/255

# (train_x, train_y, val_x, val_y) = load(
#     './gen/mask_data.pickle', compression='lz4')

input = np.array([img]).reshape((1,128,128,1))
# input = val_x[:1]
# cv2.imwrite('./gen/test.png', val_x[0]*255)
print(input.shape)

inference = model(input)[0]
processed = np.argmax(inference, axis=2)

fig=plt.figure(figsize=(8, 8))
fig.add_subplot(1,2,1)
plt.imshow(input[0])
fig.add_subplot(1,2,2)
plt.imshow(processed)
plt.show()

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
    plt.imshow(val_y[a, :, :, :3])
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
plt.show()