import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from compress_pickle import load

from tensorflow.keras import layers, models
from tensorflow.python.ops.numpy_ops.np_math_ops import inner

BATCH_SIZE = 20
EPOCHS = 3

# with open('./gen/mask_data.pickle', 'rb') as handle:
#     (train_x, train_y) = pickle.load(handle)
(train_x, train_y, val_x, val_y) = load(
    './gen/mask_data.pickle', compression='lz4')

# print(train_x)


def get_model(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model


# input = layers.Input(shape=(128, 128, 1))
# inner_layer = layers.Conv2D(15, 5)(input)
# inner_layer = layers.MaxPooling2D(pool_size=(2, 2))(inner_layer)
# inner_layer = layers.BatchNormalization()(inner_layer)
# inner_layer = layers.ReLU()(inner_layer)

# # inner_layer = layers.Conv2D(11, 21)(inner_layer)
# # inner_layer = layers.MaxPooling2D(pool_size=(2,2))(inner_layer)
# # inner_layer = layers.BatchNormalization()(inner_layer)
# # inner_layer = layers.ReLU()(inner_layer)

# # # inner_layer = layers.Conv2D(5, 3)(inner_layer)
# inner_layer = layers.Flatten()(inner_layer)
# inner_layer = layers.Dense(1024, activation='relu')(inner_layer)
# # inner_layer = layers.Dropout(0.2)(inner_layer)
# inner_layer = layers.Dense(11*32*32)(inner_layer)
# inner_layer = layers.Reshape((32, 32, -1))(inner_layer)
# # inner_layer = layers.Conv2D(3, 3)(inner_layer)
# output = layers.Activation('softmax')(inner_layer)

# model = models.Model(inputs=input, outputs=output)
model = get_model((128, 128), 11)

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True,to_file='./gen/model.png')

# opt = tf.keras.optimizers.Adam(learning_rate=0.0002)

# model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# train_count = int(train_x.shape[0]*TRAIN_LEN)
# TODO: Sparse categorical
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
# model.fit(train_x[:train_count], train_y[:train_count], batch_size=BATCH_SIZE,
#           epochs=EPOCHS, validation_data=(train_x[train_count:], train_y[train_count:]))
model.fit(train_x, train_y, batch_size=BATCH_SIZE,
          epochs=EPOCHS, validation_data=(val_x, val_y))

model.save('./gen/model')

# inference = model(train_x[8000:8010])
inference = model(val_x[0:10])
processed = np.argmax(inference, axis=3)

# print(inference[0])

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
