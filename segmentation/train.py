import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from compress_pickle import load

from tensorflow.keras import layers, models
from tensorflow.python.keras.backend import reshape
from data_generator import DataGenerator

BATCH_SIZE = 10
EPOCHS = 180

SEED = 6
# Set random seeds
tf.random.set_seed(SEED)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

# with open('./gen/mask_data.pickle', 'rb') as handle:
#     (train_x, train_y) = pickle.load(handle)

(train_x, train_y, val_x, val_y) = load(
    './gen/mask_data.pickle', compression='lz4')

# print(train_y.shape)
# exit()


def get_model(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(64, 5, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        # x = layers.Dropout(0.1)(x)
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
        # x = layers.Dropout(0.1)(x)
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
    # outputs = layers.Dense(96*96)(x)

    # outputs = layers.Reshape((96,96))(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model


def get_model_new(img_size, num_classes):
    inputs = tf.keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(64, 5, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual
    size_outputs = [x]

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for i, filters in enumerate([64, 128, 256]):
        # x = layers.Dropout(0.1)(x)
        x = layers.SeparableConv2D(filters, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        # x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
        size_outputs.append(x)

    ### [Second half of the network: upsampling inputs] ###

    for i, filters in enumerate([256, 128, 64]):
        # x = layers.Dropout(0.1)(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.UpSampling2D(2)(x)

        # Project residual
        # residual = layers.UpSampling2D(2)(previous_block_activation)
        # residual = layers.Conv2D(filters, 1, padding="same")(residual)
        residual2 = layers.Conv2D(
            filters, 1, padding="same")(size_outputs[-2-i])
        x = layers.add([x,  residual2])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        num_classes, 3, activation="softmax", padding="same")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


# model = get_model_new((288, 288), 11)
model = tf.keras.models.load_model('./gen/model')

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True, to_file='./gen/model.png')

# opt = tf.keras.optimizers.Adam(learning_rate=0.0002)

# model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# train_count = int(train_x.shape[0]*TRAIN_LEN)
# TODO: Sparse categorical
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy", metrics=['accuracy'])

# model.fit(train_x[:train_count], train_y[:train_count], batch_size=BATCH_SIZE,
#           epochs=EPOCHS, validation_data=(train_x[train_count:], train_y[train_count:]))

# model.fit(train_x, train_y, batch_size=BATCH_SIZE,
#           epochs=EPOCHS, validation_data=(val_x, val_y))

training_generator = DataGenerator(BATCH_SIZE, 1500, False)
validation_generator = DataGenerator(BATCH_SIZE, 300, True)
model.fit(x=training_generator, validation_data=validation_generator,
          epochs=EPOCHS, use_multiprocessing=False)

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
plt.savefig('./gen/val_out')
# plt.show()
