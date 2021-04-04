import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers, models

BATCH_SIZE = 10
EPOCHS = 5

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_test_original = y_test

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Add a channels dimension
# x_train = x_train[..., tf.newaxis].astype("float32")
# x_test = x_test[..., tf.newaxis].astype("float32")

print(y_train)

input = layers.Input(shape=(28, 28, 1))
inner_layer = layers.Conv2D(5, 3)(input)
inner_layer = layers.Flatten()(inner_layer)
output = layers.Dense(10, activation='softmax')(inner_layer)

model = models.Model(inputs=input, outputs=output)

model.summary()

# opt = tf.keras.optimizers.Adam(learning_rate=0.0002)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=BATCH_SIZE,
          epochs=EPOCHS, validation_data=(x_test, y_test))

inference = model(x_test)
inference_label = np.argmax(inference, axis=1)
print(inference)

rows = 3
cols = 3
axes = []
fig = plt.figure()
for a in range(rows*cols):
    axes.append(fig.add_subplot(rows, cols, a+1))
    lbl = inference_label[a]
    axes[-1].set_title(f'{lbl} {inference[a,lbl]*100 : .2f}% - {y_test_original[a]}')
    plt.imshow(x_test[a])
fig.tight_layout()
plt.show()
