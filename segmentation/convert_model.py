import tensorflow as tf
import tensorflowjs as tfjs

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(
    './gen/model')  # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('./gen/model.tflite', 'wb') as f:
    f.write(tflite_model)
tfjs.converters.load_keras_model
tfjs.converters.convert_tf_saved_model('./gen/model', './gen/model_js')
# model = tf.keras.models.load_model('./gen/model')
# tfjs.converters.save_keras_model(model, './gen/model_js')
