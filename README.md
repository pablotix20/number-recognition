# Number segmentation using TensorFlow
This project serves as a demonstration on a segmentation model (image in - image out) using TensorFlow.

The model does a per-pixel classification of number digits (0-9) or background for anything else. This can be postprocessed to recognize numbers in the image with precise location and shape information.

Training data is dynamically generated, for which random background images must be provided. The generation algorithm automatically places handwritten digits on top of these backgrounds, while creating labels and applying several transformations to increase realism.

## Segmentation model

## Results
