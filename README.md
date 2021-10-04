# Number segmentation using TensorFlow
This project serves as a demonstration on a segmentation model (image in - image out) using TensorFlow.

The model does a per-pixel classification of number digits (0-9) or background for anything else. This can be postprocessed to recognize numbers in the image with precise location and shape information.

Training data is dynamically generated, for which random background images must be provided. The generation algorithm automatically places handwritten digits on top of these backgrounds, while creating labels and applying several transformations to increase realism.

## Segmentation model
The segmentation model follos a downscale-upscale process. With a 288x288x1 monochrome input to a 144x144x11 output. Each of the output layers represent the likelyness for that pixel to be a 0-9 digit or background.

![model](https://user-images.githubusercontent.com/10696506/135134308-8ae04ab7-f653-46cd-8224-36abdb47d59b.png)

## Training
Note that background images need to be provided, these are not included in the repository. Place images in the `segmentation/backgrounds` folder to be used.
To start training, run the following command inside the segmentation folder:

```
python train.py
```

You may adjust epoch count and batch size in the train.py file. The model is automatically saved after training.

## Results

You may run `realtime_inference.py` for a live demo using your webcam.

You can also run `inference.py` over a static image file, which produces an output such as:

![Figure_1](https://user-images.githubusercontent.com/10696506/135840659-ebffc062-98c8-4c8f-9562-48dea3e6f311.png)

Channels - and 0-9 correspond to the output of the net, bounding boxes are generated on post-processing.

## Tensorflow.js

The TF model can be converted to be used in TensorFlow for JavaScript. Note that the implementation provided is not finalished, and does not include features such as post-processing.
