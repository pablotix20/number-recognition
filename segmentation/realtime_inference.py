import cv2
from numpy.core.numeric import outer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from compress_pickle import load
from tools import classify_numbers

model = tf.keras.models.load_model('./gen/model')
testimg = cv2.imread('./gen/photo_2021-04-05_17-14-39.jpg', 0)


def inference(img):
    # print(img.shape)
    resized = cv2.resize(img, (288, 288))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    # print(img.shape)
    # img = np.array(testimg)
    scaled = gray/255
    # print(img.shape)
    # print(img)

    input = np.array([scaled]).reshape((1, 288, 288, 1))
    cv2.imshow('input', cv2.resize(input[0], (512, 512)))

    inference = model(input)[0]
    processed = np.argmax(inference, axis=2)
    unique, counts = np.unique(processed, return_counts=True)
    print(dict(zip(unique, counts)))

    _, output = classify_numbers(resized, inference)

    return processed, output


cam = cv2.VideoCapture(0)

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("cam", frame)
    (inf, output) = inference(frame)
    # print(output)
    inf = inf.reshape(144, 144, 1)*25.5
    inf = cv2.resize(inf, (512, 512))
    cv2.imshow('Inference', inf)
    print(output.shape)
    cv2.imshow('Output', cv2.resize(output, (512, 512)))
    # print(frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
