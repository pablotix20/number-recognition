import cv2
import os
import random as rnd


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            img = cv2.resize(img, (256, 256))
            images.append(img)
    return images


images = load_images_from_folder('./backgrounds')


def get_random_background(width, height):
    img = images[rnd.randint(0, len(images)-1)]
    x = rnd.randint(0, img.shape[1]-width)
    y = rnd.randint(0, img.shape[0]-height)
    return img[y:y+height, x:x+width]
