import numpy as np
import matplotlib.pyplot as plt
from compress_pickle import load
import cv2

X_OFF_WEIGHT = 1
Y_OFF_WEIGHT = 3
MAX_SQ_DISTANCE = 25**2
OUT_DOWNSCALING_X = 2
OUT_DOWNSCALING_Y = 2

dx, dy = 3, 3  # extra for rectangles

MIN_SIZE = 10

OPENING = False
OPENING_KERNEL = np.array(
    [[0, 1, 0],
     [1, 1, 1],
     [0, 1, 0]],
    np.uint8)


def nearest_nonzero_idx(a, x, y):
    r, c = np.nonzero(a)  # indeices of nonzero elemenyts in array a
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]


def binarize(inference):
    """
    inference: n*m*11
    """
    processed = np.argmax(inference, axis=2)
    layers = np.zeros_like(inference).astype(np.uint8)
    for i in range(11):
        layers[:, :, i] = processed == i

    return layers


def read_numbers(img, img_labels_bin, title=None, show=True):
    """
    img is the original image, either in color or greyscale
    img_labels_bis is the binarized output of the CNN

    Outputs the tags and its bounding boxes. 
    Shows the image with bounding boxes and tags.
    """
    j = 0 #number of numbers detected
    val_and_centroids = []

    for n in range(10):
        img_bin = img_labels_bin[:, :, n+1]
        if OPENING:
            img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, OPENING_KERNEL)

        num_nums, img_labels, stats, centroids = cv2.connectedComponentsWithStats(
            img_bin, connectivity=4)
        #    cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
        #    cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
        #    cv2.CC_STAT_WIDTH The horizontal size of the bounding box
        #    cv2.CC_STAT_HEIGHT The vertical size of the bounding box
        #    cv2.CC_STAT_AREA The total area (in pixels) of the connected component
        sizes = stats[:, -1]

        for i in range(1, num_nums):
            if sizes[i] > MIN_SIZE:
                nx, ny = centroids[i]
                s = stats[i]
                #top_left = (stats[i,1], stats[i,0])
                #bot_right = (stats[i,1]+stats[i,3], stats[i,0]+stats[i,2])

                val_and_centroids.append(
                    [n, ny, nx, s[1], s[0], s[1]+s[3], s[0]+s[2]])

                j += 1
    nums = []
    pos = []
    if j != 0:
        num_stats = np.array(
            sorted(val_and_centroids[:j], key=lambda x: x[2])).astype(np.uint16)

        first = 1

        next_val, next_y, next_x, next_bot, next_top = \
            None, None, None, None, None

        while(len(num_stats)):
            if first:
                _val, _y, _x, _top_y, _top_x, _bot_y, _bot_x = num_stats[0]
                _bot = [_bot_x*OUT_DOWNSCALING_X +
                        dx, _bot_y*OUT_DOWNSCALING_Y+dy]
                _top = [_top_x*OUT_DOWNSCALING_X -
                        dx, _top_y*OUT_DOWNSCALING_Y-dy]
                num_stats = np.delete(num_stats, 0, axis=0)
            else:
                _val, _y, _x, _top, _bot = next_val, next_y, next_x, next_top, next_bot

            if first:
                nums.append(str(_val))
                pos.append([_top, _bot])

            if len(num_stats) != 0:
                # vector of all remaining numbers x coordinates
                all_x = num_stats[:, 2].astype(np.int64)
                # vector of all remaining numbers y coordinates
                all_y = num_stats[:, 1].astype(np.int64)

                distances_sq = (((all_x - _x)*X_OFF_WEIGHT)**2 +
                                ((all_y - _y)*Y_OFF_WEIGHT)**2).astype(np.float)

                min_idx = distances_sq.argmin()

                next_val, next_y, next_x, next_top_y, next_top_x, next_bot_y, next_bot_x = num_stats[
                    min_idx]
                next_top = [next_top_x*OUT_DOWNSCALING_X -
                            dx, next_top_y*OUT_DOWNSCALING_Y-dy]
                next_bot = [next_bot_x*OUT_DOWNSCALING_X +
                            dx, next_bot_y*OUT_DOWNSCALING_Y+dy]

                if distances_sq[min_idx] < MAX_SQ_DISTANCE and _x < next_x:
                    nums[-1] += str(next_val)
                    num_stats = np.delete(num_stats, min_idx, axis=0)
                    first = 0
                    pos[-1][-1] = next_bot
                else:
                    first = 1
        
        # if len(img.shape) == 2:
        #     img = cv2.merge([img, img, img])

        for n, (a, b) in zip(nums, pos):
            img = cv2.rectangle(img, tuple(a), tuple(
                b), color=(1, 0.2, 0), thickness=1)
            cv2.putText(
                img, str(n), (a[0], a[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (1, 0.2, 0.1), 1)

        #image = cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        #cv2.putText(image, 'Fedex', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        if show:
            plt.figure()
            plt.imshow(img)
            plt.axis(False)
            if title is not None:
                plt.title(title)

            plt.show()
    # print(nums)
    # print(pos)
    # print(img)
    return nums, pos, img


## uncomment to see training samples


##Test samples
#img_tags = load('composite_img_tags.pkl')[:10]

#(img_set, img_tags) = load('composite_inline_dataset.pkl')

# for im, tags in zip(img_set[:10], img_tags[:10]):
#    read_numbers(im, tags)

# plt.show()
