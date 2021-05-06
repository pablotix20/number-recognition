import numpy as np
import matplotlib.pyplot as plt
from compress_pickle import load
import cv2

X_OFF_WEIGHT = 1
Y_OFF_WEIGHT = 3
MAX_SQ_DISTANCE = 25**2

MIN_SIZE = 10

OPENING = False
OPENING_KERNEL = np.array(
                [[0,1,0],
                 [1,1,1],
                 [0,1,0]],
                 np.uint8)

def nearest_nonzero_idx(a, x,y):
    r,c = np.nonzero(a) #indeices of nonzero elemenyts in array a
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]


def read_numbers(img_tags):
    h,w = img_tags.shape
    _, img_bin = cv2.threshold(img_tags,0,255,cv2.THRESH_BINARY)
    if OPENING: img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, OPENING_KERNEL)

    num_nums, img_labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity  = 4)
    
    val_and_centroids = np.zeros((num_nums, 3), np.int16)

    sizes = stats[:,-1]
    j = 0
    for i in range(1, num_nums):
        if  sizes[i] > MIN_SIZE: 
            nx, ny = centroids[i]
            y, x = nearest_nonzero_idx(img_tags, ny, nx)

            value = img_tags[y, x] - 1 # add -1 to get the real value 
            val_and_centroids[j] = (value, ny, nx)
            j+=1

    num_stats = np.array(sorted(val_and_centroids[:j], key=lambda x: x[2]))

    first = 1
    nums = []
    pos = []

    next_val, next_y, next_x = None, None, None

    while(len(num_stats)):
        if first:
            _val, _y, _x = num_stats[0]
            num_stats = np.delete(num_stats, 0, axis=0)
        else:
            _val, _y, _x = next_val, next_y, next_x
        
        if first:
            nums.append(str(_val))
            pos.append([_x, _y])

        if len(num_stats) != 0:
            all_x = num_stats[:,2].astype(np.int64) #vector of all remaining numbers x coordinates
            all_y = num_stats[:,1].astype(np.int64) #vector of all remaining numbers y coordinates
 
            distances_sq = (((all_x - _x)*X_OFF_WEIGHT)**2 + ((all_y - _y)*Y_OFF_WEIGHT)**2).astype(np.float)

            min_idx = distances_sq.argmin()

            next_val, next_y, next_x = num_stats[min_idx]
        
            if distances_sq[min_idx] < MAX_SQ_DISTANCE and _x < next_x:
                nums[-1] += str(next_val)
                num_stats = np.delete(num_stats, min_idx, axis=0)
                first = 0
            else: 
                first = 1

    plt.figure()
    plt.imshow(img_tags)
    plt.title(f'{nums}')
    return nums, pos
  
exit()
  
#Test samples
img_tags = load('composite_img_tags.pkl')[10:]
  
for im in img_tags:
    read_numbers(im)
     
plt.show()
