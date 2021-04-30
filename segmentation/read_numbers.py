import numpy as np
import matplotlib.pyplot as plt
from compress_pickle import load
import cv2

X_OFF_WEIGHT = 1
Y_OFF_WEIGHT = 4
MAX_SQ_DISTANCE = 30**2

MAX_LINES = 7
MIN_SIZE = 10

img_tags = load('composite_img_tags.pkl')[:10]

#print(img_tags.shape)

#for i in range(img_tags.shape[0]):
#    plt.figure()
#    plt.imshow(img_tags[i], cmap='gray')

#plt.show()

def nearest_nonzero_idx(a, x,y):
    r,c = np.nonzero(a) #indeices of nonzero elemenyts in array a
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]


def read_numbers(img_tags):
    h,w = img_tags.shape
    _, img_bin = cv2.threshold(img_tags,0,255,cv2.THRESH_BINARY)
    num_nums, img_labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin)
    
    val_and_centroids = np.zeros((num_nums, 3), np.int16)

    sizes = stats[:,-1]
    print(num_nums, sizes)
    j = 0
    for i in range(1, num_nums):
        if  sizes[i] > MIN_SIZE: 
            ny, nx = centroids[i]
            y,x = nearest_nonzero_idx(img_tags, nx, ny)

            value = img_tags[y, x] - 1 # add -1 to get the real value
            val_and_centroids[j] = (value, ny, nx)
            j+=1
    
    num_stats = np.array(sorted(val_and_centroids[:j], key=lambda x: x[2]))

    first = 1
    nums = []

    while(len(num_stats)):
        if first:
            _val, _y, _x = num_stats[0]
            np.delete(num_stats, 0, axis=0)
        else:
            _val, _y, _x = next_val, next_y, next_x
        
        all_x = num_stats[:,2] #vector of all remaining numbers x coordinates
        all_y = num_stats[:,1] #vector of all remaining numbers y coordinates
 
        distances_sq = ((all_x - _x)*X_OFF_WEIGHT)**2 + ((all_y - _y)*Y_OFF_WEIGHT)**2

        min_idx = distances_sq.argmin()

        next_val, next_y, next_x = num_stats[min_idx]
        
        if first:
            nums.append(str(_val))

        if distances_sq[min_idx] < MAX_SQ_DISTANCE and _x < next_x:
            nums[-1] += str(next_val)
            np.delete(num_stats, min_idx, axis=0)
            first = 0
        else: 
            nums.append(str(next_val))
            first = 1

    if first: #take into account last number if it would have been a first num, also fixes one num imgs
        nums.append(str(num_stats[j-1][0]))

    plt.figure()
    plt.imshow(img_tags)
    plt.title(f'{nums}')
    return

    l = min(K * j**.5 + 1, MAX_LINES) #assume numbers form a square for simplicity, at least one line
    num_stats = sorted(val_and_centroids[:j], key=lambda x: x[2]//(h/l)*h+x[1])

    first = 1

    print(j)

    for i in range(j-1):
        _val, _x, _y = num_stats[i]
        next_val, next_x, next_y = num_stats[i+1]

        if first:
            nums.append(str(_val))

        if (((_x - next_x)*X_OFF_WEIGHT)**2 + ((_y - next_y)*Y_OFF_WEIGHT)**2) < MAX_SQRD_DISTANCE and next_x > _x:
            first = 0
            nums[-1] += str(next_val)
        else:
            first = 1

    if first: #take into account last number if it would have been a first num, also fixes one num imgs
        nums.append(str(num_stats[j-1][0]))


    plt.figure()
    plt.imshow(img_tags)
    plt.title(f'{nums}')

for im in img_tags:
    read_numbers(im)
    print('----------------')
     
plt.show()