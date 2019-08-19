import os, random, sys
import numpy as np
import cv2
from dutil import *

NUM_IMAGES = int(input('frame number?'))
SAMPLES_PER_IMG = 1
DOTS_PER_IMG = 256
IMAGE_W = 144
IMAGE_H = 192
IMAGE_DIR = 'PICTURES'
NUM_SAMPLES = NUM_IMAGES * 2 * SAMPLES_PER_IMG
NUM_CHANNELS = 1
video = cv2.VideoCapture(0)
def yb_resize(img):
    return cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation = cv2.INTER_LINEAR)
    
def rand_dots(img, sample_ix):
    sample_ratio = float(sample_ix) / SAMPLES_PER_IMG
    return auto_canny(img, sample_ratio)

x_data = np.empty((NUM_SAMPLES, NUM_CHANNELS, IMAGE_H, IMAGE_W), dtype=np.uint8)
y_data = np.empty((NUM_SAMPLES, 3, IMAGE_H, IMAGE_W), dtype=np.uint8)
ix = 0
while ix < NUM_SAMPLES:
    check, frame = video.read()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = yb_resize(frame)
    cv2.imwrite(IMAGE_DIR+'/rgb' + str(ix) + '.png', img)
    for i in range(SAMPLES_PER_IMG):
        y_data[ix] = np.transpose(img, (2, 0, 1))
        x_data[ix] = rand_dots(img, i)
        outimg = x_data[ix][0]
        cv2.imwrite('cargb' + str(ix) + '.png', outimg)
        ix += 1
        y_data[ix] = np.flip(y_data[ix - 1], axis=2)
        x_data[ix] = np.flip(x_data[ix - 1], axis=2)
        ix += 1
    key = cv2.waitKey(1)
    sys.stdout.write('\r')
    progress = ix * 100 / NUM_SAMPLES
    sys.stdout.write(str(progress) + "%")
    sys.stdout.flush()
    assert(ix <= NUM_SAMPLES)

assert(ix == NUM_SAMPLES)
print ("Saving...")
np.save('x_data.npy', x_data)
np.save('y_data.npy', y_data)
