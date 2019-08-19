import os, random, sys
import numpy as np
import cv2
import time

DOTS_PER_IMG = 256
IMAGE_W = 144
IMAGE_H = 192
NUM_CHANNELS = 1
a=0
video = cv2.VideoCapture(0)
x_data = np.empty((1, NUM_CHANNELS, IMAGE_H, IMAGE_W), dtype=np.uint8)
def auto_canny(image, sigma=0.0):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	grayed = np.where(gray < 20, 255, 0)

	lower = sigma*128 + 128
	upper = 255
	edged = cv2.Canny(image, lower, upper)

	return np.maximum(edged, grayed)

def yb_resize(img):
	return cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation = cv2.INTER_LINEAR)
	
def rand_dots(img):
	sample_ratio = 1
	return auto_canny(img, sample_ratio)
while True:
    a=a+1
    check, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = yb_resize(frame)
    x_data[0] = rand_dots(frame)
    cv2.imshow('lines',x_data[0][0])
    key = cv2.waitKey(1)
