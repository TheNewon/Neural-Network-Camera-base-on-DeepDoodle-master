import pygame
import random
import os
import numpy as np
import cv2
from dutil import *
import scipy.misc
import string
import easygui
#User constants
device = "gpu0"
model_fname = 'Model.h5'
background_color = (0, 0, 0)
input_w = 144
input_h = 192
image_scale = 3
image_padding = 10
mouse_interps = 10
val = 255
#Derived constants
drawing_w = input_w * image_scale
drawing_h = input_h * image_scale
window_width = drawing_w*2 + image_padding*3
window_height = drawing_h + image_padding*2
doodle_x = image_padding
doodle_y = image_padding
generated_x = doodle_x + drawing_w + image_padding
generated_y = image_padding
import os, random, sys
import numpy as np
import cv2
import time

DOTS_PER_IMG = 256
IMAGE_W = 144
IMAGE_H = 192
IMAGE_DIR = 'PICTURES'
NUM_CHANNELS = 1
a=0
video = cv2.VideoCapture(0)
x_cam = np.empty((1, NUM_CHANNELS, IMAGE_H, IMAGE_W), dtype=np.uint8)


def cam_drawing():
        global cur_drawing
        check, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = yb_resize(frame)
        x_cam[0] = rand_dots(frame)
        key = cv2.waitKey(1)
        cur_drawing = x_cam[0] 
        global needs_update
        global needs_update2
        needs_update = True
        needs_update2 = True
def yb_resize(img):
	return cv2.resize(img, (IMAGE_W, IMAGE_H), interpolation = cv2.INTER_LINEAR)
	
def rand_dots(img):
	sample_ratio = 1
	return auto_canny(img, sample_ratio)


#Global variables
needs_update = True
needs_update2 = True
cur_color_ix = 1
cur_drawing = None

cur_gen = np.zeros((3, input_h, input_w), dtype=np.uint8)
rgb_array = np.zeros((input_h, input_w, 3), dtype=np.uint8)


#Keras
print("Loading Keras...")
import os
os.environ['THEANORC'] = "./" + device + ".theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print("Theano Version: " + theano.__version__)
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K
K.set_image_data_format('channels_first')

#Load the model
print("Loading Model...")
model = load_model(model_fname)

#Open a window
pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
doodle_surface_mini = pygame.Surface((input_w, input_h))
doodle_surface = screen.subsurface((doodle_x, doodle_y, drawing_w, drawing_h))
gen_surface_mini = pygame.Surface((input_w, input_h))
gen_surface = screen.subsurface((generated_x, generated_y, drawing_w, drawing_h))
pygame.display.set_caption('Deep Doodle - edit By <Guźiec>')


                        
def sparse_to_rgb(sparse_arr):
        t = np.repeat(sparse_arr, 3, axis=0)
        return np.transpose(t, (2, 1, 0))

def draw_doodle():
        pygame.surfarray.blit_array(doodle_surface_mini, rgb_array)
        pygame.transform.scale(doodle_surface_mini, (drawing_w, drawing_h), doodle_surface)
        pygame.draw.rect(screen, (255,255,255), (doodle_x, doodle_y, drawing_w, drawing_h), 1)

def draw_generated():
        pygame.surfarray.blit_array(gen_surface_mini, np.transpose(cur_gen, (2, 1, 0)))
        pygame.transform.scale(gen_surface_mini, (drawing_w, drawing_h), gen_surface)
        pygame.draw.rect(screen, (255,255,255), (generated_x, generated_y, drawing_w, drawing_h), 1)
        
#Main loop
running = True
while running:
        #Process events
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        running = False
                        break
                

        cam_drawing()
 
                        

        #Check if we need an update
        if needs_update:
                fdrawing = np.expand_dims(cur_drawing.astype(np.float32) / 255.0, axis=0)
                pred = model.predict(add_pos(fdrawing), batch_size=1)[0]
                cur_gen = (pred * 255.0).astype(np.uint8)
                #rgb_array = sparse_to_rgb(cur_drawing)
                randomStr=''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                needs_update = False
        if needs_update2:
                rgb_array = sparse_to_rgb(cur_drawing)
                needs_update2 = False
                
        
        
        #Draw to the screen
        screen.fill(background_color)
        draw_doodle()
        draw_generated()
        
        #Flip the screen buffer
        pygame.display.flip()
        pygame.time.wait(10)
