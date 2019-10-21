# Neural-Network-Camera-base-on-DeepDoodle-master
Edited and improved version of DeepDoodle-master but for camera usage
Very hard to set repository only for pro user
u will need theano + pygpu installed
CUDA required
module required: pygame, numpy, easygui, cv2, keras

Best metod to get working -> install old conda with python 3.5 in bash. install conda 3.6 in virtual env

install all modules without theano on conda 3.5 end install theano on conda 3.6 next copy theano and random module from 3.6 to 3.5

for what i need do this? python 3.5 have problem with theano and pygpu on windows so we need theano 1.0.4 from python 3.6 to copy to python 3.5 to working.
why need replace random module?
bc randomStr=''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) not working on old random module from python 3.5 so wee need replace it with 3.6 module

i did it on many pc and theano from original 3.5 bever gets work.

how to run:

u can run datagen_camera.py

then train.py

when trained doodler_cam.py or doodler_draw.py


OR


u place images to PICTURES folder

next datagen_picture.py

then train.py

when trained doodler_cam.py or doodler_draw.py

