import csv
import matplotlib.image as mpimg
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
# Initial Setup for Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.layers import Cropping2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback
from keras.regularizers import l2
import time

dataset_name=''

lines = []
with open('./'+dataset_name+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
angles = []
k = 0

for line in lines:
    if (k==0):
        print (line)
    else:
        angle = float(line[3])
        angles.append(angle)
    k+=1
    
angles = np.array(angles)   

image_paths = []
angles = []
k = 0

for line in lines:
    if (k > 0):
        angle = float(line[3])
        speed = float(line[6])
        if (speed > 0):
            if ((random.random() > 0.9)and(angle==0))or(angle>0):
                for i in range(3):
                    # Load images from center, left and right cameras
                    source_path = line[i]
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    local_path = './'+dataset_name + '/' + source_path 
                    image_paths.append(local_path)
                correction = 0.2*(1 + random.random()/2.0)

                # Steering adjustment for center images
                angles.append(angle)

                # Add correction for steering for left images
                angles.append(angle+correction)

                # Minus correction for steering for right images
                angles.append(angle-correction)
    k+=1

image_paths = np.array(image_paths)
angles = np.array(angles)   
# split into train/test sets
image_paths_train, image_paths_val, angles_train, angles_val = train_test_split(image_paths, angles,
                                                                                  test_size=0.1, random_state=17)

def preprocess_image(img):
    new_img = img[60:135,:,:]
    # scale to 64x64x3 
    new_img = cv2.resize(new_img,(64, 64), interpolation = cv2.INTER_AREA)
    return new_img

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
	
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def flip_image(img, steering):
    if random.randint(0, 1):
        return cv2.flip(img, 1), -steering
    else:
        return img, steering	

def trans_image(image,steer,x_range, y_range):
    # Translation
    tr_x = np.random.randint(-x_range, x_range+1)
    steer_ang = steer + tr_x*.004
    tr_y = np.random.randint(-y_range, y_range+1)
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(320,160))
    return image_tr,steer_ang
		
def generate_training_data(image_paths, angles, batch_size=128, validation_flag=False):
	image_paths, angles = shuffle(image_paths, angles)
	X,y = ([],[])
	while True:       
		for i in range(len(angles)):
			img = mpimg.imread(image_paths[i])
			angle = angles[i]
			img = augment_brightness_camera_images(img)
			img, angle = trans_image(img, angle, 50,20)
			img = add_random_shadow(img)
			img, angle = flip_image(img, angle)
			img = preprocess_image(img)
			X.append(img)
			y.append(angle)
			if len(X) == batch_size:
				yield (np.array(X), np.array(y))
				X, y = ([],[])
				
BATCH_SIZE = 128
ch, row, col = 3, 64, 64

class debugCallback(Callback):
    def on_epoch_end(self, batch, logs={}):
        samples = image_paths_val[:5]
        angles = angles_val[:5]
        print()
        print(logs)
        x = [angle for angle in angles]
        print("Should be: "+', '.join(['%.3f']*len(x)) % tuple(x))
        angles = []
        for filename in samples:
            img = mpimg.imread(filename)
            img = preprocess_image(img)
            angle = model.predict(img[None, :, :, :], batch_size=1)
            angles.append(angle)
        #Print predicted driving angle for first example
        print("Predicted: "+', '.join(['%.3f']*len(x)) % tuple(angles))

def commaai_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, (8, 8), subsample=(4, 4), padding='same', activation="relu"))
    model.add(Convolution2D(32, (5, 5), subsample=(2, 2), padding='same', activation="relu"))
    model.add(Convolution2D(64, (5, 5), subsample=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(.5))
    model.add(Dense(1))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mean_squared_error')

    return model
    
    
model = commaai_model()
# initialize generators
train_gen = generate_training_data(image_paths_train, angles_train,
                                   validation_flag=False, batch_size=BATCH_SIZE)
val_gen = generate_training_data(image_paths_val, angles_val,
                                 validation_flag=True, batch_size=BATCH_SIZE)
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
debug = debugCallback()
train_start_time = time.time()
history_object = model.fit_generator(callbacks=[checkpoint, debug], generator=train_gen, 
                                     validation_data=val_gen, epochs=15,
                                     steps_per_epoch=8000/BATCH_SIZE, 
                                     verbose=1, 
                                     validation_steps=128/BATCH_SIZE)
total_time = time.time() - train_start_time
print('Total training time: %.2f sec (%.2f min)' % (total_time, total_time/60))

print(model.summary())
model.save('./model.h5')