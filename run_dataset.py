import moviepy.editor as mpy
import matplotlib.image as mpimg
import csv
import numpy as np
import cv2
from skimage import draw
from keras.models import load_model

model = load_model('model09.h5')
lines = []
image_paths = []
with open('./data/driving_log.csv') as csvfile:
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
		source_path = line[0]
		local_path = "./data/" + source_path
		image_paths.append(local_path)
	k+=1
    
angles = np.array(angles) 
image_paths = np.array(image_paths)

def draw_angle_line(image, angle, color):
    img = np.copy(image)
    s = img.shape
    line_len = s[0]//2
    angle = angle / 360 * np.pi * 100 # Times 100 just to make it more visible
    line_y, line_x = int(line_len * np.cos(angle)), int(line_len * np.sin(angle))
    rr,cc = draw.line(s[0]-1, s[1]//2, s[0]-1-line_y, s[1]//2 + line_x)
    img[rr,cc,:] = color
    return img, angle

def preprocess_image(img):
	new_img = img[70:134,:,:]
	new_img = cv2.resize(new_img,(64, 64), interpolation = cv2.INTER_AREA)
	return new_img

def make_frame(t):
	num = int(t*12)
	img = mpimg.imread(image_paths[num])
	new_img = preprocess_image(img)
	angle = model.predict(new_img[None,:,:,:])
	#draw pred angle
	blue = (25,160,235)
	img, angle_pred = draw_angle_line(img, angle, blue)
	angle = angles[num]
	#draw dataset angle
	green = (25,235,75)
	img, angle = draw_angle_line(img, angle, green)
	return img

clip = mpy.VideoClip(make_frame, duration=200) # 100 seconds
clip.write_videofile("video_predict.mp4",fps=12, codec='mpeg4', bitrate="50000k")
#clip.write_gif("data.gif",fps=60)