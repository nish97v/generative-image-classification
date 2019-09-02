import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import sys

def loader(path, num_of_images, face):
	count = 0
	image_set = []
	I = num_of_images
	file = glob.glob('{}/annotations/FDDB-fold-01-ellipseList.txt'.format(path))
	f = open(file[0], "r")
	lines = f.read().split('\n')
	for i in range(len(lines)):
		breaker = True
		if(lines[i].startswith('200')):
			if(count == I):
				image_set = np.array(image_set)
				image = np.reshape(image_set, (num_of_images, 100))
				return
			img_path = path + '/originalPics/' + str(lines[i]) + '.jpg'
			img = cv2.imread(img_path,0)
			if(face == False):
				x = y = 30
				dx = dy = 30
			else:	
				for l in range(int(lines[i+1])):
					a = lines[i+2+(8*l)].split(' ')
					if(len(a) == 7):
						dx = int(float(a[0])); dy = int(float(a[1])); y = int(float(a[3])); x = int(float(a[4]))
					else:
						breaker = False
						continue	
			if(breaker == False):
				continue		
			img = img[x-dx:x+dx, y-dy:y+dy]
			(a, b) = img.shape
			if(not((a == 0) or (b == 0))):
				count += 1
				img = cv2.resize(img, (60, 60))
				img = cv2.resize(img, (10, 10))
				if(face == False):
					cv2.imwrite('{}/nonfaces/'.format(path) + str(count) + '.jpg',img )
				else:
					cv2.imwrite('{}/faces/'.format(path) + str(count) + '.jpg',img )
				image_set.append(img)

def load_wrapper(path, face, train):
	print('Face : ', face)
	print('Train : ', train)
	if(face == True):
		print('Face')
		folder = glob.glob('{}/faces/*.jpg'.format(path))
	else:
		print('Non face')
		folder = glob.glob('{}/nonfaces/*.jpg'.format(path))
	image = []
	for file in folder:
		img =  cv2.imread(file,0)
		image.append(img)
	image = np.array(image)
	image = np.reshape(image, (len(image), 100))
	if(train == True):
		return image[:1000]
	else:
		return image[1000:1100]