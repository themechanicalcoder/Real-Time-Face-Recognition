from os import listdir
import os
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2 
import imutils

detector = MTCNN()
# extract a single face from a given photograph
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract_face(filename, required_size=(160, 160)):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    roi_color=image
    for (x,y,w,h) in faces:
        roi_color = image[y:y+h, x:x+w]
    image=cv2.resize(roi_color,(160,160))
    #print(image)
    return asarray(image)
 
# load images and extract faces for all images in a directory
def load_faces(directory):
	faces = list()
	# enumerate files
	for filename in listdir(directory):
		# path
		path = os.path.join(directory,filename)
		# get face
		face = extract_face(path)
		# store
		faces.append(face)
	return faces
 
# load a dataset that contains one subdir for each class that in turn contains image
extract_face("D:\\Dataset\\data\\train\\Aqil\\frame0.jpg").shape

def load_dataset(directory):
	X, y = list(), list()
	# enumerate folders, on per class
	for subdir in listdir(directory):
		# path
		path = os.path.join(directory,subdir)
		# skip any files that might be in the dir
		#if not isdir(path):
		#	continue
		# load all faces in the subdirectory
		faces = load_faces(path)
		# create labels
		labels = [subdir for _ in range(len(faces))]
		# summarize progress
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		# store
		X.extend(faces)
		y.extend(labels)
	return asarray(X),asarray(y)
 
# load train dataset
if __name__ == "__main__": 
    
    trainX, trainy = load_dataset('D:\\Dataset\\data\\train')
    print(trainX.shape, trainy.shape)
    
    # load test dataset
    testX, testy = load_dataset('D:\\Dataset\\data\\val')
    print(testX.shape, testy.shape)
    
    # save arrays to one file in compressed format
    savez_compressed('friends.npz', trainX, trainy, testX, testy)