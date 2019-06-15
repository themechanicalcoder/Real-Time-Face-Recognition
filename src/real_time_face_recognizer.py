# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:38:10 2019

@author: Gourav
"""
from random import choice 
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
import numpy as np


#load data
data=load('friends.npz')
test_x_faces=data['arr_2']

data=load('friends-faces-embeddings.npz')
train_x,train_y,test_x,test_y=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

in_encoder=Normalizer(norm='l2')
train_x=in_encoder.transform(train_x)
test_x=in_encoder.transform(test_x)


#transforming the labels into one-hot-arrays
out_encoder=LabelEncoder()
out_encoder.fit(train_y)
train_y=out_encoder.transform(train_y)
test_y=out_encoder.transform(test_y)

# fit model
svm_model=SVC(kernel='linear',probability=True)
svm_model.fit(train_x,train_y)




import keras
import cv2
import numpy as np
from face_preprocessing import extract_face
#from embeddings import get_embedding
from numpy import asarray,expand_dims
import urllib

def get_embedding(face_pixels):
    face_pixels=face_pixels.astype('float32')
    mean,std=face_pixels.mean(),face_pixels.std()
    face_pixels=(face_pixels-mean)/std
    samples=np.expand_dims(face_pixels,axis=0)
    yhat=facenet_model.predict(samples)
    return yhat[0]

#loading models
facenet_model=keras.models.load_model('facenet_keras.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    
    #extracting the face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        cropped = gray[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]
        
        cv2.imshow('img',roi_color)
        roi_color=cv2.resize(roi_color,(160,160))
        cv2.imshow('new',roi_color)
        roi_color=asarray(roi_color)
        
        #getting the embeddings
        embedding=get_embedding(roi_color)
        samples=expand_dims(embedding,axis=0)
        
        #getting class and proobability and classl using svm_model
        yhat_class=svm_model.predict(samples)
        yhat_prob=svm_model.predict_proba(samples)
        class_index=yhat_class[0]
        class_probability=yhat_prob[0,class_index]*100
        predict_names=out_encoder.inverse_transform(yhat_class)
        
        #writing on the image
        font = cv2.FONT_HERSHEY_DUPLEX
        s="{} Acc={}".format(predict_names[0],class_probability)
        if(class_probability<70):
            s="Unknown"
        cv2.putText(img,s,(x-w,y),font,.7,(0,0,255),1)
        cv2.imshow("Predicted image",img)
        

    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()

