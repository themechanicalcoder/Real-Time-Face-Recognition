# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:05:21 2019

@author: Gourav
"""

import numpy as np
import keras

def get_embedding(model,face_pixels):
    face_pixels=face_pixels.astype('float32')
    mean,std=face_pixels.mean(),face_pixels.std()
    face_pixels=(face_pixels-mean)/std
    samples=np.expand_dims(face_pixels,axis=0)
    yhat=model.predict(samples)
    return yhat[0]

data=np.load('friends.npz')
train_x,train_y,test_x,test_y=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

model=keras.models.load_model('facenet_keras.h5')

new_train_x=list()

for face in train_x :
    embedding =get_embedding(model,face)
    new_train_x.append(embedding)
new_train_x=np.asarray(new_train_x)
print(new_train_x.shape)


new_test_x=list()
for face in test_x :
    embedding =get_embedding(model,face)
    new_test_x.append(embedding)
new_test_x=np.asarray(new_test_x)

np.savez_compressed('friends-faces-embeddings.npz',new_train_x,train_y,
new_test_x,test_y)
print("here")
