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
from sklearn.externals import joblib
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
model=SVC(kernel='linear',probability=True)
model.fit(train_x,train_y)

filename='face_Recogniser.pkl'
joblib.dump(model, filename, compress=9)
encoder_file='face_encoder'
joblib.dump(model, encoder_file, compress=9)

#testing model on a random example of the dataset
selection=choice([i for i in range(test_x.shape[0])])
random_face_pixels=test_x_faces[selection]
random_face_emb=test_x[selection]
random_face_class=test_y[selection]
random_face_name=out_encoder.inverse_transform([random_face_class])

#prediciton for the face 
samples=expand_dims(random_face_emb,axis=0)
yhat_class=model.predict(samples)
yhat_prob=model.predict_proba(samples)

class_index=yhat_class[0]
class_probability=yhat_prob[0,class_index]*100
predict_names=out_encoder.inverse_transform(yhat_class)
print("Predicted {}-{}".format(predict_names[0],class_probability))
print("Expected {}".format(random_face_name[0]))
pyplot.imshow(random_face_pixels)
title="{}-({})".format(predict_names[0],class_probability)
pyplot.title(title)
pyplot.show()



