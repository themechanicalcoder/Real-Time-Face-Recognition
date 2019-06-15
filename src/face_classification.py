# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:35:00 2019

@author: Gourav
"""
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

data=load('friends-faces-embeddings.npz')
train_x,train_y,test_x,test_y=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']

#Normalizing the embeddings 
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

# predict
yhat_train=model.predict(train_x)
yhat_test=model.predict(test_x)

#score
score_train=accuracy_score(train_y,yhat_train)
score_test=accuracy_score(test_y,yhat_test)

print("Training Accuracy={} Test Accuracy={}".format(score_train,score_test))