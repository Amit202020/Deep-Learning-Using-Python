# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:05:09 2021

@author: A687640
"""

#load the dataset from 
from keras.datasets import mnist
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

#print(X_train.ndim)
#print(X_train.shape)
#print(X_train.dtype)

#print(X_test.ndim)
#print(X_test.shape)
#print(X_test.dtype)


import matplotlib.pyplot as plt
plt.imshow(X_train[0])

#print(X_train[0].shape)

# reshape our dataset
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#output catagory is represented using binary variable
from keras.utils import to_categorical
Y_train = to_categorical(Y_train)
Y_test  = to_categorical(Y_test) 


#Build the model
from  keras.models import Sequential
from  keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


model = Sequential()
#add layer like convolution
model.add(Conv2D(64 , kernel_size=3 , activation='relu', input_shape=(28,28,1)))

#add max pulling layer
model.add(MaxPooling2D(pool_size = (2, 2)))

#add flatten layer
model.add(Flatten())

#full connection
model.add(Dense(10,activation='softmax')
          
          
#compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
          

#train the model
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=3)
          
          
          
          
          
          
          
          
          
          
          
          





























