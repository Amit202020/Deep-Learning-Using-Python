# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:13:28 2021

@author: A687640
"""

#load dataset
import pandas as pd
df= pd.read_csv('housepricedata.csv')
#print(df)

#seperate independent and dependent variable
dsarray = df.values
#print(dsarray)
X = dsarray[:,0:10]
print(X)
Y = dsarray[:,10]
#print(Y)

#scale the values between 0 and 1
from sklearn import preprocessing
dd =  preprocessing.MaxAbsScaler()
X_scale = dd.fit_transform(X)
#print(X_scale)

#split the dataset
from sklearn.model_selection import train_test_split
X_train , Y_train , X_test , Y_test =  train_test_split(X_scale,Y,test_size=0.3)
print(Y_test)



#Train and Build the ANN
from keras.models import Sequential
from keras.layers import Dense


#initialize the model
model  = Sequential(
    [
        Dense(32, activation="relu", input_shape=(10,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
    )

#compile the model
model.compile(optimizer="sgd",loss="binary_crossentropy",metrics=['accuracy'])


#fit the model
fitd = model.fit(X_train,Y_train,batch_size=10,epochs=100) 


#evaluate the model
print(model.evaluate(X_test,Y_test)[1])
























































