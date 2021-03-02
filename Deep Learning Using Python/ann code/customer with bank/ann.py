# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('cust.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#print(X)


# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer


labelencoder=LabelEncoder()
X[:,1]=labelencoder.fit_transform(X[:,1])
preprocess = ColumnTransformer(transformers=[('cat', OneHotEncoder(),[1,2])],remainder="passthrough")
X = np.array(preprocess.fit_transform(X))

#print(X[0])



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)





# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)



# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# Initialising the ANN
classifier = Sequential()


# Adding the input layer and the first hidden layer
classifier.add(Dense(7,  activation = 'relu', input_dim = 13))



# Adding the second hidden layer
classifier.add(Dense(7, activation = 'relu'))


# Adding the output layer
classifier.add(Dense( 1, activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)



# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)




# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



