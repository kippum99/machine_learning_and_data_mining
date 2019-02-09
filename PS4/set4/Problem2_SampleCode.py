###############################################################################
# CS 155 HW 4 Problem 2 Sample Code
# Suraj Nair

# This sample code is meant as a guide on how to use keras
# and how to use the relevant model layers. This not a guide on
# how to design a network and the network in this example is 
# intentionally designed to have poor performace.
###############################################################################

import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout

## Importing the MNIST dataset using Keras
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## Visualizing an image (optional)
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.show()

## In your homework you should transform each input data point
## into a single vector here and should transform the 
## labels into a one hot vector using np_utils.to_categorical

## Also if you choose to do any data normalization (recommended)
## you should do it here

## Create your own model here given the constraints in the problem
model = Sequential()
model.add(Flatten(input_shape=(28,28,)))  # Use np.reshape instead of this in hw
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Activation('relu'))
## Once you one-hot encode the data labels, the line below should be predicting probabilities of each of the 10 classes
## e.g. it should read: model.add(Dense(10)), not model.add(Dense(1))
model.add(Dense(1))
model.add(Activation('softmax'))

## Printing a summary of the layers and weights in your model
model.summary()

## In the line below we have specified the loss function as 'mse' (Mean Squared Error) because in the above code we did not one-hot encode the labels.
## In your implementation, since you are one-hot encoding the labels, you should use 'categorical_crossentropy' as your loss.
## You will likely have the best results with RMS prop or Adam as your optimizer.  In the line below we use Adadelta
model.compile(loss='mse',optimizer='adadelta', metrics=['accuracy'])

fit = model.fit(X_train, y_train, batch_size=128, nb_epoch=10,
    verbose=1)

## Printing the accuracy of our model, according to the loss function specified in model.compile above
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
