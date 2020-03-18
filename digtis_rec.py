import tensorflow as tf 
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D

mnist = tf.keras.datasets.mnist
# load train and test dataset
def load_dataset():
    (x_train,y_train),(x_test,y_test)= mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print(x_train.shape)


    return x_train, y_train, x_test, y_test


def define_model():
    x_train, y_train, x_test, y_test = load_dataset()
    model = Sequential()
 #model.add(Conv2D(28, kernal_size=(3,3), input_shape=(28, 28, 1)))
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(200,activation = tf.nn.relu))
    model.add(Dropout(0.3))
    model.add(Dense(10,activation = tf.nn.softmax))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(x= x_train , y = y_train , epochs= 1)
    return model

define_model()

def eval():
    x_train, y_train, x_test, y_test = load_dataset()
    model = define_model()
    model.evaluate(x_test, y_test)
    return model

eval()

