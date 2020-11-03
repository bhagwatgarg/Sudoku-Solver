import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

def get_data():
  imgs=[[(f'./EnglishFnt/English/Fnt/Sample0{format(i, "02d")}/'+j) for j in os.listdir(f'./EnglishFnt/English/Fnt/Sample0{format(i, "02d")}/')] for i in range(2, 11)]
  x=[]
  y=[]
  z=np.zeros((10))

  for i in range(len(imgs[0])):
    x.append(np.zeros((64, 64, 1)))
    z1=np.zeros_like(z)
    z1[0]=1
    y.append(z1)

  for i in range(9):
    for j in (imgs[i]):
      a=cv2.imread(j)
      a=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
      a=cv2.bitwise_not(a)
      a=cv2.resize(a, (64, 64))
      # a=cv2.GaussianBlur(a, (3,3), 3)
      # a=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
      # a=cv2.Canny(a, 200, 200)
      x.append(a.reshape((64, 64, 1)))
      z1=np.zeros_like(z)
      z1[i+1]=1
      y.append(z1)

  x=np.array(x)//255
  y=np.array(y)

  x_train, x_test, y_train, y_test=train_test_split(x, y,  test_size=0.3)
  # x_test, x_val, y_test, y_val=train_test_split(x_test, y_test, test_size=0.5)
  return x_train, x_test, y_train, y_test


def get_model():

  model=tf.keras.models.Sequential()

  model.add(tf.keras.layers.Conv2D(48, (2,2), (2,2), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.MaxPool2D((2,2)))

  model.add(tf.keras.layers.Conv2D(64, (2,2), (2,2), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.MaxPool2D((2,2)))

  # model.add(tf.keras.layers.Conv2D(64, (2,2), (2,2)))
  # model.add(tf.keras.layers.BatchNormalization())
  # model.add(tf.keras.layers.Activation('relu'))
  # model.add(tf.keras.layers.MaxPool2D((2,2)))

  model.add(tf.keras.layers.Conv2D(64, (2,2), (2,2)))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.MaxPool2D((2,2)))

  model.add(tf.keras.layers.Flatten())
  # model.add(tf.keras.layers.Dense(32))
  model.add(tf.keras.layers.Dense(10))
  model.add(tf.keras.layers.Softmax())
  return model

def train_model():
  print('Getting Data...')
  x_train, x_test, y_train, y_test=get_data()
  print('Training Model...')
  model=get_model()
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.1), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
  history=model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=10)

  model.save('./Model/model.h5')

if __name__=='__main__':
  train_model()