import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

def get_data():
  imgs=[[(f'./data/English/Fnt/Sample0{format(i, "02d")}/'+j) for j in os.listdir(f'./data/English/Fnt/Sample0{format(i, "02d")}/')] for i in range(2, 11)]
  x=[]
  y=[]
  z=np.zeros((10))

  for i in range(len(imgs[0])):
    a=(np.random.random((64, 64, 1))>0.95)
    mean_px = a.mean().astype(np.float32)
    # a = (a - mean_px)
    x.append(a)
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


      a=a//255
      # a=(a+(np.random.random(a.shape)>0.95))%2
      # mean_px = a.mean().astype(np.float32)
      # a = (a - mean_px)


      x.append(a.reshape((64, 64, 1)))
      z1=np.zeros_like(z)
      z1[i+1]=1
      y.append(z1)

  x=np.array(x)
  y=np.array(y)

  

  x_train, x_test, y_train, y_test=train_test_split(x, y,  test_size=0.3)
  for i in range(len(x_test)):
    x_test[i]=(x_test[i]-x_test[i].mean())/x_test[i].std()
  # x_train=(x_train+(np.random.random(x_train.shape)>0.95))%2
  # x_test, x_val, y_test, y_val=train_test_split(x_test, y_test, test_size=0.5)
  return x_train, x_test, y_train, y_test


def get_model():

  model=tf.keras.models.Sequential()

  model.add(tf.keras.layers.Conv2D(48, (2,2), (2,2), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
  # model.add(tf.keras.layers.MaxPool2D((2,2)))

  model.add(tf.keras.layers.Conv2D(64, (2,2), (2,2), padding='same'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.MaxPool2D((2,2)))

  model.add(tf.keras.layers.Conv2D(64, (2,2), (2,2)))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
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

def train_model(data_func):
  print('Getting Data...')
  x_train, x_test, y_train, y_test=data_func()

  gen=tf.keras.preprocessing.image.ImageDataGenerator(
      samplewise_center=True,
      samplewise_std_normalization=True,
      rotation_range=5,
      width_shift_range=0.2,
      height_shift_range=0.05,
  #     brightness_range=None,
  #     shear_range=0.1,
      zoom_range=[0.9, 1.1],
  #     channel_shift_range=0.0,
      fill_mode='nearest',
  #     cval=0.0,
      horizontal_flip=False,
      vertical_flip=False,
      rescale=None,
      preprocessing_function=None,
      data_format=None,
      # validation_split=0.2,
      dtype=None,
  )

  print('Training Model...')
  model=get_model()
  model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.1), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
  history=model.fit(gen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, validation_data=(x_test, y_test), epochs=15)

  model.save('./Model/model')

if __name__=='__main__':
  train_model(get_data)