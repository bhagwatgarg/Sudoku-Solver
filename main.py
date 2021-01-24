import cv2
import numpy as np
import os
import sys
from contours import get_sudoku
from convertImage import persist_image
import tensorflow as tf
from model import get_data, train_model
from solve import Solver

def split_image(img, img_size, crop):
  """
    Splits the sudoku image into 9*9 parts with each part itself an image of a part of the input image
    Arguments:
      img: sudoku image
      img_size: the size of the image
      crop: number of pixels to be cropped off from each side of the sub-images
    Returns:
      An array with 9*9 sub-images of the input image
  """
  i, j, add = 0, 0, img_size//9
  new=np.zeros((9, 9, add-2*crop, add-2*crop))

  for i in range(9):
    for j in range(9):
      x=(img[add*j+crop:add*(j+1)-crop, add*i+crop:add*(i+1)-crop]+(np.random.random((64, 64))>0.95))/255
      # if(x.std().astype(np.float)<0):
      #   x=(np.random.random((64, 64))>0.95)
      mean=x.mean().astype(np.float32)
      std=x.std().astype(np.float)

      x = (x - mean)/std

      new[j][i]=x
  # new=(new+(np.random.random(new.shape)>0.95))%2
  return new

def getNums(img):
  """
    This function returns the predictions made by the model. If model isnt found, it is trained on the data first
    Arguments:
      img: an array containing 9*9 sub-images of the sudoku puzzle
    Returns:
      A 9*9 array containg the predictions made by the model
  """
  if len(os.listdir('./Model'))==0:
    train_model(get_data)
  model=tf.keras.models.load_model('./Model/model')
  # nums=np.zeros((img.shape[0], img.shape[1]))
  img=img.reshape((-1, img.shape[2], img.shape[3], 1))
  print(img.shape)
  nums=model.predict(img)
  nums=np.argmax(nums, axis=1).reshape((9, 9))
  return nums

def print_sudoku(nums):
  """
    Prints the sudoku in a nice form
    (Not required at all)
  """
  for i in range(9):
    s=''
    for j in range(9):
      s=s+str(nums[i][j])+' '
    print(s)

def main(image='sample.jpeg'):
  """
    Main function which calls all the other functions for carrying out certain tasks
    Arguments:
      image: Path to the sudoku image
  """
  img_size=80*9
  # add=img_size//9
  crop=8
  imgi=get_sudoku(image, img_size)

  img=cv2.bilateralFilter(imgi, 5, 20, 20)
  img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # img=cv2.dilate(img, np.ones((4,4), np.uint8), iterations=1, )
  # img=cv2.Canny(img, 150, 150)
  # img=cv2.dilate(img, np.ones((4,4), np.uint8), iterations=1, )
  # img=cv2.erode(img, np.ones((3,3), np.uint8), iterations=1)

  img= cv2.bitwise_not(cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1])
  # print(img.max())

  # img=img/255

  img2=split_image(img, img_size, crop)
  cv2.imshow('out', img)
  a=img2[5][0]
  # print(img2[8][1].max())
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  a = cv2.bitwise_not(cv2.morphologyEx(cv2.bitwise_not(a),cv2.MORPH_OPEN,kernel))

  # a=cv2.dilate(a, (3,3), iterations=3)

  # a=cv2.convertScaleAbs(a)
  # # cv2.cvtColor(a, cv2.COLOR_)
  # cnts, hierarchy = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # cv2.drawContours(a, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=2)
  # print(a.shape)
  nums=getNums(img2)
  # print(nums)
  # solve(nums)
  s=Solver(nums)
  print(nums)
  s.solve()
  # print_sudoku(nums)

  # cv2.imshow('out2', a)
  persist_image()


if __name__=='__main__':
  if sys.argv.__len__()!=2:
    print("Usage:\npython main.py path/to/image")
  else:
    main(sys.argv[1])
