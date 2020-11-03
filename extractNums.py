import cv2
import numpy as np
import os
from contours import get_sudoku
from convertImage import persist_image
import tensorflow as tf
from model import train_model

def split_image(img, img_size, crop):
  i, j, add = 0, 0, img_size//9
  new=np.zeros((9, 9, add-2*crop, add-2*crop))

  for i in range(9):
    for j in range(9):
      new[j][i]=img[add*j+crop:add*(j+1)-crop, add*i+crop:add*(i+1)-crop]
  return new

  # img2=img[add*i:add*(i+1), add*i:add*(i+1)]
  # for j in range(1, 9):
  #   img2=np.hstack((img2, img[add*i:add*(i+1), add*j:add*(j+1)]))
  # img3=img2
  # for i in range(1, 9):
  #   img2=img[add*i:add*(i+1), :add]
  #   for j in range(1, 9):
  #     img2=np.hstack((img2, img[add*i:add*(i+1), add*j:add*(j+1)]))
  #   img3=np.vstack((img3, img2))
  # return img3
def getNums(img):
  if len(os.listdir('./Model'))==0:
    train_model()
  model=tf.keras.models.load_model('./Model/model.h5')
  # nums=np.zeros((img.shape[0], img.shape[1]))
  img=img.reshape((-1, img.shape[2], img.shape[3], 1))
  print(img.shape)
  nums=model.predict(img//255)
  nums=np.argmax(nums, axis=1).reshape((9, 9))
  return nums
  
def print_sudoku(nums):
  for i in range(9):
    s=''
    for j in range(9):
      s=s+str(nums[i][j])+' '
    print(s)

def main():
  img_size=74*9
  add=img_size//9
  crop=5
  imgi=get_sudoku('./sudoku2.jpeg', img_size)

  img=cv2.bilateralFilter(imgi, 5, 20, 20)
  img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # img=cv2.dilate(img, np.ones((4,4), np.uint8), iterations=1, )
  # img=cv2.Canny(img, 150, 150)
  # img=cv2.dilate(img, np.ones((4,4), np.uint8), iterations=1, )
  # img=cv2.erode(img, np.ones((3,3), np.uint8), iterations=1)

  img= cv2.bitwise_not(cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1])

  img2=split_image(img, img_size, crop)
  # print(img2.shape)
  cv2.imshow('out', img)
  a=img2[5][7]
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  a = cv2.bitwise_not(cv2.morphologyEx(cv2.bitwise_not(a),cv2.MORPH_OPEN,kernel))

  # a=cv2.dilate(a, (3,3), iterations=3)

  # a=cv2.convertScaleAbs(a)
  # # cv2.cvtColor(a, cv2.COLOR_)
  # cnts, hierarchy = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # cv2.drawContours(a, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=2)
  # print(a.shape)
  nums=getNums(img2)
  print_sudoku(nums)

  cv2.imshow('out2', a)
  persist_image()


if __name__=='__main__':
  main()