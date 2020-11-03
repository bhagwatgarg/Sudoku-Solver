import cv2
import numpy as np

def persist_image():
  while True:
    if cv2.waitKey(1) & 0xff==ord('q'):
      break

def show_image():

  img=cv2.imread('./sherlock.jpg')
  img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #change color
  # img=cv2.GaussianBlur(img, (5,5), 1) #blur
  img=cv2.Canny(img, 200, 200)  #detect edges

  img=cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1) #make edges thicker
  img=cv2.erode(img, np.ones((3,3)), iterations=1)  #make edges thinner
  cv2.imshow('sherlock', img)
  persist_image()

if __name__=='__main__':
  show_image()