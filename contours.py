import cv2
import numpy as np
from convertImage import persist_image

#https://stackoverflow.com/questions/61643039/improving-canny-edge-detection

def sort_contours_size(cnts):
  """ Sort contours based on the size"""

  cnts_sizes = [cv2.contourArea(contour) for contour in cnts]
  arr=reversed(sorted(zip(cnts_sizes, cnts), key=lambda x: x[0]))
  cnts=[a[1] for a in arr]
  return cnts_sizes, cnts

def arrange_points(points):
  [midx, midy]=np.sum(points, axis=0)/4
  points_arr=np.zeros_like(points)
  for point in points:
    [x,y]=point
    if x<=midx and y<=midy:
      points_arr[0]=np.array(point)
    elif x>=midx and y<=midy:
      points_arr[1]=np.array(point)
    elif x<=midx and y>=midy:
      points_arr[2]=np.array(point)
    else:
      points_arr[3]=np.array(point)
  return points_arr

def get_sudoku(name='./sudoku.jpeg', img_size=500):
  imgi=cv2.imread(name)
  # print(imgi)
  #processing
  img=cv2.cvtColor(imgi, cv2.COLOR_BGR2GRAY)
  img=cv2.GaussianBlur(img, (3,3), 3)
  img=cv2.Canny(img, 150, 150)
  img=cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1)
  img=cv2.erode(img, np.ones((3,3), np.uint8), iterations=1)
  points=(cv2.goodFeaturesToTrack(img, 4, 0.8, 100))
  cnt, h=(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
  # cv2.drawContours(img, cnt, -1, (255, 255, 255), 5)
  img2=np.zeros_like(img)
  sizes, cnt=sort_contours_size(cnt)

  cv2.drawContours(img2, cnt, 0, (255, 255, 255), 5)
  points=(cv2.goodFeaturesToTrack(img2, 4, 0.5, 50))
  points1=np.float32([[point[0][0], point[0][1]] for point in points])

  points1=arrange_points(points1)

  points2=np.float32([[0, 0], [img_size, 0], [0, img_size], [img_size, img_size]])

  mat=cv2.getPerspectiveTransform(points1, points2)
  final=cv2.warpPerspective(imgi, mat, (img_size, img_size))


  for point in points1:
    cv2.circle(img2, (point[0], point[1]), 5, (0, 0, 255), cv2.FILLED)
  # img=cv2.dilate(img, np.ones((2,2), np.uint8), iterations=1)
  if __name__=='__main__':
    cv2.imshow('out', final)
    persist_image()
  else:
    return final



if __name__=='__main__':
  get_sudoku()