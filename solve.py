import numpy as np

class Solver():
  def __init__(self, ar):
    self.a=ar
    self.b=ar
    self.check_if_solvable()

  def check_if_solvable(self):
    self.solvable=True
    for i in range(0, 9):
      for j in range(0, 9):
        if self.a[i][j]==0:
          continue
        if self.check(i, j)[self.a[i][j]]==0:
          self.solvable=False
          return False

  def solve(self):
    if not self.solvable:
      print('Suduko not Solvable')
      return False
    res=False
    if self.a[0][0]!=0:
      res=self.back(0, 1)
    else:
      for i in range(1, 10):
        self.a[0][0]=i
        res=self.back(0, 1)
        if res:
          break
    if res:
      self.check_if_solvable()
      print(self.a)
      return self.a
    else: print("Not Solvable")
    return False

  def back(self, i, j):
    if i>8 or j>8: return True
    nexti, nextj=i+int(j==8), (j+1)%9
    if self.a[i][j]!=0: return self.back(nexti, nextj)
    possible=self.check(i, j)
    for k in range(1, 10):
      if not possible[k]:
        continue
      self.a[i][j]=k
      res=self.back(nexti, nextj)
      if res:
        return True
    self.a[i][j]=0
    return False

  def check(self, i, j):
    possible=np.ones((10), np.int)
    for k in range(0, 9):
      if k==j: continue
      possible[self.a[i][k]]=0

    for k in range(0, 9):
      if k==i: continue
      possible[self.a[k][j]]=0
    for a1 in range(0, 3):
      for b1 in range(0, 3):
        if (i//3)*3+a1==i and (j//3)*3+b1==j:
          continue
        possible[self.a[(i//3)*3+a1][(j//3)*3+b1]]=0
    return possible

if __name__=='__main__':
  a=np.zeros((9, 9), np.int)
  for i in range(0, 9):
    a[i, :]=np.array([int(k) for k in input().split(' ')])
  # solve(a)
  s=Solver(a)
  s.solve()

''' 
0 0 4 3 0 0 2 0 9
0 0 5 0 0 9 0 0 1
0 7 0 0 6 0 0 4 3
0 0 6 0 0 2 0 8 7
1 9 0 0 0 7 4 0 0
0 5 0 0 8 3 0 0 0
6 0 0 0 0 0 1 0 5
0 0 3 5 0 8 6 9 0
0 4 2 9 1 0 3 0 0
'''

'''
8 6 4 3 7 1 2 5 9
3 2 5 8 4 9 7 6 1
9 7 1 2 6 5 8 4 3
4 3 6 1 9 2 5 8 7
1 9 8 6 5 7 4 3 2
2 5 7 4 8 3 9 1 6
6 8 9 7 3 4 1 2 5
7 1 3 5 2 8 6 9 4
5 4 2 9 1 6 3 7 8
'''
