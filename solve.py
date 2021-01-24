import numpy as np

class Solver():
  def __init__(self, ar):
    """
      Initialize the solver with the unsolved sudoku array
      Arguments:
        ar: 9*9 array with missing values substituted as 0
    """
    self.a=ar                 #copy of ar
    self.b=ar                 #copy of ar
    self.check_if_solvable()  #checks if sudoku is solvable or not

  def check_if_solvable(self):
    """
      Checks if sudoku is solvable or not
      Returns:
        True : Sudoku is solvable
        False: Sudoku is not solvable
    """

    self.solvable=True      #status of sudoku
    for i in range(0, 9):
      for j in range(0, 9):
        if self.a[i][j]==0:
          continue
        if self.check(i, j)[self.a[i][j]]==0:
          self.solvable=False
          return False

  def solve(self):
    """
      A functions which is called to solve the sudoku
      Returns:
        Solved Sudoku: If sudoku is solvable
        False        : Otherwise
    """
    if not self.solvable:
      print('Suduko not Solvable')
      return False
    res=self.back(0, 0)
    # if self.a[0][0]!=0:
    #   res=self.back(0, 1)
    # else:
    #   for i in range(1, 10):
    #     self.a[0][0]=i
    #     res=self.back(0, 1)
    #     if res:
    #       break
    if res:
      self.check_if_solvable()
      print("Sudoku Solved!")
      print(self.a)
      return self.a
    else: print("Not Solvable")
    return False

  def back(self, i, j):
    """
      A recursive back-tracking algorithm for solving sudoku
      Arguments:
        i: row index
        j: column index
      Returns:
        True : if a solution exists
        False: if no solution exists
    """
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
    """
      Finds which values are valid for a given cell in the sudoku matrix
      Arguments:
        i: row index
        j: column index
      Returns:
        A numpy array of size 10 with the value at each index signifying whether it is possible to place that number in that location
        (1 if possible; 0 otherwise)
    """
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
  # t=time.time()
  s.solve()
  # print(str(time.time()-t))

#S1
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

#S2
'''
8 0 0 0 0 0 0 0 0
0 0 3 6 0 0 0 0 0
0 7 0 0 9 0 2 0 0
0 5 0 0 0 7 0 0 0
0 0 0 0 4 5 7 0 0
0 0 0 1 0 0 0 3 0
0 0 1 0 0 0 0 6 8
0 0 8 5 0 0 0 1 0
0 9 0 0 0 0 4 0 0
'''
'''
8 1 2 7 5 3 6 4 9
9 4 3 6 8 2 1 7 5
6 7 5 4 9 1 2 8 3
1 5 4 2 3 7 8 9 6
3 6 9 8 4 5 7 2 1
2 8 7 1 6 9 5 3 4
5 2 1 9 7 4 3 6 8
4 3 8 5 2 6 9 1 7
7 9 6 3 1 8 4 5 2
'''
