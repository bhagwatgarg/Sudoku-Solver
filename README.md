# Sudoku Solver
<p align=center>
  <img src='https://img.shields.io/github/issues/bhagwatgarg/sudoku-solver'>
  <img src='https://img.shields.io/github/forks/bhagwatgarg/sudoku-solver'>
  <img src='https://img.shields.io/github/stars/bhagwatgarg/sudoku-solver'>
</p>
<h1 align="center">
  <br>
  <img src='icon.png' width='160'>
  
</h1>

<h4 align="center">A tool that solves sudokus by extracting sudoku puzzles from the user-provided images</h4>

## How Does it Work
- The user provided image is processed using the OpenCV library and the puzzle is extracted from the image
- A neural network model is then used to extract the numbers from the matrix
- Then the sudoku is solved using backtracking algorithm

## Using the tool
Clone the repo
```
git clone https://github.com/bhagwatgarg/sudoku-solver
cd Sudoku-Solver
```
Install required dependencies and run the tool.
```
pip install -r requirements.txt
python main.py sample.jpeg
```
If it does not work properly, then try installing the tested package versions by uncommenting the tested versions in requirements.txt

## Limitations
<p>The tool relies on OpenCV for extracting the boundaries of the puzzle. Sometimes the boundaries arent clear enough which could lead to unwanted results.</p>


<hr>
<div>Icons made by <a href="https://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>