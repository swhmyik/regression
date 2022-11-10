from matplotlib.figure import Figure
import numpy as np

def main():
  # x, f(x)の準備
  x= np.linspace(-1,1,101)
  y= np.sin(np.pi * x)##m
  #グラフの作成
  fig = Figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.set_title('$y= \\sin(\\pi x)$')
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.plot(x,y)
  ax.axhline(color='#777777')
  ax.axvline(color='#777777')
  fig.savefig('out.png')

if __name__ == '__main__':
  main()