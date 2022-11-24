from matplotlib.figure import Figure
import numpy as np
import japanize_matplotlib as _

def main():
  # x, f(x)の準備
  x= np.linspace(-1,1,101)
  y= np.sin(np.pi * x)##m
  #sample
  x_sample = np.random.uniform(-1,1,(20, ))
  range_y = np.max(y) - np.min(y)
  noise_sample = np.random.normal(0, range_y*0.05, (20,))
  y_sample = np.sin(np.pi * x_sample) + noise_sample
  #グラフの作成
  fig = Figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.set_title('$y= \\sin(\\pi x)$')
  ax.set_xlabel('$x$')
  ax.set_ylabel('$y$')
  ax.plot(x,y, label='真の関数 ')
  ax.axhline(color='#777777')
  ax.axvline(color='#777777')
  ax.scatter(x_sample, y_sample, color='red', label='学習サンプル')
  ax.legend()

  fig.savefig('out.png')

if __name__ == '__main__':
  main()