from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
  resolution = [800, 1600, 2400, 3200, 4000]
  runtime = [26.5758, 117.635, 304.751, 553.8, 923.403]

  plt.figure()
  plt.plot(resolution, runtime, '.-')
  plt.xticks(resolution)
  plt.xlabel("resolution")
  plt.ylabel("runtime [ms]")
  plt.title("Denoising Runtime vs. Resolution")

  filter_size = [5, 10, 20, 40, 80, 100]
  runtime = [1.03856, 2.08966, 4.5568, 11.1124, 20.5313, 27.465]

  plt.figure()
  plt.plot(filter_size, runtime, '.-')
  plt.xticks(filter_size)
  plt.xlabel("filter size")
  plt.ylabel("runtime [ms]")
  plt.title("Denoising Runtime vs. Filter Size")

  plt.show()
