from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
  iterations = np.array([300, 500, 1000, 2000])
  times_cache_first_bounce = np.array([5.18, 8.21, 15.26, 30.07])  # s
  times_no_cache = np.array([4.94, 8.62, 15.92, 30.55])  # s
  times_cache_radix_sort = np.array([8.69, 14.20, 28.01, 54.99])  # s

  plt.figure()
  plt.plot(iterations, times_cache_first_bounce, '.-', iterations,
           times_no_cache, '.-', iterations, times_cache_radix_sort, '.-')
  plt.xlabel("No. iterations")
  plt.ylabel("Runtime [s]")
  plt.title("Runtime vs. No. iterations")
  plt.legend([
      "Cache first bounce, no sort", "No cache, no sort",
      "Cache first bounce, radix sort"
  ])
  plt.show()
