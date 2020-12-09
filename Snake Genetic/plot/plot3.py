import numpy as np
import matplotlib.pyplot as plt

best_run = np.loadtxt('best_scores.txt')

plt.bar(np.arange(1, 101), best_run)
plt.xlabel('runs')
plt.ylabel('highest score')
plt.show()
