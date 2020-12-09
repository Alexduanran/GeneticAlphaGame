import sys
sys.path.append('/Users/alexduanran/Desktop/Machine Learning/Final Project/Pong Genetic/Snake Genetic')

from snake_test import Main
import numpy as np

for i in range(1, 101):
    scores = []
    mean_scores = []
    print(i)
    try:
        main = Main('Snake Genetic/plot/best_snakes_each_generation', 'snake'+str(i))
    except Exception as e:
        print(e)
        pass
    apples, apples_avg = main.getAvg()
    scores.append(apples)
    mean_scores.append(apples_avg)
    np.savetxt(f'Snake Genetic/plot/best_snakes_scores/scores{i}.txt', scores)
    np.savetxt(f'Snake Genetic/plot/best_snakes_scores/mean_scores{i}.txt', mean_scores)