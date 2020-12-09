import sys
sys.path.append('/Users/alexduanran/Desktop/Machine Learning/Final Project/Pong Genetic/Flappy_Genetic')

from bird_test import Main
import numpy as np

for i in range(49, 51):
    scores = []
    mean_scores = []
    print(i)
    try:
        main = Main('plot/best_birds_each_generation', 'bird'+str(i))
    except Exception as e:
        print(e)
        pass
    score, score_avg = main.getAvg()
    scores.append(score)
    mean_scores.append(score_avg)
    np.savetxt(f'plot/best_birds_scores/scores{i}.txt', scores)
    np.savetxt(f'plot/best_birds_scores/mean_scores{i}.txt', mean_scores)