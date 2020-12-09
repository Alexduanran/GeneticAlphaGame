import sys
sys.path.append('/Users/alexduanran/Desktop/Machine Learning/Final Project/Pong Genetic/Pong Genetic V2')

from pong_test import Main
import numpy as np

for i in range(1, 51):
    scores = []
    mean_scores = []
    print(i)
    try:
        main = Main('plot/best_paddle_each_generation', 'paddle'+str(i))
    except Exception as e:
        print(e)
        pass
    score, score_avg = main.getAvg()
    scores.append(score)
    mean_scores.append(score_avg)
    np.savetxt(f'plot/best_paddle_scores/scores{i}.txt', scores)
    np.savetxt(f'plot/best_paddle_scores/mean_scores{i}.txt', mean_scores)