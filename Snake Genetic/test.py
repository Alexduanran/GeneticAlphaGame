import sys
from snake_genetic import *

best_scores = []

for i in range(100):
    print(i)
    try:
        main = Main()
        best_scores.append(main.best_apples)
    except Exception as e:
        print(e)
        pass

np.savetxt(f'plot/best_scores.txt', best_scores)