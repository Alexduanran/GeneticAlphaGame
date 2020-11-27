from snake_genetic import *

for i in range(500):
    print(i)
    try:
        main = Main()
        print('apples:', main.best_apples)
    except Exception as e:
        print(e)
        pass