import paddle, ball, config
import numpy as np
import copy

class Population:
    def __init__(self, size):
        self.size = size
        self.paddles = [paddle.Paddle() for _ in range(size)]
        self.balls = [ball.Ball() for _ in range(size+1)]
        self.fitness = []

    def new_population(self, winner, ballx, champion):
        new_population = []
        # self.calculateFitness()
        for _ in range(self.size - 2):
            parent = self.selectParent()
            parent_coefs = self.mutateCoefs(parent.coefs)
            new_population.append(paddle.Paddle(coefs=parent_coefs))
        winner.reset()
        champion.reset()
        new_population.append(winner)
        new_population.append(champion)
        self.balls = [ball.Ball(x=ballx) for _ in range(self.size+1)]
        self.paddles = new_population

    def calculateFitness(self):
        self.fitness = []
        for paddle in self.paddles:
            self.fitness.append(paddle.calculateFitness())
        # print(self.fitness)

    def selectParent(self):
        selection = np.random.choice(np.arange(self.size), 50, replace=False)
        max_fitness = -1 * float('inf')
        max_ = -1
        for i in selection:
            if self.fitness[i] > max_fitness:
                max_fitness = self.fitness[i]
                max_ = i
        print(max_fitness)
        return self.paddles[max_]

    #Returns mutated coefs
    def mutateCoefs(self, coefs):
	    newCoefs = copy.deepcopy(coefs)
	    for i in range(len(newCoefs)):
		    for row in range(len(newCoefs[i])):
			    for col in range(len(newCoefs[i][row])):    
			        if np.random.random() <= config.MUTATION_RATE:
				        newCoefs[i][row][col] = np.random.normal(newCoefs[i][row][col], 1)
	    return newCoefs