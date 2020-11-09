import sys
from typing import List
from snakeClass import *
import numpy as np
from scipy.spatial.distance import euclidean
import pygame
from neural_network import FeedForwardNetwork, sigmoid, linear, relu
from settings import settings
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, roulette_wheel_selection, tournament_selection
from genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.crossover import uniform_binary_crossover, single_point_binary_crossover
from math import sqrt
from decimal import Decimal
import random

RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GRAY = (0, 50, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 51)

class Main():
    def __init__(self):
        
        self._mutation_bins = np.cumsum([settings['probability_gaussian'],
                                        settings['probability_random_uniform']
        ])
        self._crossover_bins = np.cumsum([settings['probability_SBX'],
                                         settings['probability_SPBX']
        ])
        self._SPBX_type = settings['SPBX_type'].lower()
        self._SBX_eta = settings['SBX_eta']
        self._mutation_rate = settings['mutation_rate']

        # Determine size of next gen based off selection type
        self._next_gen_size = None
        if settings['selection_type'].lower() == 'plus':
            self._next_gen_size = settings['num_parents'] + settings['num_offspring']
        elif settings['selection_type'].lower() == 'comma':
            self._next_gen_size = settings['num_offspring']
        else:
            raise Exception('Selection type "{}" is invalid'.format(settings['selection_type']))

        
        self.board_size = settings['board_size']
        
        individuals: List[Individual] = []

        # Create initial generation
        for _ in range(settings['num_parents']):
            individual = Snake(200, 200, 3, "R", 20, 
                                board_size=self.board_size,
                                hidden_layer_architecture=settings['hidden_network_architecture'],
                                hidden_activation=settings['hidden_layer_activation'],
                                output_activation=settings['output_layer_activation'])
            individuals.append(individual)

        self.population = Population(individuals)
        self.apple = Apple(20, 100, 100, RED, self.board_size[0], self.board_size[1])

        self.current_generation = 0

        # Pygame
        pygame.init()
        screen = pygame.display.set_mode(self.board_size)
        pygame.display.set_caption('snake')

        # The loop will carry on until the user exit the game (e.g. clicks the close button).
        carryOn = True
        # The clock will be used to control how fast the screen updates
        clock = pygame.time.Clock()

        # Best of single generation
        self.winner = None
        # Best paddle of all generations
        self.champion = None
        self.champion_fitness = -1 * np.inf

        self.num_apples = 0
        self.best_apples = 0

        while carryOn:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    carryOn = False

            screen.fill(BLACK)
            #  draw the grid
            for x in range(0, self.board_size[0], 20):
                pygame.draw.line(screen, GRAY, (x, 0), (x, self.board_size[1]))
            for y in range(0, self.board_size[1], 20):
                pygame.draw.line(screen, GRAY, (0, y), (self.board_size[0], y))
            #  draw the apple
            pygame.draw.rect(screen, self.apple.color, self.apple.rect)
            
            font = pygame.font.Font('freesansbold.ttf', 18)
            generation_text = font.render("Generation: %d" % self.current_generation, True, WHITE)
            apples_text = font.render("Apples: %d" % self.num_apples, True, WHITE)
            best_apples_text = font.render("Best: %d" % self.best_apples, True, WHITE)
            screen.blit(generation_text, (self.board_size[0] - 150, 10))
            screen.blit(apples_text, (self.board_size[0] - 150, 40))
            screen.blit(best_apples_text, (self.board_size[0] - 150, 70))

            self.still_alive = 0
            self.not_reach_apple = 0
            # Loop through the snakes in the generation
            for snake in self.population.individuals:
                # Update snake if still alive
                if snake.is_alive:
                    self.still_alive += 1
                    if not snake.reach_apple:
                        self.not_reach_apple += 1
                        #----------------------------------------inputs for neural network--------------------------------------------
                        inputs = np.zeros((12, ))
                        # Direction of Snake
                        if snake.direction == "U":
                            inputs[0] = 1
                        elif snake.direction == "R":
                            inputs[1] = 1
                        elif snake.direction == "D":
                            inputs[2] = 1
                        elif snake.direction == "L":
                            inputs[3] = 1

                        # Apple position (wrt snake head)
                        # (0,0) at Top-Left Corner: U: -y; R: +x
                        if self.apple.y < snake.y:
                            # apple north snake
                            inputs[4] = 1
                        if self.apple.x > snake.x:
                            # apple east snake
                            inputs[5] = 1
                        if self.apple.y > snake.y:
                            # apple south snake
                            inputs[6] = 1
                        if self.apple.x < snake.x:
                            # apple west snake
                            inputs[7] = 1
                        
                        # Obstacle (Walls, body) position (wrt snake head)
                        body_x = [rect.x for rect in snake.body]
                        body_y = [rect.y for rect in snake.body]
                        body_pos = [(rect.x, rect.y) for rect in snake.body]
                        if snake.direction != "D" and \
                        (snake.y <= 0 or (snake.x, snake.y-20) in body_pos):
                            # obstacle at north
                            inputs[8] = 1
                        if snake.direction != "L" and \
                        (snake.x >= self.board_size[0]-20 or (snake.x+20, snake.y) in body_pos):
                            # obstacle at east
                            inputs[9] = 1
                        if snake.direction != "U" and \
                        (snake.y >= self.board_size[1]-20 or (snake.x, snake.y+20) in body_pos):
                            # obstacle at south
                            inputs[10] = 1
                        if snake.direction != "R" and \
                        (snake.x <= 0 or (snake.x-20, snake.y) in body_pos):
                            # obstacle at west
                            inputs[11] = 1
                        #----------------------------------------inputs for neural network--------------------------------------------
                        snake.updateDirection(inputs)
                        pos_cur = [snake.x, snake.y]
                        snake.addHead()
                        pos_next = [snake.x, snake.y]
                        pos_apple = [self.apple.x, self.apple.y]
                        d1 = euclidean(pos_apple, pos_cur)
                        d2 = euclidean(pos_apple, pos_next)
                        snake.distance += 1 if d1 > d2 else -1
                        if snake.isDead() or snake.isOutOfBounds(self.board_size[0], self.board_size[1]):
                            snake.is_alive = False
                        if not (snake.head.colliderect(self.apple.rect)):
                            snake.deleteTail()
                            snake.steps += 1
                            snake.total_steps += 1
                            if snake.steps > self.board_size[0] / 2 + 20:
                                snake.total_steps = np.inf
                                snake.is_alive = False
                        else:
                            snake.apples += 1
                            snake.steps = 0
                            snake.reach_apple = True

                # Draw every snake except the best ones
                if snake.is_alive and snake != self.winner and snake != self.champion:
                    snake.winner = False
                    snake.champion = False
                    # draw the snake
                    for part in snake.body:
                        pygame.draw.rect(screen, GREEN, part)
                        part_small = part.inflate(-3, -3)
                        pygame.draw.rect(screen, WHITE, part_small, 3)

            # Draw the winning and champion snake last
            if self.winner is not None and self.winner.is_alive:
                # draw the snake
                for part in snake.body:
                    pygame.draw.rect(screen, BLUE, part)
                    part_small = part.inflate(-3, -3)
                    pygame.draw.rect(screen, WHITE, part_small, 3)
            if self.champion is not None and self.champion.is_alive:
                # draw the snake
                for part in snake.body:
                    pygame.draw.rect(screen, YELLOW, part)
                    part_small = part.inflate(-3, -3)
                    pygame.draw.rect(screen, WHITE, part_small, 3)

            if self.not_reach_apple == 0:
                self.apple.move()
                self.num_apples += 1
                if self.num_apples > self.best_apples:
                    self.best_apples = self.num_apples
                for snake_ in self.population.individuals:
                    if snake_.is_alive:
                        snake.steps = 0
                        snake_.reach_apple = False

            # Generate new generation when all have died out
            if self.still_alive == 0:
                self.next_generation()
            
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
        
            # --- Limit to 60 frames per second
            clock.tick(60)

        #Once we have exited the main program loop we can stop the game engine:
        pygame.quit()

    def next_generation(self):
        self.current_generation += 1
        self.num_apples = 0

        # Calculate fitness of individuals
        for individual in self.population.individuals:
            individual.calculate_fitness()

        # Find winner from each generation and champion
        self.winner = self.population.fittest_individual
        if self.winner.fitness > self.champion_fitness:
            self.champion_fitness = self.winner.fitness
            self.champion = self.winner
        self.winner.winner = True
        self.champion.champion = True
        self.winner.reset()
        self.champion.reset()

        # Print results from each generation
        print('======================= Gneration {} ======================='.format(self.current_generation))
        print('----Max fitness:', self.population.fittest_individual.fitness)
        # print('----Best Score:', self.population.fittest_individual.score)
        print('----Average fitness:', self.population.average_fitness)
        
        self.population.individuals = elitism_selection(self.population, settings['num_parents'])
        
        random.shuffle(self.population.individuals)
        next_pop: List[Snake] = []
        x = random.randrange(0, (self.board_size[0] - 20), 20)
        y = random.randrange(0, (self.board_size[1] - 20), 20)
        self.winner.x = x
        self.winner.y = y
        self.champion.x = x
        self.champion.y = y
        # parents + offspring selection type ('plus')
        if settings['selection_type'].lower() == 'plus':
            next_pop.append(self.winner)
            next_pop.append(self.champion)

        while len(next_pop) < self._next_gen_size:
            p1, p2 = roulette_wheel_selection(self.population, 2)

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Each W_l and b_l are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]  
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                # Crossover
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # Assign children from crossover/mutation
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l

                # Clip to [-1, 1]
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
                np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])

             # Create children from chromosomes generated above
            c1 = Snake(x, y, 3, "R", 20, board_size=p1.board_size, chromosome=c1_params, hidden_layer_architecture=p1.hidden_layer_architecture,
                       hidden_activation=p1.hidden_activation, output_activation=p1.output_activation)
            c2 = Snake(x, y, 3, "R", 20, board_size=p2.board_size, chromosome=c2_params, hidden_layer_architecture=p2.hidden_layer_architecture,
                       hidden_activation=p2.hidden_activation, output_activation=p2.output_activation)

            # Add children to the next generation
            next_pop.extend([c1, c2])
        
        # Set the next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rand_crossover = random.random()
        crossover_bucket = np.digitize(rand_crossover, self._crossover_bins)
        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None

        # SBX
        if crossover_bucket == 0:
            child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, self._SBX_eta)
            child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, self._SBX_eta)

        # Single point binary crossover (SPBX)
        elif crossover_bucket == 1:
            child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights, major=self._SPBX_type)
            child1_bias, child2_bias =  single_point_binary_crossover(parent1_bias, parent2_bias, major=self._SPBX_type)
        
        else:
            raise Exception('Unable to determine valid crossover based off probabilities')

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        scale = .2
        rand_mutation = random.random()
        mutation_bucket = np.digitize(rand_mutation, self._mutation_bins)

        mutation_rate = self._mutation_rate
        if settings['mutation_rate_type'].lower() == 'decaying':
            mutation_rate = mutation_rate / sqrt(self.current_generation + 1)

        # Gaussian
        if mutation_bucket == 0:
            # Mutate weights
            gaussian_mutation(child1_weights, mutation_rate, scale=scale)
            gaussian_mutation(child2_weights, mutation_rate, scale=scale)

            # Mutate bias
            gaussian_mutation(child1_bias, mutation_rate, scale=scale)
            gaussian_mutation(child2_bias, mutation_rate, scale=scale)
        
        # Uniform random
        elif mutation_bucket == 1:
            # Mutate weights
            random_uniform_mutation(child1_weights, mutation_rate, -1, 1)
            random_uniform_mutation(child2_weights, mutation_rate, -1, 1)

            # Mutate bias
            random_uniform_mutation(child1_bias, mutation_rate, -1, 1)
            random_uniform_mutation(child2_bias, mutation_rate, -1, 1)

        else:
            raise Exception('Unable to determine valid mutation based off probabilities.')


if __name__ == "__main__":
    main = Main()