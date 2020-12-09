import sys
import shutil
from typing import List
from paddle import *
from ball import *
import numpy as np
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

class Main():
    def __init__(self, folder, name):
        
        self.index = 1
        self.hits = []

        
        self.board_size = settings['board_size']
        
        self.paddle = self.load_paddle(folder, name, settings)

        # Pygame
        pygame.init()
        screen = pygame.display.set_mode(self.board_size)
        pygame.display.set_caption('pong')

        # The loop will carry on until the user exit the game (e.g. clicks the close button).
        carryOn = True
        # The clock will be used to control how fast the screen updates
        clock = pygame.time.Clock()

        self.num_hit = 0

        ball_x = np.random.randint(0, 800)
        xspeed = 15 if np.random.random() > 0.5 else -15

        self.ball = Ball(x=ball_x, xspeed=xspeed)

        while carryOn and self.index <= 10:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    carryOn = False

            screen.fill(BLACK)

            font = pygame.font.Font('freesansbold.ttf', 18)
            hit_text = font.render("Hits: %d" % self.num_hit, True, WHITE)
            screen.blit(hit_text, (self.board_size[0] - 150, 60))

        

        
            #----------------------------------------inputs for neural network--------------------------------------------
            inputs = np.array([[self.paddle.x_pos], [self.ball.xspeed], [self.paddle.xspeed], [self.ball.x], [self.ball.yspeed], [self.ball.y]])
            # inputs = np.array([[paddle.x_pos], [balls[i].xspeed], [paddle.xspeed], [balls[i].x]])
            #----------------------------------------inputs for neural network--------------------------------------------
            self.paddle.update(inputs)
            self.ball.update(self.paddle)
            self.num_hit = max(self.num_hit, self.paddle.hit)
            self.ball.update_pos()
            self.paddle.move()

            self.paddle.draw(screen)
            self.ball.draw(screen)
    
            if not self.paddle.is_alive:
                self.reset()
            
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
        
            # --- Limit to 60 frames per second
            clock.tick(60)

        #Once we have exited the main program loop we can stop the game engine:
        pygame.quit()

    def reset(self):
        self.index += 1
        self.hits.append(self.paddle.hit)

        self.num_hit = 0
        self.paddle.hit = 0
        self.paddle.distance_travelled = 0
        self.paddle.ball_travelled = 0
        self.paddle.distance_to_ball = 0
        self.paddle.is_alive = True
        self.paddle.x_pos = 400
        ball_x = np.random.randint(0, 800)
        xspeed = 15 if np.random.random() > 0.5 else -15
        self.ball.x = ball_x
        self.ball.xspeed = xspeed
        self.ball.y = 50


    def load_paddle(self, population_folder: str, individual_name: str, settings: Optional[Union[Dict[str, Any], str]] = None) -> Paddle:
        # if not settings:
        #     f = os.path.join(population_folder, 'settings.json')
        #     if not os.path.exists(f):
        #         raise Exception("settings needs to be passed as an argument if 'settings.json' does not exist under population folder")
            
        #     with open(f, 'r', encoding='utf-8') as fp:
        #         settings = json.load(fp)

        # elif isinstance(settings, dict):
        #     settings = settings

        # elif isinstance(settings, str):
        #     filepath = settings
        #     with open(filepath, 'r', encoding='utf-8') as fp:
        #         settings = json.load(fp)

        params = {}
        for fname in os.listdir(os.path.join(population_folder, individual_name)):
            extension = fname.rsplit('.npy', 1)
            if len(extension) == 2:
                param = extension[0]
                params[param] = np.load(os.path.join(population_folder, individual_name, fname))
            else:
                continue

        # # Load constructor params for the specific snake
        # constructor_params = {}
        # snake_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
        # with open(snake_constructor_file, 'r', encoding='utf-8') as fp:
        #     constructor_params = json.load(fp)

        paddle = Paddle(self.board_size, chromosome=params, hidden_layer_architecture=settings['hidden_network_architecture'],
                              hidden_activation=settings['hidden_layer_activation'],
                              output_activation=settings['output_layer_activation'])
        return paddle

    def getAvg(self):
        return self.hits, np.mean(self.hits)

if __name__ == "__main__":
    main = Main('plot/best_paddle_each_generation', 'paddle2')