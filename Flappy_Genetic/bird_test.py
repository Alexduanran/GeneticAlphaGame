import sys
sys.path.append('/Users/alexduanran/Desktop/Machine Learning/Final Project/Pong Genetic/Flappy_Genetic')

import math
from typing import List
from bird import *
from ball import *
import numpy as np
import pygame
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

        self.index = 0
        self.scores = []
        
        self.Window_Width = settings['Window_Width']
        self.Window_Height = settings['Window_Height']
        

        #### Pygame init ####
        pygame.init()
        self.screen = pygame.display.set_mode((self.Window_Width, self.Window_Height))
        pygame.display.set_caption('Flappy Bird')
        self.game_font = pygame.font.Font("04B_19.ttf", 40)


        self.bird = self.load_bird(folder, name, settings)

        #### Pygame ####

        

        # The loop will carry on until the user exit the game (e.g. clicks the close button).
        game_active = True
        # The clock will be used to control how fast the screen updates
        clock = pygame.time.Clock()

        # load assets
        self.bg_surface = pygame.image.load("assets/background-day.png").convert()
        self.bg_surface = pygame.transform.scale2x(self.bg_surface)

        self.floor_surface = pygame.image.load("assets/base.png").convert()
        self.floor_surface = pygame.transform.scale2x(self.floor_surface)
        self.floor_x = 0

        self.pipe_surface = pygame.image.load("assets/pipe-green.png")
        self.pipe_surface = pygame.transform.scale2x(self.pipe_surface)
        self.pipe_flip_surface = pygame.transform.flip(self.pipe_surface, False, True)
        self.pipe_list = []
        self.pipe_list.extend(self.create_pipe())
        # self.SPAWNPIPE = pygame.USEREVENT
        # pygame.time.set_timer(self.SPAWNPIPE, 1200)               # don't use this because sometimes the game lags and reuslts in pipes spawning closer than intended
        self.spawn_pipe_counter = 0

        # scores
        self.score = 0
        self.high_score = 0
        





        while game_active and self.index <= 100:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_active = False
                    pygame.quit()
                    # sys.exit()        // this or exit()
                    exit()
                # if event.type == self.SPAWNPIPE and game_active:
                #     self.pipe_list.extend(self.create_pipe())

            # screen.fill(BLACK)
            self.screen.blit(self.bg_surface, (0, 0))


            # Pipes
            self.spawn_pipe_counter += 2
            
            # if self.spawn_pipe_counter % 72 == 0:
            if self.spawn_pipe_counter % settings['pipe_interval_in_frames'] == 0:
                self.pipe_list.extend(self.create_pipe())
            self.pipe_list = self.move_pipes(self.pipe_list)
            self.draw_pipes(self.pipe_list) 

            # Floor
            self.floor_x = (self.floor_x - 6) % - self.Window_Width
            self.draw_floor()

            # TODO: should change this for better visibility
            font = pygame.font.Font('freesansbold.ttf', 18)
            score_text = font.render("Score: %d" % self.score, True, WHITE)
            self.screen.blit(score_text, (self.Window_Width - 150, 60))

               
            #----------------------------------------inputs for neural network--------------------------------------------
            # next_pipe = next(pipe for pipe in self.pipe_list if pipe.right > settings['init_bird_x_pos'])
            # next_next_pipe = next(pipe for pipe in self.pipe_list if pipe.right > next_pipe.right)
            next_pipe, next_next_pipe = self.get_next_pipes()
            self.bird.x_distance_to_next_pipe_center = next_pipe.right - settings['init_bird_x_pos']
            self.bird.y_distance_to_next_pipe_center = (next_pipe.top - 150) - self.bird.y_pos
            if next_next_pipe != None:
                self.bird.x_distance_to_next_next_pipe_center = next_next_pipe.right - settings['init_bird_x_pos']
                self.bird.y_distance_to_next_next_pipe_center = (next_next_pipe.top - 150) - self.bird.y_pos
            else:
                self.bird.x_distance_to_next_next_pipe_center = None
                self.bird.y_distance_to_next_next_pipe_center = None
            inputs = np.array([[self.bird.y_speed], [self.bird.y_pos], [self.bird.x_distance_to_next_pipe_center], [self.bird.y_distance_to_next_pipe_center], [self.bird.y_distance_to_next_next_pipe_center]])
            # inputs = np.array([[paddle.x_pos], [ball_distance_left_wall], [ball_distance_right_wall], [balls[i].xspeed], [paddle.xspeed], [balls[i].y], [balls[i].yspeed]])
            # inputs = np.array([[paddle.x_pos], [balls[i].xspeed], [paddle.xspeed], [balls[i].x]])
            #----------------------------------------inputs for neural network--------------------------------------------
            self.bird.update(inputs)
            self.bird.move(self.pipe_list)
            self.score = max(self.score, self.bird.score)
            
            if not self.bird.is_alive or self.score > 160:
                self.reset()
            else:
                self.bird.draw(self.screen)




            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.update()
        
            # --- Limit to 60 frames per second
            clock.tick(60)

        #Once we have exited the main program loop we can stop the game engine:
        pygame.quit()


    def reset(self):
        self.index += 1
        self.scores.append(math.floor(self.bird.score-3.4))

        self.pipe_list = []
        self.pipe_list.extend(self.create_pipe())
        # self.SPAWNPIPE = pygame.USEREVENT
        # pygame.time.set_timer(self.SPAWNPIPE, 1200)               # don't use this because sometimes the game lags and reuslts in pipes spawning closer than intended
        self.spawn_pipe_counter = 0

        self.bird.is_alive = True
        self.bird.x_distance_to_next_pipe_center = 0
        self.bird.y_distance_to_next_pipe_center = 0
        self.bird.score = 0
        self.bird.x_pos = settings['init_bird_x_pos']
        self.bird.y_pos = settings['init_bird_y_pos']
        # self.bird.y_speed = 0

        # scores
        self.score = 0


    def draw_floor(self):
        self.screen.blit(self.floor_surface, (self.floor_x, self.Window_Height - 100))
        self.screen.blit(self.floor_surface, (self.floor_x + self.Window_Width, self.Window_Height - 100))

    def create_pipe(self):
        random_pipe_pos = random.choice([400, 500, 600, 700, 800])
        bottom_pipe = self.pipe_surface.get_rect(midtop = (self.Window_Width + 100, random_pipe_pos))
        top_pipe = self.pipe_surface.get_rect(midbottom = (self.Window_Width + 100, random_pipe_pos - 300))
        return bottom_pipe, top_pipe

    def move_pipes(self, pipes):
        for pipe in pipes:
            pipe.centerx -= 10
        return pipes

    def draw_pipes(self, pipes):
        for pipe in pipes:
            if pipe.bottom > self.Window_Height:
                self.screen.blit(self.pipe_surface, pipe)
            else:
                self.screen.blit(self.pipe_flip_surface, pipe)

    def get_next_pipes(self):
        next_pipe = None
        next_next_pipe = None
        for pipe in self.pipe_list:
            if pipe.right > settings['init_bird_x_pos'] and next_pipe == None:
                next_pipe = pipe
            elif next_pipe != None and next_next_pipe == None:
                next_next_pipe = pipe
        return next_pipe, next_next_pipe


    def load_bird(self, population_folder: str, individual_name: str, settings: Optional[Union[Dict[str, Any], str]] = None) -> Bird:
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

        bird = Bird(chromosome=params,
                    hidden_layer_architecture=settings['hidden_network_architecture'],
                    hidden_activation=settings['hidden_layer_activation'],
                    output_activation=settings['output_layer_activation'])
        return bird

    def getAvg(self):
        return self.scores, np.mean(self.scores)

    

    

if __name__ == "__main__":
    main = Main('plot/best_birds_each_generation', 'bird40')