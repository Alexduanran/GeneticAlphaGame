import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any
from fractions import Fraction
import random
import pygame
import sys
import os
import json

from misc import *
from genetic_algorithm.individual import Individual
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,180,0)
BLUE = (50,200,255)

class Paddle(Individual):
    def __init__(self, board_size: Tuple[int, int],
                 chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                 x_pos: Optional[int] = 400, 
                 y_pos: Optional[int] = 580,
                 xspeed: Optional[int] = 0,
                 hidden_layer_architecture: Optional[List[int]] = [12, 20],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid'
                 ):

        self._fitness = 0  # Overall fitness
        self.hit = 0
        self.distance_travelled = 0
        self.ball_travelled = 0
        self.distance_to_ball = 0
        self.is_alive = True

        self.board_size = board_size
        self.hidden_layer_architecture = hidden_layer_architecture
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.xspeed = xspeed

        # Setting up network architecture
        # Each "Vision" has 3 distances it tracks: wall, apple and self
        # there are also one-hot encoded direction and one-hot encoded tail direction,
        # each of which have 4 possibilities.
        num_inputs = 6 #@TODO: Add one-hot back in 
        self.network_architecture = [num_inputs]                          # Inputs
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden layers
        self.network_architecture.append(3)                               # 4 outputs, ['u', 'd', 'l', 'r']
        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
        )

        # If chromosome is set, take it
        if chromosome:
            # self._chromosome = chromosome
            self.network.params = chromosome
            # self.decode_chromosome()
        else:
            # self._chromosome = {}
            # self.encode_chromosome()
            pass

        

    @property
    def fitness(self):
        return self._fitness
    
    def calculate_fitness(self):
        # Give positive minimum fitness for roulette wheel selection
        # self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)**1.3) * (self.score**1.2))
        # self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)) * (self.score))
        self._fitness = (2 ** self.hit + self.hit * 2.1) * 200 + ((1 - min(self.distance_travelled / self.ball_travelled, 1)) * 400) + (self.board_size[0] - self.distance_to_ball) * 0.5
        # self._fitness = self.hit ** 1.2
        self._fitness = max(self._fitness, .1)

    @property
    def chromosome(self):
        # return self._chromosome
        pass

    def encode_chromosome(self):
        # # L = len(self.network.params) // 2
        # L = len(self.network.layer_nodes)
        # # Encode weights and bias
        # for layer in range(1, L):
        #     l = str(layer)
        #     self._chromosome['W' + l] = self.network.params['W' + l].flatten()
        #     self._chromosome['b' + l] = self.network.params['b' + l].flatten()
        pass

    def decode_chromosome(self):
        # # L = len(self.network.params) // 2
        # L = len(self.network.layer_nodes)
        # # Decode weights and bias
        # for layer in range(1, L):
        #     l = str(layer)
        #     w_shape = (self.network_architecture[layer], self.network_architecture[layer-1])
        #     b_shape = (self.network_architecture[layer], 1)
        #     self.network.params['W' + l] = self._chromosome['W' + l].reshape(w_shape)
        #     self.network.params['b' + l] = self._chromosome['b' + l].reshape(b_shape)
        pass

    def reset(self):
        self._fitness = 0
        self.hit = 0
        self.distance_travelled = 0
        self.ball_travelled = 0
        self.distance_to_ball = 0
        self.is_alive = True

    def update(self, inputs=None, ball=None):
        if inputs is not None:
            self.network.feed_forward(inputs)
        elif ball:
            self.x_pos = ball.x - 50
            return
        if self.network.out == 0:
            self.xspeed = -15
        elif self.network.out == 1:
            self.xspeed = 15
        elif self.network.out == 2:
            self.xspeed = 0
        return True

    def move(self):
        self.x_pos += self.xspeed
        self.distance_travelled += abs(self.xspeed)
        if self.x_pos < 0:
            self.x_pos = 0
        elif self.x_pos > self.board_size[0]-100:
            self.x_pos = self.board_size[0]-100

    # Draw the paddle
    def draw(self, screen, winner=False, champion=False):
	    if not self.is_alive:
		    return
	    if champion:
		    pygame.draw.rect(screen,BLACK,[self.x_pos,self.y_pos,100,20])
		    pygame.draw.rect(screen,GREEN,[self.x_pos+2,self.y_pos+2,100-4,20-4])
	    elif winner:
		    pygame.draw.rect(screen,BLACK,[self.x_pos,self.y_pos,100,20])
		    pygame.draw.rect(screen,BLUE,[self.x_pos+2,self.y_pos+2,100-4,20-4])
	    else:
		    pygame.draw.rect(screen,BLACK,[self.x_pos,self.y_pos,100,20])
		    pygame.draw.rect(screen,WHITE,[self.x_pos+2,self.y_pos+2,100-4,20-4])
        
       

