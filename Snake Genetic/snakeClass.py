from pygame import Rect
import random
import numpy as np

from typing import Tuple, Optional, Union, Set, Dict, Any
from misc import *
from genetic_algorithm.individual import Individual
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name


class Snake:
    """Snake object for snake game.
    Attributes:
    - length
    - x
    - y
    - color
    - direction
    - boxSize
    - body"""

    def __init__(self, x, y, length, direction, boxSize,
                board_size: Tuple[int, int],
                chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                hidden_layer_architecture: Optional[List[int]] = [12, 20],
                hidden_activation: Optional[ActivationFunction] = 'relu',
                output_activation: Optional[ActivationFunction] = 'sigmoid'):
        self.x = x
        self.y = y
        self.length = length
        self.direction = direction
        self.boxSize = boxSize
        self.board_size = board_size
        self.body = []

        self._fitness = 0
        self.is_alive = True
        self.reach_apple = False
        self.apples = 0
        self.total_steps = 0
        self.steps = 0

        self.winner = False
        self.champion = False

        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        k1 = 0
        k2 = 0
        if self.direction == "R":
            k1 = -1
        if self.direction == "L":
            k1 = 1
        if self.direction == "U":
            k2 = 1
        if self.direction == "D":
            k2 = -1

        for i in range(self.length):
            tempRect = Rect(self.x + k1*i * self.boxSize,
                            self.y + k2*i * self.boxSize, 
                            self.boxSize, self.boxSize)
            self.body.append(tempRect)
        self.head = self.body[0]

        # Setting up network architecture
        # Each "Vision" has 3 distances it tracks: wall, apple and self
        # there are also one-hot encoded direction and one-hot encoded tail direction,
        # each of which have 4 possibilities.
        num_inputs = 12 #@TODO: Add one-hot back in 
        self.network_architecture = [num_inputs]                          # Inputs
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden layers
        self.network_architecture.append(4)                               # 4 outputs, ['u', 'd', 'l', 'r']
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
        self._fitness = (2 ** self.apples + self.apples * 2.1) * 400 - self.total_steps * 2
        self._fitness = max(self._fitness, .1)

    def updateDirection(self, inputs):
        self.network.feed_forward(inputs)
        if self.network.out == 0:
            if self.direction != "D":
                self.direction = "U"
        elif self.network.out == 1:
            if self.direction != "U":
                self.direction = "D"
        elif self.network.out == 2:
            if self.direction != "L":
                self.direction = "R"
        elif self.network.out == 3:
            if self.direction != "R":
                self.direction = "L"

    def addHead(self):
        if self.direction == 'R':
            newHead = Rect(self.x + self.boxSize,
                           self.y, self.boxSize, self.boxSize)
        elif self.direction == 'L':
            newHead = Rect(self.x - self.boxSize,
                           self.y, self.boxSize, self.boxSize)
        elif self.direction == 'D':
            newHead = Rect(self.x,
                           self.y + self.boxSize, self.boxSize, self.boxSize)
        elif self.direction == 'U':
            newHead = Rect(self.x,
                           self.y - self.boxSize, self.boxSize, self.boxSize)
        self.body.insert(0, newHead)
        self.head = self.body[0]
        self.x = self.head.x
        self.y = self.head.y

    def deleteTail(self):
        del self.body[-1]

    def isDead(self):
        for part in self.body[1:]:
            if self.head.colliderect(part):
                return True
        return False

    def isOutOfBounds(self, max_width, max_height):
        if self.head.x > max_width - self.boxSize:
            return True
        elif self.head.x < 0:
            return True
        if self.head.y > max_height - self.boxSize:
            return True
        elif self.head.y < 0:
            return True
        return False

    def reset(self):
        self._fitness = 0
        self.apples = 0
        self.is_alive = True
        self.steps = 0
        self.total_steps = 0
        self.x = 200
        self.y = 200
        self.length = 3

        k1 = 0
        k2 = 0
        if self.direction == "R":
            k1 = -1
        if self.direction == "L":
            k1 = 1
        if self.direction == "U":
            k2 = 1
        if self.direction == "D":
            k2 = -1
        for i in range(self.length):
            tempRect = Rect(self.x + k1*i * self.boxSize,
                            self.y + k2*i * self.boxSize, 
                            self.boxSize, self.boxSize)
            self.body.append(tempRect)
        self.head = self.body[0]

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


class Apple:
    """Apple Object for the snake game.
    Attributes:
    - boxLength
    - x
    - y"""

    def __init__(self, boxLength, x, y, color, board_x, board_y):
        self.boxLength = boxLength
        self.x = x
        self.y = y
        self.color = color
        self.board_x = board_x
        self.board_y = board_y
        self.rect = Rect(self.x, self.y, self.boxLength, self.boxLength)

    def move(self, avoid=None):
        # avoid: a list of (x,y)
        while True:
            random_x = random.randrange(0, (self.board_x - self.boxLength), self.boxLength)
            random_y = random.randrange(0, (self.board_y - self.boxLength), self.boxLength)
            if avoid == None or not (random_x, random_y) in avoid:
                break

        self.rect = Rect(random_x, random_y, self.boxLength, self.boxLength)
        self.x = random_x
        self.y = random_y

class Wall():
    """
    Wall Object
    """

    def __init__(self, size, body):

        self.size = size
        self.pos = []
        self.body = []

    def reset_pos(self, mode=0, pos=None):
        """ Reset Wall, modes:
            - -1: empty (no walls)
            - 0: only four walls
            - 1: random obstacles
            - 2: user-input walls
        """
        self.pos = []

        if mode == 0:
            None

        if mode == 1:
            None

        if mode == 2:
            self.pos = pos

        

    def reset_body(self):
        # reset body of wall using self.pos

        self.body = []
        for (x,y) in self.pos:
            body.append(Rect(x, y, self.size, self.size))
