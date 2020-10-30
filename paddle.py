import math
import config
import pygame
import numpy as np

class Paddle:
    def __init__(self, x=400, y=config.SIZE[1]-20, xspeed=0, coefs=0):
        self.x = x
        self.y = y
        self.xspeed = xspeed
        self.coefs = self.generateCoefs(config.LAYERS) if coefs == 0 else coefs
        self.alive = True
        self.output = 2
        self.winner = False
        self.champion = False

        # Fitness
        self.hit = 0
        self.distance_tranvelled = 0
        self.distance_to_ball = 0
        self.moving_duration = 0

    # Creates random coefficients for the neural network 
    def generateCoefs(self, layer_structure):
        coefs = []
        for i in range(len(layer_structure)-1):
            coefs.append(np.random.rand(layer_structure[i], layer_structure[i+1])*2-1)
        return coefs

    #Predicts the output for a given input given an array of coefficients
    def calculateOutput(self, input, layer_structure, g="identity"):
        #The values of the neurons for each layer will be stores in "layers", so here the input layer is added to start
        #(Stuff is transposed since we need columns for matrix multiplication)
        layers = [np.transpose(input)]
        #The current layer will be affected by the previous layer, so here we define the starting previousLayer as the input 
        previousLayer = np.transpose(input)
        
        reduced_layer_structure = layer_structure[1:]
        #Loops through the all the layers except the input
        for k in range(len(reduced_layer_structure)):
            #creates an empty array of the correct size
            currentLayer = np.empty((reduced_layer_structure[k],1))
            #The resulting layer is a matrix multiplication of the previousLayer and the coefficients
            result = np.matmul(np.transpose(self.coefs[k]),previousLayer)
            #The value of each neuron is then put through a function g()
            for i in range(len(currentLayer)):
                if g == "identity":
                    currentLayer[i] = result[i]
                elif g == "relu":
                    currentLayer[i] = max(0, result[i])
                elif g == "tanh":
                    currentLayer[i] = math.tanh(result[i])
                elif g == "logistic":
                    try:
                        currentLayer[i] = 1 / (1 + math.exp(-1*result[i]))
                    except OverflowError:
                        currentLayer[i] = 0
            #The current layer is then added to the layers list, and the previousLayer variable is updated
            layers.append(currentLayer)
            previousLayer = currentLayer.copy()
	
        #Returns the index of the highest value neuron in the output layer (aka layers[-1])
        #E.g. if the 7th neuron has the highest value, returns 7

        self.output = layers[-1].tolist().index(max(layers[-1].tolist()))

    # Fitness function
    def calculateFitness(self):
        return 200 * self.hit - 0.5 * self.distance_to_ball 

    # Reset score, speed and position
    def reset(self):
	    self.x = 400
	    self.xlast = 400
	    self.xspeed = 0
	    self.alive = True
	    self.hit = 0
	    self.distance_tranvelled = 0
	    self.distance_to_ball = 0

    # Update the paddle
    def update(self, ball=None):
        if ball:
            self.x = ball.x - 50
        if self.output == 0:
            self.xspeed = -5
            self.moving_duration += 1
        elif self.output == 1:
            self.xspeed = 5
            self.moving_duration += 1
        elif self.output == 2:
            self.xspeed = 0
            self.moving_duration = 0
        self.x += self.xspeed
        self.distance_tranvelled += abs(self.xspeed)

        if self.x < 0:
            self.x = 0
        elif self.x > config.SIZE[0]-100:
            self.x = config.SIZE[0]-100

    # Draw the paddle
    def draw(self, screen):
	    if not self.alive:
		    return
	    if not self.winner and not self.champion:
		    pygame.draw.rect(screen,config.BLACK,[self.x,self.y,100,20])
		    pygame.draw.rect(screen,config.RED,[self.x+2,self.y+2,100-4,20-4])
	    if self.champion:
		    pygame.draw.rect(screen,config.BLACK,[self.x,self.y,100,20])
		    pygame.draw.rect(screen,config.GREEN,[self.x+2,self.y+2,100-4,20-4])
	    elif self.winner:
		    pygame.draw.rect(screen,config.BLACK,[self.x,self.y,100,20])
		    pygame.draw.rect(screen,config.BLUE,[self.x+2,self.y+2,100-4,20-4])