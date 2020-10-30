import game, paddle, ball, population, config
import pygame
import numpy as np

pygame.init()
screen = pygame.display.set_mode(config.SIZE)
pygame.display.set_caption("pong")

# The loop will carry on until the user exit the game (e.g. clicks the close button).
carryOn = True
 
# The clock will be used to control how fast the screen updates
clock = pygame.time.Clock()
 
# -------- Main Program Loop -----------
training_paddle = paddle.Paddle(y=0)
population_ = population.Population(100)
winner = None
generation = 1

# Best paddle of all generations
champion = None
champion_fitness = -1 * float('inf')

while carryOn:
    # --- Main event loop
    for event in pygame.event.get(): # User did something
        if event.type == pygame.QUIT: # If user clicked close
            carryOn = False # Flag that we are done so we exit this loop
 
    # --- Game logic should go here
    still_alive = 0
    high_score = -9e99
    high_score_index = -1
 
    # --- Drawing code should go here
    # First, clear the screen to black. 
    # Draw and update the training paddle
    screen.fill(config.BLACK)
    training_paddle.draw(screen)
    training_paddle.update(population_.balls[-1])
    population_.balls[-1].update(training_paddle)

    # For each paddle in the population
    for i, paddle in enumerate(population_.paddles):
        balls = population_.balls
        distance = ((balls[i].y - paddle.y) ** 2 + (balls[i].x - paddle.x) ** 2) ** 0.5
        inputs = np.array([paddle.x, distance, balls[i].xspeed])
        paddle.calculateOutput(inputs, config.LAYERS)

        # Update the paddel if it is still alive
        if paddle.alive:
            still_alive += 1
            paddle.update()
            balls[i].update(paddle)


        # Draw every paddle except the winning one
        if paddle.alive and paddle != winner and paddle != champion:
            paddle.winner = False
            paddle.champion = False
            paddle.draw(screen)
            balls[i].draw(screen)

    winner_index = -1
    # If the generation has died out
    # Select winner and champion
    if still_alive == 0:
        generation += 1
        population_.calculateFitness()
        winner_index = population_.fitness.index(max(population_.fitness))
        winner = population_.paddles[winner_index]
        winner.winner = True
        if population_.fitness[winner_index] > champion_fitness:
            champion_fitness = population_.fitness[winner_index]
            champion = winner
            champion.champion = True
        population_.new_population(winner, np.random.randint(0, 800), champion)
        print(generation)

    # Draw the winning and champion paddle last
    if champion:
        champion.draw(screen)
    if winner and winner != champion:
        # print(winner.winner, winner.champion)
        winner.draw(screen)
 
    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()
     
    # --- Limit to 60 frames per second
    clock.tick(60)
 
#Once we have exited the main program loop we can stop the game engine:
pygame.quit()