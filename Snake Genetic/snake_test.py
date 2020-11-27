import os
from snakeClass import *
from settings import settings
import pygame

RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GRAY = (0, 50, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 51)

class Main():
    def __init__(self):
        
        self.board_size = settings['board_size']
        x = random.randrange(0, (self.board_size[0] - 20), 20)
        y = random.randrange(0, (self.board_size[1] - 20), 20)
        self.snake = self.load_snake('best_snakes', 'snake54615-40', settings)
        self.snake.x, self.snake.y = x, y
        self.apple = Apple(20, 100, 200, RED, self.board_size[0], self.board_size[1])
        self.apple.move()

        # Pygame
        pygame.init()
        screen = pygame.display.set_mode(self.board_size)
        pygame.display.set_caption('snake')

        # The loop will carry on until the user exit the game (e.g. clicks the close button).
        carryOn = True
        # The clock will be used to control how fast the screen updates
        clock = pygame.time.Clock()

        self.num_apples = 0

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
            apples_text = font.render("Apples: %d" % self.num_apples, True, WHITE)
            screen.blit(apples_text, (self.board_size[0] - 150, 10))
            
            #----------------------------------------inputs for neural network--------------------------------------------
            ''' # Version 1
            12 inputs
            '''
            inputs = np.zeros((12, ))
            # Direction of Snake
            if self.snake.direction == "U":
                inputs[0] = 1
            elif self.snake.direction == "R":
                inputs[1] = 1
            elif self.snake.direction == "D":
                inputs[2] = 1
            elif self.snake.direction == "L":
                inputs[3] = 1

            # Apple position (wrt snake head)
            # (0,0) at Top-Left Corner: U: -y; R: +x
            if self.apple.y < self.snake.y:
                # apple north snake
                inputs[4] = 1
            if self.apple.x > self.snake.x:
                # apple east snake
                inputs[5] = 1
            if self.apple.y > self.snake.y:
                # apple south snake
                inputs[6] = 1
            if self.apple.x < self.snake.x:
                # apple west snake
                inputs[7] = 1
            
            # Obstacle (Walls, body) position (wrt snake head)
            body_pos = [(rect.x, rect.y) for rect in self.snake.body]
            if self.snake.direction != "D" and \
            (self.snake.y  <= 0 or (self.snake.x, self.snake.y-20) in body_pos):
                # obstacle at north
                inputs[8] = 1
            if self.snake.direction != "L" and \
            (self.snake.x >= self.board_size[0]-20 or (self.snake.x+20, self.snake.y) in body_pos):
                # obstacle at east
                inputs[9] = 1
            if self.snake.direction != "U" and \
            (self.snake.y >= self.board_size[1]-20 or (self.snake.x, self.snake.y+20) in body_pos):
                # obstacle at south
                inputs[10] = 1
            if self.snake.direction != "R" and \
            (self.snake.x <= 0 or (self.snake.x-20, self.snake.y) in body_pos):
                # obstacle at west
                inputs[11] = 1
            ''
            '''
            Version 2 of inputs
            First 24 inputs are distance_to_wall, distance_to_apple
            and distance_to_body in resepct to 8 directions from the
            grid of snake's head.
            '''
            '''
            inputs = snake.look(self.apple.x, self.apple.y)
            directions = np.zeros((8, ))
            if snake.direction == "U":
                directions[0] = 1
            elif snake.direction == "R":
                directions[1] = 1
            elif snake.direction == "D":
                directions[2] = 1
            elif snake.direction == "L":
                directions[3] = 1
            Apple position (wrt snake head)
            (0,0) at Top-Left Corner: U: -y; R: +x
            if self.apple.y < snake.y:
                # apple north snake
                directions[4] = 1
            if self.apple.x > snake.x:
                # apple east snake
                directions[5] = 1
            if self.apple.y > snake.y:
                # apple south snake
                directions[6] = 1
            if self.apple.x < snake.x:
                # apple west snake
                directions[7] = 1
            inputs = np.concatenate((inputs, directions), axis=0)
            '''
            #----------------------------------------inputs for neural network--------------------------------------------
            self.snake.updateDirection(inputs)
            self.snake.addHead()
            if self.snake.isDead() or self.snake.isOutOfBounds(self.board_size[0], self.board_size[1]):
                self.reset()
            if not (self.snake.head.colliderect(self.apple.rect)):
                self.snake.deleteTail()
                self.snake.steps += 1
                if self.snake.steps > self.board_size[0] / 2 + 20:
                    self.reset()
            else:
                self.apple.move()
                self.num_apples += 1
                self.snake.steps = 0

            for part in self.snake.body:
                pygame.draw.rect(screen, GREEN, part)
                part_small = part.inflate(-3, -3)
                pygame.draw.rect(screen, WHITE, part_small, 3)
            
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
        
            # --- Limit to 60 frames per second
            clock.tick(60)

        #Once we have exited the main program loop we can stop the game engine:
        pygame.quit()

    def reset(self):
        self.snake.length = 3
        self.num_apples = 0
        self.snake.direction = 'R'
        self.snake.steps = 0
        self.snake.x = random.randrange(0, (self.board_size[0] - 20), 20)
        self.snake.y = random.randrange(0, (self.board_size[1] - 20), 20)
        k1 = 0
        k2 = 0
        if self.snake.direction == "R":
            k1 = -1
        if self.snake.direction == "L":
            k1 = 1
        if self.snake.direction == "U":
            k2 = 1
        if self.snake.direction == "D":
            k2 = -1
        self.snake.body = []
        for i in range(self.snake.length):
            tempRect = Rect(self.snake.x + k1*i * self.snake.boxSize,
                            self.snake.y + k2*i * self.snake.boxSize, 
                            self.snake.boxSize, self.snake.boxSize)
            self.snake.body.append(tempRect)
        self.snake.head = self.snake.body[0]
        self.apple.move()
        

    def load_snake(self, population_folder: str, individual_name: str, settings: Optional[Union[Dict[str, Any], str]] = None) -> Snake:
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

        snake = Snake(chromosome=params, 
                    x=200,y=200,length=3,
                    direction='R', boxSize=20, board_size=settings['board_size'],
                    hidden_layer_architecture=settings['hidden_network_architecture'],
                    hidden_activation=settings['hidden_layer_activation'],
                    output_activation=settings['output_layer_activation'],
                    )
        return snake

if __name__ == "__main__":
    main = Main()