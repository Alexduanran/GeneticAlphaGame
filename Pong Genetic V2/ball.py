import config
import pygame

class Ball:
	
    def __init__(self, x = 50, y = 50, xspeed = 5, yspeed = 5):
	    self.x = x
	    self.y = y
	    self.xlast = x-xspeed
	    self.ylast = y-yspeed
	    self.xspeed = xspeed
	    self.yspeed = yspeed
	    self.alive = True
	    self.distance_travelled = 0
	
    #Update position based on speed 
    def update(self, paddle):
        self.xlast = self.x
        self.ylast = self.y
        
        self.x += self.xspeed
        self.y += self.yspeed

        self.distance_travelled += abs(self.xspeed)
        
        #Accounts for bouncing off walls and paddle
        if self.x<0:
            self.x=0
            self.xspeed *= -1
        elif self.x>config.SIZE[0]-15:
            self.x=config.SIZE[0]-15
            self.xspeed *= -1
        elif self.y<35:
            self.y=35
            self.yspeed *= -1
        elif self.x>paddle.x and self.x<paddle.x+100 and self.ylast<config.SIZE[1]-35 and self.y>=config.SIZE[1]-35:
            self.yspeed *= -1
            paddle.hit += 1
            # paddle.distance_travelled = 0
            self.distance_to_ball = 0
        elif self.y>config.SIZE[1]:
            self.yspeed *= -1
            paddle.ball_travelled = self.distance_travelled
            paddle.alive = False
            paddle.distance_to_ball = abs(self.x - paddle.x)
            # paddle.score -= 1000
            # paddle.score -= round(abs((paddle.x+50)-self.x)/100,2)
            
			
	#Draw ball to screen	   
    def draw(self, screen):
	    pygame.draw.circle(screen, config.WHITE,[self.x,self.y], 15)