import pygame
import sys
import random
import gym
from gym import spaces
import numpy as np

class GeometryDashEnv(gym.Env):
    def __init__(self):
        super(GeometryDashEnv, self).__init__()
        pygame.init()

        # Screen configuration
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Geometry Dash")

        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        # Player
        self.player_width = 50
        self.player_height = 50
        self.player_x = self.width // 4
        self.player_y = self.height // 2 - self.player_height // 2
        self.player_speed = 5

        # Gravity and jumps
        self.gravity = 1
        self.jump_force = -15
        self.fall_speed = 0

        self.air_jumps = 0
        self.max_air_jumps = 2

        # List of obstacles
        self.obstacles = []

        # Time to generate new obstacles
        self.obstacle_timer = 0
        self.obstacle_frequency = 50 

        # Define action space and observation space
        self.action_space = spaces.Discrete(2)  # actions: jump or stay still
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

        # Initialize Pygame clock
        self.clock = pygame.time.Clock()

        self.reward = 0
        self.trial = 0

    def step(self, action):
        done = False

        # Move the player
        if action == 1 and (self.player_y == self.height - self.player_height or self.air_jumps < self.max_air_jumps):
            self.fall_speed = self.jump_force
            if self.player_y != self.height - self.player_height:
                self.air_jumps += 1
                self.reward -= 2
        else:
            self.fall_speed += self.gravity
            self.reward += 1

        self.player_y += self.fall_speed

        # Limit player position to the ground
        if self.player_y > self.height - self.player_height:
            self.player_y = self.height - self.player_height
            self.fall_speed = 0
            self.air_jumps = 0

        # Generate new random obstacles
        self.obstacle_timer += 1
        if self.obstacle_timer == self.obstacle_frequency:
            obstacle_height = random.randint(50, 60)
            self.obstacles.append([self.width, self.height - obstacle_height, 30, obstacle_height])
            self.obstacle_timer = 0

        # Move obstacles
        for obstacle in self.obstacles:
            obstacle[0] -= 5

        # Remove off-screen obstacles
        self.obstacles = [obstacle for obstacle in self.obstacles if obstacle[0] + obstacle[2] > 0]

        # Collisions with obstacles
        for obstacle in self.obstacles:
            if (
                self.player_x < obstacle[0] + obstacle[2]
                and self.player_x + self.player_width > obstacle[0]
                and self.player_y < obstacle[1] + obstacle[3]
                and self.player_y + self.player_height > obstacle[1]
            ):
                if self.player_y < obstacle[1] + obstacle[3] and self.player_y + self.player_height > obstacle[1] + obstacle[3] - 20:
                    done = True
                    self.reward -= 0
                    return self.get_frame(), self.reward, done

        # Update screen
        self.screen.fill(self.white)
        pygame.draw.rect(self.screen, self.black, [self.player_x, self.player_y, self.player_width, self.player_height])
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.black, obstacle)

        font = pygame.font.Font(None, 36)
        reward_text = font.render(f"Reward: {self.reward}", True, self.black)
        trial_text = font.render(f"Trial: {self.trial}", True, self.black)
        self.screen.blit(reward_text, (10, 10))
        self.screen.blit(trial_text, (10, 40))

        pygame.display.flip()
        self.clock.tick(100)
        print(self.reward)
        return self.get_frame(), self.reward, done

    def reset(self): 
        # Reset the environment
        self.player_y = self.height // 2 - self.player_height // 2
        self.fall_speed = 0
        self.reward = 0
        self.obstacles = []
        self.air_jumps = 0 
        self.trial += 1
        return self.get_frame()

    def get_frame(self):
        # Capture the screen as observation 
        data = pygame.surfarray.array3d(self.screen)
        return np.transpose(data, (1, 0, 2))

    def close(self):
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    env = GeometryDashEnv()
    total_reward = 0
    done = False

    while not done:
        action = 0
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        action = 1
        obs, total_reward, done = env.step(action)

    print(f"Game over. Total reward: {total_reward}")
    env.close()
