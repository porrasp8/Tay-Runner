import pygame
import sys
import random
import gym
from gym import spaces
import numpy as np

class GeometryDashEnv(gym.Env):
    def __init__(self):
        super(GeometryDashEnv, self).__init__()

        # Inicializar Pygame
        pygame.init()

        # Configuración de la pantalla
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Geometry Dash")

        # Colores
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        # Jugador
        self.player_width = 50
        self.player_height = 50
        self.player_x = self.width // 4
        self.player_y = self.height // 2 - self.player_height // 2
        self.player_speed = 5

        # Gravedad
        self.gravity = 1
        self.jump_force = -15
        self.fall_speed = 0

        # Contador de saltos en el aire y límite de saltos
        self.air_jumps = 0
        self.max_air_jumps = 2

        # Lista de obstáculos
        self.obstacles = []

        # Tiempo para generar nuevos obstáculos
        self.obstacle_timer = 0
        self.obstacle_frequency = 50  # Ajusta la frecuencia según sea necesario

        # Definir el espacio de acciones y el espacio de observaciones
        self.action_space = spaces.Discrete(2)  # acciones: saltar o estar quieto
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)

        # Inicializar Pygame clock
        self.clock = pygame.time.Clock()

        self.reward = 0

    def step(self, action):
        done = False

        # Mover el jugador
        if action == 1 and (self.player_y == self.height - self.player_height or self.air_jumps < self.max_air_jumps):
            self.fall_speed = self.jump_force
            if self.player_y != self.height - self.player_height:
                self.air_jumps += 1
        else:
            self.fall_speed += self.gravity

        self.player_y += self.fall_speed

        # Limitar la posición del jugador al suelo
        if self.player_y > self.height - self.player_height:
            self.player_y = self.height - self.player_height
            self.fall_speed = 0
            self.air_jumps = 0

        # Generar nuevos obstáculos aleatorios
        self.obstacle_timer += 1
        if self.obstacle_timer == self.obstacle_frequency:
            obstacle_height = random.randint(50, 60)
            self.obstacles.append([self.width, self.height - obstacle_height, 30, obstacle_height])
            self.obstacle_timer = 0

        # Mover los obstáculos
        for obstacle in self.obstacles:
            obstacle[0] -= 5

        # Eliminar obstáculos fuera de la pantalla
        self.obstacles = [obstacle for obstacle in self.obstacles if obstacle[0] + obstacle[2] > 0]

        # Colisiones con obstáculos
        for obstacle in self.obstacles:
            if (
                self.player_x < obstacle[0] + obstacle[2]
                and self.player_x + self.player_width > obstacle[0]
                and self.player_y < obstacle[1] + obstacle[3]
                and self.player_y + self.player_height > obstacle[1]
            ):
                if self.player_y < obstacle[1] + obstacle[3] and self.player_y + self.player_height > obstacle[1] + obstacle[3] - 20:
                    # Solo muere si toca por debajo
                    done = True
                    #reward = -1  # Penalizar colisión
                    return self.get_frame(), self.reward, done


        # Actualizar pantalla
        self.screen.fill(self.white)
        pygame.draw.rect(self.screen, self.black, [self.player_x, self.player_y, self.player_width, self.player_height])
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.black, obstacle)

        self.reward += 1
        font = pygame.font.Font(None, 36)
        reward_text = font.render(f"Reward: {self.reward}", True, self.black)
        self.screen.blit(reward_text, (10, 10))

        pygame.display.flip()

        # Controlar la velocidad de actualización
        self.clock.tick(100)

        
        return self.get_frame(), self.reward, done

    def reset(self): 
        # Reiniciar el entorno
        self.player_y = self.height // 2 - self.player_height // 2
        self.fall_speed = 0
        self.obstacles = []
        self.air_jumps = 0  # Reiniciar el contador de saltos al reiniciar el juego
        return self.get_frame()


    def get_frame(self):
        # Capturar la pantalla como observación (imagen)
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

    print(f"Juego terminado. Recompensa total: {total_reward}")
    env.close()
