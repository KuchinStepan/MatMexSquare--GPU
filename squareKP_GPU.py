import time
import numpy as np

from scene_render import *


koeff = 50
WIDTH = 20 * koeff
HEIGHT = 16 * koeff



camera = np.array([-30, -30, 0])
render(WIDTH, HEIGHT, camera)

t0 = time.time()
render(WIDTH, HEIGHT, camera)
print(time.time() - t0)


if __name__ == '__main__':
    import pygame

    running = True
    i = 0
    t0 = time.time()

    print(time.time() - t0)

    pygame.init()
    pygame.mixer.init()  # для звука
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Game")
    clock = pygame.time.Clock()

    camera = np.array([-30, -30, 0])

    while running:
        scene = render(WIDTH, HEIGHT, camera)
        pygame.surfarray.blit_array(screen, scene)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_w]:
                camera = np.array([camera[0], camera[1], camera[2] + 2])
            if keys[pygame.K_s]:
                camera = np.array([camera[0], camera[1], camera[2] - 2])
            if keys[pygame.K_a]:
                camera = np.array([camera[0] + 2, camera[1], camera[2]])
            if keys[pygame.K_d]:
                camera = np.array([camera[0] - 2, camera[1], camera[2]])
            if keys[pygame.K_z]:
                camera = np.array([camera[0], camera[1] + 2, camera[2]])
            if keys[pygame.K_x]:
                camera = np.array([camera[0], camera[1] - 2, camera[2]])
        i += 1
        pygame.display.flip()
        clock.tick(15)
