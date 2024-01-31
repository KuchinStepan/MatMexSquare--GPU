import time
import numpy as np
from scene_render import SceneRender


coefficient = 50
WIDTH = 20 * coefficient
HEIGHT = 16 * coefficient
RENDER = SceneRender(WIDTH, HEIGHT)


def create_image():
    from PIL import Image

    scene = RENDER.render(image_render=True)
    img = Image.fromarray(scene.T)
    img.show()


def interactive_square():
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

    while running:
        scene = RENDER.render(False)
        pygame.surfarray.blit_array(screen, scene)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_d]:
                RENDER.move_horizontal(1)
            if keys[pygame.K_a]:
                RENDER.move_horizontal(-1)
            if keys[pygame.K_w]:
                RENDER.move_straight(1)
            if keys[pygame.K_s]:
                RENDER.move_straight(-1)
            if keys[pygame.K_SPACE]:
                RENDER.move_vertical(1)
            if keys[pygame.K_z]:
                RENDER.move_vertical(-1)
        i += 1
        pygame.display.flip()
        clock.tick(15)


if __name__ == '__main__':
    # create_image()
    interactive_square()
