import time
import pygame
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


def control_movements(keys, step, angle):
    if keys[pygame.K_d]:
        RENDER.move_horizontal(step)
    if keys[pygame.K_a]:
        RENDER.move_horizontal(-step)
    if keys[pygame.K_w]:
        RENDER.move_straight(step)
    if keys[pygame.K_s]:
        RENDER.move_straight(-step)
    if keys[pygame.K_SPACE]:
        RENDER.move_vertical(step)
    if keys[pygame.K_z]:
        RENDER.move_vertical(-step)

    if keys[pygame.K_f]:
        RENDER.rotate_horizontal(angle)
    if keys[pygame.K_g]:
        RENDER.rotate_horizontal(-angle)


def interactive_square():
    running = True
    t0 = time.time()
    RENDER.render(False)
    print(time.time() - t0)

    pygame.init()

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
            control_movements(keys, 1, 0.01)

        pygame.display.flip()
        clock.tick(15)


if __name__ == '__main__':
    # create_image()
    interactive_square()
