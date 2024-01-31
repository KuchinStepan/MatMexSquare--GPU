import time
from numba import cuda


# ГЕНЕРАЦИЯ
import numpy as np, imageio.v2 as imageio

from scene_render import *


DEVICE = cuda.get_current_device()


koeff = 50
WIDTH = 20 * koeff
HEIGHT = 16 * koeff


wall = imageio.imread('wall.png')[:, :, 0]
logo = imageio.imread('logo.png')[:, :, 0].T

q, q1, q2 = np.array([-50, 50, 30]), np.array([100, -60, 0]), np.array([0, 0, -60])
light = np.array([-40, -20, 150])
logo_list = logo.tolist()
wall_list = wall.tolist()

rects = [
    [[-50, 50, -40], [100, 100, 0], [0, 0, 100]],
    [[-50, 50, -40], [100, 100, 0], [100, -100, 0]],
    [[50, -50, -40], [100, 100, 0], [0, 0, 100]],
    [[50, 150, -40], [100, -100, 0], [0, 0, 100]],
    [[-50, 50, 40], [100, 100, 0], [100, -100, 0]],
    [[10, 100, 30], [20, -60, 10], [-10, -10, -40]],
    [[30, 40, 40], [75, 21, -24], [-10, -10, -40]],
    [[0, 90, -10], [20, -60, 10], [75, 21, -24]]
]

images_link = [0, 0, 0, 0, 0, 1, 1, 1]
images = [wall_list, logo_list]
reflections = [False, False, False, False, False, True, True, True]
# reflections = list(map(lambda x: not x, reflections))

# result = [[[0, 0, 0]] * HEIGHT for _ in range(WIDTH)]
result = [[0] * HEIGHT for _ in range(WIDTH)]

tpb = (DEVICE.WARP_SIZE // 2, DEVICE.WARP_SIZE // 2)  # block_size или количество потоков на блок
bpg = (int(np.ceil(WIDTH / tpb[0])), int(np.ceil(HEIGHT / tpb[1])))


d_q = cuda.to_device(q)
d_q1 = cuda.to_device(q1)
d_q2 = cuda.to_device(q2)
# d_w = cuda.to_device(w)
# d_h = cuda.to_device(h)
d_size = cuda.to_device([WIDTH, HEIGHT])
d_rects = cuda.to_device(rects)
d_light = cuda.to_device(light)
d_result = cuda.to_device(result)
d_reflections = cuda.to_device(reflections)
d_im_0 = cuda.to_device(wall_list)
d_im_1 = cuda.to_device(logo_list)
d_images_link = cuda.to_device(images_link)


def render(w, h, cam=np.array([-30, -30, 0])):
    d_camera = cuda.to_device(cam)
    generate_light_scene[bpg, tpb](d_camera, d_q, d_q1, d_q2, d_size, d_rects, d_light, d_result, d_im_0, d_im_1, d_reflections, d_images_link)
    res = d_result.copy_to_host()

    return np.array(res)


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
