import time
from numba import cuda
from vector_cuda_exctensions import *


@cuda.jit
def ray_plane_intersection(u, r, p, p1, p2):
    normal = cross(p1, p2)

    if abs(dot(r, normal)) < 1e-9:
        return -1

    pu = add(u, mul(p, -1))
    proection_nPU = dot(normal, pu) / (norm(normal))
    proection_nR = dot(normal, r) / (norm(normal))

    if proection_nPU * proection_nR < -(1e-9):
        return abs(proection_nPU / proection_nR)
    return -1


@cuda.jit
def get_pixel(image, x, p1, p2):
    '''Пиксель, соответствующий x в изображении, натянутом на p1 и p2, либо -1'''
    w = len(image)
    h = len(image[0])

    p1_Length = norm(p1)
    p2_Length = norm(p2)

    proectionP1 = dot(x, p1) / p1_Length
    proectionP2 = dot(x, p2) / p2_Length

    ind1 = int((proectionP1 / p1_Length) * w)
    ind2 = int((proectionP2 / p2_Length) * h)

    if 1e-9 <= proectionP1 < p1_Length and 1e-9 <= proectionP2 < p2_Length and abs(get_mixed_product(x, p1, p2)) < 1e-9:
        return image[ind1][ind2]

    return -1


@cuda.jit
def AddLight(p, p1, p2, light, pixelNow, camera, r):
    Olp = add(light, mul(p, -1))
    normal = cross(p1, p2)

    proection_nOLP = dot(normal, Olp) / (norm(normal))
    proection_nCam = dot(normal, add(camera, mul(p, -1))) / (norm(normal))

    # Если камера и свет находятся по разные стороны от rect, то значит сторона в тени(стр 10)
    # тк в таком случае освещена задняя сторона, которую мы не видим
    if proection_nOLP * proection_nCam < 0:
        return pixelNow * 0.1
    else:
        Cos = abs(dot(Olp, normal) / (norm(normal) * norm(Olp)))
        return pixelNow * 0.1 + pixelNow * 0.9 * Cos
    return pixelNow


@cuda.jit
def trace_ray_light(camera, ray, rectangles, light, im_0, im_1, images_link):
    '''Цвет пикселя, в который врезается луч, либо 0, если он уходит в никуда'''
    result = 0
    minT = 100000000000
    for i in range(len(rectangles)):  # перебор всех прямоугольников rect
        rect = rectangles[i]
        p = rect[0]
        p1 = rect[1]
        p2 = rect[2]
        im_link = images_link[i]
        image = im_0 if im_link == 0 else im_1
        tNow = ray_plane_intersection(camera, ray, p, p1, p2)
        if tNow > 1e-9 and tNow < minT:
            pixelNow = get_pixel(image, add(camera, add(mul(ray, tNow), mul(p, -1))), p1, p2)
            if pixelNow != -1:
                minT = tNow

                result = AddLight(p, p1, p2, light, pixelNow, camera, mul(ray, tNow))
    return result


@cuda.jit
def reflect(x, n):
    '''Вернуть отражение вектора x относительно вектора n'''
    proection_nX = dot(n, x) / (norm(n))
    # Строю диагональ ромба
    diagonal = mul(mul(mul(n, 2), proection_nX), 1 / norm(n))
    return add(diagonal, mul(x, -1))  # x - это верный ответ, только если x и n коллинеарны


@cuda.jit
def trace_ray_full(camera, ray, rectangles, light, im_0, im_1, reflections, images_link):
    '''Цвет пикселя, в который врезается луч, либо 0, если он уходит в никуда'''
    result = 0
    minT = 10000000000
    for i in range(len(rectangles)):  # перебор всех прямоугольников rect
        rect = rectangles[i]
        p = rect[0]
        p1 = rect[1]
        p2 = rect[2]
        image_link = images_link[i]
        image = im_0 if image_link == 0 else im_1
        refl = reflections[i]
        # здесь мог бы быть ваш код
        tNow = ray_plane_intersection(camera, ray, p, p1, p2)
        if tNow > 1e-9 and tNow < minT:
            pixelNow = get_pixel(image, add(camera, add(mul(ray, tNow), mul(p, -1))), p1, p2)
            if pixelNow != -1:
                minT = tNow
                if not (refl):
                    result = AddLight(p, p1, p2, light, pixelNow, camera, mul(ray, tNow))
                else:
                    newRay = reflect(mul(ray, -tNow), cross(p1, p2))
                    reflectPix = trace_ray_light(add(camera, mul(ray, tNow)), newRay, rectangles, light, im_0, im_1, images_link)

                    ref_coef = 0.7
                    c = pixelNow * (1 - ref_coef) + ref_coef * reflectPix
                    result = AddLight(p, p1, p2, light, c, camera, mul(ray, tNow))

    return result


# ГЕНЕРАЦИЯ
import numpy as np, imageio.v2 as imageio


@cuda.jit
def generate_light_scene(camera, q, q1, q2, size, rectangles, light, result, im_0, im_1, reflections, images_link):
    w, h = size
    x, y = cuda.grid(2)
    if x < result.shape[0] and y < result.shape[1]:
        k1 = ((x + 0.5) / w)
        k2 = ((y + 0.5) / h)
        pixel = add(q, add(mul(q1, k1), mul(q2, k2)))
        ray = add(pixel, mul(camera, -1))
        brightness = trace_ray_full(camera, ray, rectangles, light, im_0, im_1, reflections, images_link)
        # result[x][y][0] = brightness
        # result[x][y][1] = brightness
        # result[x][y][2] = brightness
        result[x][y] = brightness


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

tpb = (DEVICE.WARP_SIZE // 2, DEVICE.WARP_SIZE // 2)  # blocksize или количество потоков на блок
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
