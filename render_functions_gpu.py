from numba import cuda
from vector_cuda_exctensions import *


@cuda.jit
def ray_plane_intersection(u, r, p, p1, p2):
    normal = cross(p1, p2)

    if abs(dot(r, normal)) < 1e-9:
        return -1

    pu = add(u, mul(p, -1))
    projection_n_pu = dot(normal, pu) / (norm(normal))
    projection_n_r = dot(normal, r) / (norm(normal))

    if projection_n_pu * projection_n_r < -1e-9:
        return abs(projection_n_pu / projection_n_r)
    return -1


@cuda.jit
def get_pixel(image, x, p1, p2):
    # Пиксель, соответствующий x в изображении, натянутом на p1 и p2, либо -1
    w = len(image)
    h = len(image[0])

    p1_length = norm(p1)
    p2_length = norm(p2)

    projection_p1 = dot(x, p1) / p1_length
    projection_p2 = dot(x, p2) / p2_length

    ind1 = int((projection_p1 / p1_length) * w)
    ind2 = int((projection_p2 / p2_length) * h)

    if (1e-9 <= projection_p1 < p1_length
            and 1e-9 <= projection_p2 < p2_length
            and abs(get_mixed_product(x, p1, p2)) < 1e-9):
        return image[ind1][ind2]

    return -1


@cuda.jit
def add_light(p, p1, p2, light, pixel_now, camera):
    olp = add(light, mul(p, -1))
    normal = cross(p1, p2)

    projection_n_olp = dot(normal, olp) / (norm(normal))
    projection_n_cam = dot(normal, add(camera, mul(p, -1))) / (norm(normal))

    # Если камера и свет находятся по разные стороны от rect, то значит сторона в тени(стр 10)
    # тк в таком случае освещена задняя сторона, которую мы не видим
    if projection_n_olp * projection_n_cam < 0:
        return pixel_now * 0.1
    else:
        cos = abs(dot(olp, normal) / (norm(normal) * norm(olp)))
        return pixel_now * 0.1 + pixel_now * 0.9 * cos


@cuda.jit
def trace_ray_light(camera, ray, rectangles, light, im_0, im_1, images_link):
    # Цвет пикселя, в который врезается луч, либо 0, если он уходит в никуда
    result = 0
    min_t = 100000000000
    for i in range(len(rectangles)):  # перебор всех прямоугольников rect
        rect = rectangles[i]
        p = rect[0]
        p1 = rect[1]
        p2 = rect[2]
        im_link = images_link[i]
        image = im_0 if im_link == 0 else im_1
        t_now = ray_plane_intersection(camera, ray, p, p1, p2)
        if 1e-9 < t_now < min_t:
            pixel_now = get_pixel(image, add(camera, add(mul(ray, t_now), mul(p, -1))), p1, p2)
            if pixel_now != -1:
                min_t = t_now

                result = add_light(p, p1, p2, light, pixel_now, camera)
    return result


@cuda.jit
def reflect(x, n):
    # Вернуть отражение вектора x относительно вектора n
    projection_n_x = dot(n, x) / (norm(n))
    # Строю диагональ ромба
    diagonal = mul(mul(mul(n, 2), projection_n_x), 1 / norm(n))
    return add(diagonal, mul(x, -1))  # x - это верный ответ, только если x и n коллинеарны


@cuda.jit
def trace_ray_full(camera, ray, rectangles, light, im_0, im_1, reflections, images_link):
    # Цвет пикселя, в который врезается луч, либо 0, если он уходит в никуда
    result = 0
    min_t = 10000000000
    for i in range(len(rectangles)):  # перебор всех прямоугольников rect
        rect = rectangles[i]
        p = rect[0]
        p1 = rect[1]
        p2 = rect[2]
        image_link = images_link[i]
        image = im_0 if image_link == 0 else im_1
        refl = reflections[i]
        # здесь мог бы быть ваш код
        t_now = ray_plane_intersection(camera, ray, p, p1, p2)
        if 1e-9 < t_now < min_t:
            pixel_now = get_pixel(image, add(camera, add(mul(ray, t_now), mul(p, -1))), p1, p2)
            if pixel_now != -1:
                min_t = t_now
                if not refl:
                    result = add_light(p, p1, p2, light, pixel_now, camera)
                else:
                    new_ray = reflect(mul(ray, -t_now), cross(p1, p2))
                    reflect_pix = trace_ray_light(add(camera, mul(ray, t_now)), new_ray, rectangles,
                                                  light, im_0, im_1, images_link)

                    ref_coef = 0.7
                    c = pixel_now * (1 - ref_coef) + ref_coef * reflect_pix
                    result = add_light(p, p1, p2, light, c, camera)

    return result


@cuda.jit
def generate_light_scene_brightness(camera, light, q, q1, q2, size, rectangles, images_link, reflections, im_0, im_1, result):
    w, h = size
    x, y = cuda.grid(2)
    if x < result.shape[0] and y < result.shape[1]:
        k1 = ((x + 0.5) / w)
        k2 = ((y + 0.5) / h)
        pixel = add(q, add(mul(q1, k1), mul(q2, k2)))
        ray = add(pixel, mul(camera, -1))
        brightness = trace_ray_full(camera, ray, rectangles, light, im_0, im_1, reflections, images_link)
        result[x][y] = brightness


@cuda.jit
def generate_light_scene_rgb(camera, light, q, q1, q2, size, rectangles, images_link, reflections, im_0, im_1, result):
    w, h = size
    x, y = cuda.grid(2)
    if x < result.shape[0] and y < result.shape[1]:
        k1 = ((x + 0.5) / w)
        k2 = ((y + 0.5) / h)
        pixel = add(q, add(mul(q1, k1), mul(q2, k2)))
        ray = add(pixel, mul(camera, -1))
        brightness = trace_ray_full(camera, ray, rectangles, light, im_0, im_1, reflections, images_link)
        result[x][y][0] = brightness
        result[x][y][1] = brightness
        result[x][y][2] = brightness
