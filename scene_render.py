from numba import cuda
from render_functions_gpu import *
from scene_rectangles import RectangleContainer
import numpy as np
import imageio.v2 as imageio


class SceneRender:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.is_for_image = False

        self.DEVICE = cuda.get_current_device()
        self.tpb = (self.DEVICE.WARP_SIZE // 2, self.DEVICE.WARP_SIZE // 2)
        self.bpg = (int(np.ceil(width / self.tpb[0])), int(np.ceil(height / self.tpb[1])))

        self._init_scene_gpu()

    def _init_scene_gpu(self):
        wall_list = imageio.imread('wall.png')[:, :, 0].tolist()
        logo_list = imageio.imread('logo.png')[:, :, 0].T.tolist()
        self.q, self.q1, self.q2 = np.array([-50, 50, 30]), np.array([100, -60, 0]), np.array([0, 0, -60])
        light = np.array([-40, -20, 150])
        rects, images_link, reflections = RectangleContainer().get_rects_images_reflections()
        self.camera = np.array([-30, -30, 0])

        result = [[[0, 0, 0]] * self.height for _ in range(self.width)]

        self.d_q = cuda.to_device(self.q)
        self.d_q1 = cuda.to_device(self.q1)
        self.d_q2 = cuda.to_device(self.q2)
        self.d_size = cuda.to_device([self.width, self.height])
        self.d_rects = cuda.to_device(rects)
        self.d_light = cuda.to_device(light)
        self.d_reflections = cuda.to_device(reflections)
        self.d_im_0 = cuda.to_device(wall_list)
        self.d_im_1 = cuda.to_device(logo_list)
        self.d_images_link = cuda.to_device(images_link)
        self.d_camera = cuda.to_device(self.camera)

        self.d_result_rgb = cuda.to_device(result)
        self.d_result_image = None

    def _set_for_image_generation(self):
        result = [[0] * self.height for _ in range(self.width)]
        self.d_result_image = cuda.to_device(result)

    def _set_for_pygame_generation(self):
        result = [[[0, 0, 0]] * self.height for _ in range(self.width)]
        self.d_result_rgb = cuda.to_device(result)

    def _switch_mode(self, is_image_render):
        if self.is_for_image == is_image_render:
            return
        if self.is_for_image and not is_image_render:
            self._set_for_pygame_generation()
        else:
            self._set_for_image_generation()
        self.is_for_image = not self.is_for_image

    def get_view_direction(self):
        return self.q + (self.q1 + self.q2) / 2 - self.camera

    def move_horizontal(self, pixel):
        direction = self.get_view_direction()
        norm_dir = direction / np.linalg.norm(direction)

        move_vector = np.array([norm_dir[1], -norm_dir[0], 0]) * pixel

        self.camera = self.camera + move_vector
        self.q = self.q + move_vector
        self.d_q = cuda.to_device(self.q)
        self.d_camera = cuda.to_device(self.camera)

    def move_vertical(self, pixel):
        move_vector = np.array([0, 0, pixel])

        self.camera = self.camera + move_vector
        self.q = self.q + move_vector
        self.d_q = cuda.to_device(self.q)
        self.d_camera = cuda.to_device(self.camera)

    def move_straight(self, pixel):
        direction = self.get_view_direction()
        move_vector = np.array([direction[0], direction[1], 0]) / np.linalg.norm(direction) * pixel

        self.camera = self.camera + move_vector
        self.q = self.q + move_vector
        self.d_q = cuda.to_device(self.q)
        self.d_camera = cuda.to_device(self.camera)

    def render(self, image_render=False):
        self._switch_mode(image_render)

        if image_render:
            generate_light_scene_brightness[self.bpg, self.tpb](
                self.d_camera, self.d_light,
                self.d_q, self.d_q1, self.d_q2,
                self.d_size,
                self.d_rects, self.d_images_link, self.d_reflections,
                self.d_im_0, self.d_im_1,
                self.d_result_image
            )
            res = self.d_result_image.copy_to_host()
        else:
            generate_light_scene_rgb[self.bpg, self.tpb](
                self.d_camera, self.d_light,
                self.d_q, self.d_q1, self.d_q2,
                self.d_size,
                self.d_rects, self.d_images_link, self.d_reflections,
                self.d_im_0, self.d_im_1,
                self.d_result_rgb
            )
            res = self.d_result_rgb.copy_to_host()

        return np.array(res)
