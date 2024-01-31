import time

import numpy as np
from scene_render import SceneRender


coefficient = 50
WIDTH = 20 * coefficient
HEIGHT = 16 * coefficient


def create_image():
    from PIL import Image

    render = SceneRender(WIDTH, HEIGHT)

    scene = render.render(image_render=True)

    img = Image.fromarray(scene.T)
    img.show()


def interactive_square():
    import pygame


if __name__ == '__main__':
    create_image()
