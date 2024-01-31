import numpy as np
from squareKP_GPU import render


coefficient = 50
WIDTH = 20 * coefficient
HEIGHT = 16 * coefficient


def create_image():
    from PIL import Image

    scene = render(WIDTH, HEIGHT)

    img = Image.fromarray(scene.T)
    img.show()


def interactive_square():
    import pygame


if __name__ == '__main__':
    create_image()
