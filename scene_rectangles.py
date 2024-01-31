IMAGE_TO_LINK = {
    'wall.png': 0,
    'logo.png': 1
}


class Rectangle:
    def __init__(self, p, p1, p2, image, reflection):
        self.p = p
        self.p1 = p1
        self.p2 = p2
        self.image = image
        self.reflection = reflection


class RectangleContainer:
    def __init__(self):
        self.rectangles: list[Rectangle] = [
            Rectangle([-50, 50, -40], [100, 100, 0], [0, 0, 100], 'wall.png', False),
            Rectangle([-50, 50, -40], [100, 100, 0], [100, -100, 0], 'wall.png', False),
            Rectangle([50, -50, -40], [100, 100, 0], [0, 0, 100], 'wall.png', False),
            Rectangle([50, 150, -40], [100, -100, 0], [0, 0, 100], 'wall.png', False),
            Rectangle([-50, 50, 40], [100, 100, 0], [100, -100, 0], 'wall.png', False),
            Rectangle([10, 100, 30], [20, -60, 10], [-10, -10, -40], 'logo.png', True),
            Rectangle([30, 40, 40], [75, 21, -24], [-10, -10, -40], 'logo.png', True),
            Rectangle([0, 90, -10], [20, -60, 10], [75, 21, -24], 'logo.png', True)
        ]

    def get_rects_images_reflections(self):
        rects = [[rect.p, rect.p1, rect.p2] for rect in self.rectangles]
        images = [IMAGE_TO_LINK[rect.image] for rect in self.rectangles]
        reflections = [rect.reflection for rect in self.rectangles]

        return rects, images, reflections
