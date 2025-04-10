import numpy as np

class ComponentToPinMapper:
    def __init__(self):
        pass

    def find_nearest_pins(self, box, pins):
        x1, y1, x2, y2 = box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        distances = sorted(pins, key=lambda p: np.linalg.norm(np.array(center) - np.array(p)))
        return distances[:2]
