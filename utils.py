import numpy as np

def proyection(point: np.array):
    return point[1:]


def map_to_sphere(point: np.array) -> np.array:
    first_cordinate = 1-sum(point)
    return np.append([first_cordinate], point)