""" Compress an image using Voronoi cells """
import numpy
import cv2
import pdb
import os
from collections import namedtuple, OrderedDict
from itertools import product
from sortedcontainers import SortedSet
from utils import print_progress

script_folder, _ = os.path.split(os.path.abspath(__file__))
raw_images_folder = os.path.join(script_folder, 'images', 'raw')
compressed_images_folder = os.path.join(script_folder, 'images', 'compressed')

Position = namedtuple('Position', ['i', 'j'])


# region Compression

def compress(image_file):

    cells = image_cell_grid(image_file)


# endregion


# region Voronoi cells

class Cell:

    def __init__(self, colour, position=None, weight=1, neighbours=None):
        self.position = position
        self._colour = colour
        self.weight = weight
        self.neighbours = SortedSet(neighbours, key=self.colour_difference)

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, colour):
        self._colour = colour
        self.neighbours = SortedDict(self.neighbours)

    def colour_difference(self, cell):
        return sum((n - m) ** 2 for n, m in zip(self.colour, cell.colour)) ** 0.5

    def most_similar_neighbour(self):
        return self.neighbours.peekitem(0)


def inbounds(row, col, height, width):
    return 0 <= row < height and 0 <= col < width


def grid_neighbour_indices(row, col, height, width):
    offsets = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    return [(row + row_offset, col + col_offset) for row_offset, col_offset in offsets if
            inbounds(row + row_offset, col + col_offset, height, width)]


def grid_neighbours(grid, row, col):
    indices = grid_neighbour_indices(row, col, len(grid), len(grid[0]))
    return [grid[i][j] for i, j in indices]


def colour_at_pixel(image_data, i, j):
    _, _, colour_dimension = image_data.shape
    return [image_data.item((i, j, k)) for k in range(colour_dimension)]


def image_cell_grid(image_file):
    print(f'Loading cell grid for {image_file}')
    image_data = cv2.imread(image_file)
    height, width, colour_dimension = image_data.shape
    cells = [[Cell(colour=colour_at_pixel(image_data, i, j), position=Position(i, j))
              for j in range(width)] for i in range(height)]
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            print_progress(int(i * width + j), height * width, bar_length=70)
            for neighbour in grid_neighbours(cells, i, j):
                cell.neighbours.add(neighbour)
    return cells

# endregion


if __name__ == '__main__':

    raw_image = os.path.join(raw_images_folder, input('Enter an image name: ') or 'bliss.png')
    while not os.path.exists(raw_image):
        raw_image = os.path.join(raw_images_folder, input('That file could not be found, try another: '))

    compress(raw_image)  # TODO: Set up return image and save to compressed_images_folder
