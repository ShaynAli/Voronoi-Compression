""" Compress an image using Voronoi cells """
import numpy as np
import cv2
import pdb
import os
from collections import namedtuple, OrderedDict
from itertools import product
from sortedcontainers import SortedDict, SortedSet
import utils

script_folder, _ = os.path.split(os.path.abspath(__file__))
raw_images_folder = os.path.join(script_folder, 'images', 'raw')
compressed_images_folder = os.path.join(script_folder, 'images', 'compressed')


# region Compression

def compress(raw_image_file, compressed_image_file, n_cells=100):

    cell_grid = image_cell_grid(raw_image_file)

    cells = chain(*cell_grid)

    for _ in range(n_merges):
        Cell.merge_cells(*Cell.least_difference_edge())
        pdb.set_trace()

    compressed_image = voronoi_fill(cells)

    cv2.imwrite(compressed_image_file, compressed_image)

# endregion


# region Voronoi cells

class Cell:

    edge_set = SortedSet(key=lambda cells: Cell.compare_colours(*cells))

    @staticmethod
    def add_edge(first_cell, second_cell):
        first_cell.neighbours.add(second_cell)
        second_cell.neighbours.add(first_cell)
        edge_set.add(frozenset([first_cell, second_cell]))

    @staticmethod
    def remove_edge(first_cell, second_cell):
        first_cell.neighbours.remove(second_cell)
        second_cell.neighbours.remove(first_cell)
        edge_set.remove(frozenset([first_cell, second_cell]))

    @staticmethod
    def least_difference_edge():
        return edge_set[0]

    @staticmethod
    def compare_colours(first_cell, second_cell):
        return np.linalg.norm(first_cell.colour - second_cell.colour)

    @staticmethod
    def merge_cells(first_cell, second_cell):

        weights = [first_cell.weight, second_cell.weight]

        colour = utils.weighted_vector_average([first_cell.colour, second_cell.colour], weights=weights)
        position = utils.weighted_vector_average([first_cell.position, second_cell.position], weights=weights)
        weight = sum(weights)
        neighbours = {*first_cell.neighbours, *second_cell.neighbours}

        for neighbour in first_cell.neighbours:
            Cell.remove_edge(first_cell, neighbour)

        for neighbour in second_cell.neighbours:
            Cell.remove_edge(second_cell, neighbour)

        del first_cell
        del second_cell

        cell = Cell(colour, position, weight, neighbours)

        for neighbour in neighbours:
            Cell.add_edge(cell, neighbour)

        return cell

    def __init__(self, colour, position=None, weight=1, neighbours=set()):
        self._colour = colour
        self.position = position
        self.weight = weight
        self.neighbours = SortedSet(neighbours, key=lambda cell: Cell.compare_colours(self, cell))

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, colour):
        raise NotImplementedError()  # Implementing this would require updating all neighbours and the edge set

    def least_difference_neighbour(self):
        return self.neighbours[0]


def inbounds(row, col, max_row, max_col, min_row=0, min_col=0):
    return min_row <= row < max_row and min_col <= col < max_col


def grid_neighbour_indices(row, col, height, width):
    offsets = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    return [(row + row_offset, col + col_offset) for row_offset, col_offset in offsets if
            inbounds(row + row_offset, col + col_offset, height, width)]


def grid_neighbours(grid, row, col):
    indices = grid_neighbour_indices(row, col, len(grid), len(grid[0]))
    return [grid[i][j] for i, j in indices]


def image_cell_grid(image_file):
    print(f'Loading cell grid for {image_file}')
    image_data = cv2.imread(image_file)
    height, width, colour_dimension = image_data.shape
    cells = [[Cell(colour=image_data[i, j], position=np.array([i, j], dtype=np.float64))
              for j in range(width)] for i in range(height)]
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            utils.print_progress(int(i * width + j), height * width, bar_length=70)
            for neighbour in grid_neighbours(cells, i, j):
                cell.neighbours.add(neighbour)
    return cells


def voronoi_fill(cells):
    pass  # TODO

# endregion


if __name__ == '__main__':

    try:
        raw_image_name = argv[1]
    except IndexError:
        raw_image_name = input('Enter an image name: ') or 'bliss.png'

    raw_image = os.path.join(raw_images_folder, raw_image_name)
    while not os.path.exists(raw_image):
        raw_image_name = input('File not found, try another: ') or 'bliss.png'
        raw_image = os.path.join(raw_images_folder, raw_image_name)

    compressed_image_name = raw_image_name
    compressed_image = os.path.join(compressed_images_folder, compressed_image_name)
    compress(raw_image, compressed_image)
