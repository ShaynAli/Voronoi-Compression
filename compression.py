""" Compress an image using Voronoi cells """
import numpy as np
import scipy
import cv2
from sortedcontainers import SortedSet
from os import path
import sys
from itertools import product, chain, count
import utils

import pdb

script_folder, _ = path.split(path.abspath(__file__))
raw_images_folder = path.join(script_folder, 'images', 'raw')
compressed_images_folder = path.join(script_folder, 'images', 'compressed')


# region Compression

def compress(raw_image_file, compressed_image_file, n_edges=10000, verbose=False):

    def debug_print(value, enabled=verbose, *args, **kwargs):
        if enabled:
            print(value, *args, *kwargs)

    image_data = cv2.imread(raw_image_file)

    if image_data is None or len(image_data) == 0:
        raise FileNotFoundError(f'Image {raw_image_file} was not found')

    debug_print(f'Loading cell grid')
    image_cell_grid(image_data, print_progress=verbose)

    original_n_edges = len(Cell.edge_set)

    debug_print(f'Merging similar cells')
    while len(Cell.edge_set) > n_edges:
        Cell.merge_cells(*Cell.least_difference_edge())
        if verbose:
            utils.print_progress(original_n_edges - len(Cell.edge_set), original_n_edges - n_edges, bar_length=70)

    cells = set()
    print(f'Gathering reduced vertex set')
    for first_cell, second_cell in Cell.edge_set:
        cells.add(first_cell)
        cells.add(second_cell)
        utils.print_progress(len(cells), n_edges, bar_length=70)
    utils.print_progress(n_edges, n_edges, bar_length=70)

    compressed_image = voronoi_fill(cells, print_progress=verbose)

    cv2.imwrite(compressed_image_file, compressed_image)

# endregion


# region Voronoi cells

class Cell:

    edge_set = SortedSet(key=lambda cells: Cell.compare_colours(*cells))

    @staticmethod
    def add_edge(first_cell, second_cell):
        Cell.edge_set.add(frozenset([first_cell, second_cell]))
        first_cell._neighbours.add(second_cell)
        second_cell._neighbours.add(first_cell)


    @staticmethod
    def remove_edge(first_cell, second_cell):
        Cell.edge_set.remove(frozenset([first_cell, second_cell]))
        first_cell._neighbours.remove(second_cell)
        second_cell._neighbours.remove(first_cell)

    @staticmethod
    def remove_cell(cell):
        for neighbour in list(cell._neighbours):
            Cell.remove_edge(cell, neighbour)

    @staticmethod
    def least_difference_edge():
        return Cell.edge_set[0]

    @staticmethod
    def compare_colours(first_cell, second_cell):
        return np.linalg.norm(first_cell.colour - second_cell.colour)

    @staticmethod
    def merge_cells(first_cell, second_cell):

        try:
            Cell.remove_edge(first_cell, second_cell)
        except ValueError:
            ''' Ignore this clause - we only need to disconnect the cells if they are connected '''

        weights = [first_cell.weight, second_cell.weight]

        colour = utils.weighted_vector_average([first_cell.colour, second_cell.colour], weights=weights)
        position = utils.weighted_vector_average([first_cell.position, second_cell.position], weights=weights)
        weight = sum(weights)
        neighbours = {*first_cell._neighbours, *second_cell._neighbours}

        Cell.remove_cell(first_cell)
        Cell.remove_cell(second_cell)

        new_cell = Cell(colour, position, weight, neighbours)

        return new_cell

    def __init__(self, colour, position=None, weight=1, neighbours=set()):
        self._colour = colour
        self.position = position
        self.weight = weight
        self._neighbours = set()
        for cell in neighbours:
            Cell.add_edge(self, cell)

    def __repr__(self):
        return f'cell(c={self._colour}, p={self.position}, w={self.weight})'

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, colour):
        raise NotImplemented()  # Implementing this would require updating all neighbours and the edge set

    def least_difference_neighbour(self):
        return self._neighbours[0]


def inbounds(row, col, max_row, max_col, min_row=0, min_col=0):
    return min_row <= row < max_row and min_col <= col < max_col


def grid_neighbour_indices(row, col, height, width):
    offsets = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    return [(row + row_offset, col + col_offset) for row_offset, col_offset in offsets if
            inbounds(row + row_offset, col + col_offset, height, width)]


def grid_neighbours(grid, row, col):
    indices = grid_neighbour_indices(row, col, len(grid), len(grid[0]))
    return [grid[i][j] for i, j in indices]


def image_cell_grid(image_data, print_progress=False):
    height, width, colour_dimension = image_data.shape
    cells = [[Cell(colour=image_data[i, j], position=np.array([i, j], dtype=np.float64))
              for j in range(width)] for i in range(height)]
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            for neighbour in grid_neighbours(cells, i, j):
                Cell.add_edge(cell, neighbour)
            if print_progress:
                utils.print_progress(int(i * width + j) + 1, height * width, bar_length=70)
    return cells


def voronoi_fill(cells, print_progress=False):

    max_weight = max(cell.weight for cell in cells)

    raise NotImplementedError()

# endregion


if __name__ == '__main__':

    try:
        raw_image_name = sys.argv[1]
        raw_image = path.join(raw_images_folder, raw_image_name)
        if not path.exists(raw_image):
            print(f'File {sys.argv[1]} not found')
            sys.exit(1)

    except IndexError:
        raw_image_name = input('Enter an image name: ') or 'bliss.png'
        raw_image = path.join(raw_images_folder, raw_image_name)
        while not path.exists(raw_image):
            raw_image_name = input('File not found, try another: ') or 'bliss.png'
            raw_image = path.join(raw_images_folder, raw_image_name)

    compressed_image_name = raw_image_name
    compressed_image = path.join(compressed_images_folder, compressed_image_name)

    compress(raw_image, compressed_image, verbose=True)

    print(f'Saved compressed image to {compressed_image}')
