import cv2
import numpy as np
from sortedcontainers import SortedSet

from itertools import chain, product
import os

import utils


class VoronoiCell:

    def __init__(self, position=None, colour=None, weight=None, neighbours=None):
        self.position = position
        self.colour = colour
        self.weight = weight or 1
        self.neighbours = neighbours or set()

    def __repr__(self):
        return f'Cell(p={self.position},c={self.colour},w={self.weight})'


class VoronoiGraph:

    def __init__(self, cells=None):
        self.edges = SortedSet(key=lambda edge: utils.colour_distance(*edge))
        for cell in cells:
            self.add_cell(cell)

    @classmethod
    def image_grid(cls, image):

        rows, cols, _ = image.shape

        cell_grid = [[VoronoiCell() for _ in range(cols)] for _ in range(rows)]

        for i, j in product(range(rows), range(cols)):
            cell = cell_grid[i][j]
            cell.position = np.array([i, j])
            cell.colour = image[i, j]
            cell.neighbours = set(utils.grid_neighbours(cell_grid, i, j))

        return cls(chain(*cell_grid))

    def add_cell(self, cell):
        for neighbour in cell.neighbours:
            self.add_edge(cell, neighbour)

    def remove_cell(self, cell):
        for neighbour in cell.neighbours:
            self.remove_edge(cell, neighbour)

    def add_edge(self, *cells):
        self.edges.add(frozenset(cells))

    def remove_edge(self, *cells):
        self.edges.remove(frozenset(cells))

    def merge_cells(self, first_cell, second_cell):

        weights = [first_cell.weight, second_cell.weight]

        position = utils.weighted_vector_average([first_cell.position, second_cell.position], weights)
        colour = utils.weighted_vector_average([first_cell.colour, second_cell.colour], weights)
        weight = first_cell.weight + second_cell.weight
        neighbours = {*first_cell.neighbours, *second_cell.neighbours}

        new_cell = VoronoiCell(position=position, colour=colour, weight=weight, neighbours=neighbours)

        self.remove_cell(first_cell)
        self.remove_cell(second_cell)

        self.add_cell(new_cell)

    def least_difference_edge(self):
        return self.edges[0]


# TODO: Rasterization
#   Map cells to pixel neighborhoods then calculate pixel's weighted-closest based on cells in the neighborhood
#   Higher weighted cells should be in more neighborhoods
#   E.g. weight=1 -> neighborhoods at a distance <= 1, weight=2 -> neighborhoods at a distance <= 2, ...,
#     weight=w -> neighborhoods at a distance of <= w
#   Make sure to use same distance measure as the one used to determine which cells in each neighborhood are closest
#   E.g. with Euclidean distance:
#     □ □ □ □ □ □ □       □ □ □ □ □ □ □       □ □ □ ■ □ □ □
#     □ □ □ □ □ □ □       □ □ □ ■ □ □ □       □ ■ ■ ■ ■ ■ □
#     □ □ □ ■ □ □ □       □ □ ■ ■ ■ □ □       □ ■ ■ ■ ■ ■ □
#     □ □ ■ ■ ■ □ □       □ ■ ■ ■ ■ ■ □       ■ ■ ■ ■ ■ ■ ■
#     □ □ □ ■ □ □ □       □ □ ■ ■ ■ □ □       □ ■ ■ ■ ■ ■ □
#     □ □ □ □ □ □ □       □ □ □ ■ □ □ □       □ ■ ■ ■ ■ ■ □
#     □ □ □ □ □ □ □       □ □ □ □ □ □ □       □ □ □ ■ □ □ □
#          w=1                 w=2                 w=3


if __name__ == '__main__':

    test_image = cv2.imread(os.path.join('images', 'raw', 'bliss_10.png'))
    graph = VoronoiGraph.image_grid(test_image)
    breakpoint()
