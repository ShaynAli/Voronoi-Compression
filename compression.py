import cv2
import numpy as np

import os
from itertools import chain, product
from math import ceil

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

    def __init__(self, cells=None, image_shape=None):
        self.edges = set()
        for cell in cells:
            self.add_cell(cell)
        self.image_shape = image_shape

    @property
    def cells(self):
        return set(chain.from_iterable(self.edges))

    @classmethod
    def image_grid(cls, image):

        rows, cols, _ = image.shape

        cell_grid = [[VoronoiCell() for _ in range(cols)] for _ in range(rows)]

        for i, j in product(range(rows), range(cols)):
            cell = cell_grid[i][j]
            cell.position = np.array([i, j])
            cell.colour = image[i, j]
            cell.neighbours = set(utils.grid_neighbours(cell_grid, i, j))

        return cls(cells=chain.from_iterable(cell_grid), image_shape=image.shape)

    def add_cell(self, cell):
        for neighbour in cell.neighbours:
            self.add_edge(cell, neighbour)
            neighbour.neighbours.add(cell)

    def remove_cell(self, cell):
        for neighbour in cell.neighbours:
            self.remove_edge(cell, neighbour)
            neighbour.neighbours.remove(cell)

    def add_edge(self, *cells):
        self.edges.add(frozenset(cells))

    def remove_edge(self, *cells):
        self.edges.remove(frozenset(cells))

    def merge_cells(self, first_cell, second_cell):

        weights = [first_cell.weight, second_cell.weight]

        position = utils.weighted_vector_average([first_cell.position, second_cell.position], weights)
        colour = utils.weighted_vector_average([first_cell.colour, second_cell.colour], weights)
        weight = first_cell.weight + second_cell.weight
        neighbours = {*first_cell.neighbours, *second_cell.neighbours} - {first_cell, second_cell}

        new_cell = VoronoiCell(position=position, colour=colour, weight=weight, neighbours=neighbours)

        self.remove_cell(first_cell)
        self.remove_cell(second_cell)

        self.add_cell(new_cell)

    def least_difference_edge(self):
        return min(self.edges, key=lambda edge: utils.weighted_colour_distance(*edge))  # Can be optimized


def compress(image, ratio=0.5):
    graph = VoronoiGraph.image_grid(image)
    target_n_edges = ratio * len(graph.edges)
    while len(graph.edges) > target_n_edges:
        first_cell, second_cell = graph.least_difference_edge()
        graph.merge_cells(first_cell, second_cell)
    return graph


def decompress(graph, neighborhood_size=10):
    # Map cells to pixel neighborhoods then calculate pixel's weighted-closest based on cells in the neighborhood
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

    rows, cols, _ = graph.image_shape
    neighborhoods = [[set() for _ in range(ceil(cols / neighborhood_size))]
                     for _ in range(ceil(rows / neighborhood_size))]

    def neighborhood_indices(position):
        return int(position[0] / neighborhood_size), int(position[1] / neighborhood_size)

    def neighborhood(position):
        neighborhood_row, neighborhood_col = neighborhood_indices(position)
        return neighborhoods[neighborhood_row][neighborhood_col]

    # Populate neighborhoods
    for cell in graph.cells:
        neighborhood(cell.position).add(cell)
        # Add to grid neighbors where their distance is <= the cell's weight
        for nearby_neighborhood in utils.grid_neighbours_within_distance(neighborhoods, *neighborhood_indices(cell.position), max_distance=cell.weight):  # noqa
            nearby_neighborhood.add(cell)

    def closest_cell(position):
        return min(neighborhood(position), key=lambda c: np.linalg.norm(position - c.position) / c.weight)

    # Rasterize
    image_array = np.ndarray(graph.image_shape)
    for i, j in product(range(rows), range(cols)):
        image_array[i, j, :] = closest_cell((i, j)).colour

    return image_array


if __name__ == '__main__':

    test_image = 'bliss_500.png'

    input_image = cv2.imread(os.path.join('images', 'raw', test_image))
    compressed_graph = compress(input_image)
    output_image = decompress(compressed_graph)
    cv2.imwrite(os.path.join('images', 'decompressed', test_image), output_image)

    # TODO: Add argparser and update readme with usage
