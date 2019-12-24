import numpy as np

import sys


def print_progress(iteration, total, enabled=True, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Modified from https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a, which was written by Aubrey Taylor
    """

    if not enabled:
        return

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '|' * (bar_length - filled_length)

    sys.stdout.write(f'\r{prefix} {bar}  {percents}% {suffix}')

    if iteration == total:
        sys.stdout.write(' DONE\n')

    sys.stdout.flush()


def weighted_vector_average(vectors, weights):
    return np.average(vectors, weights=weights, axis=0)


def inbounds(row, col, max_row, max_col, min_row=0, min_col=0):
    return min_row <= row < max_row and min_col <= col < max_col


def bound_to_range(value, minimum, maximum):
    return min(maximum - 1, max(minimum, value))


def grid_neighbour_indices(row, col, height, width):
    offsets = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    return [(row + row_offset, col + col_offset) for row_offset, col_offset in offsets if
            inbounds(row + row_offset, col + col_offset, height, width)]


def grid_neighbours(grid, row, col):
    indices = grid_neighbour_indices(row, col, len(grid), len(grid[row]))
    return (grid[i][j] for i, j in indices)


def colour_distance(first_cell, second_cell):
    return np.linalg.norm(first_cell.colour - second_cell.colour)


def weighted_colour_distance(first_cell, second_cell):
    return (first_cell.weight + second_cell.weight) * colour_distance(first_cell, second_cell)
