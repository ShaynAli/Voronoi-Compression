import numpy as np

import sys


def print_progress(iteration, total, enabled=True, description='', bar_length=30):
    
    if not enabled:  # Simplifies calling code as it removes the need to be wrapped in an if/else
        return
    
    filled_length = round(bar_length * iteration / total)
    bar = filled_length * '█' + (bar_length - filled_length) * '|'
    percentage = f'{100 * (iteration / total):4.1f}'
    progress_end_indicator = ' DONE\n' if iteration == total else ''
    
    sys.stdout.write(f'\r{description:30} {bar}  {percentage}%{progress_end_indicator}')
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


def grid_neighbour_indices_within_distance(grid, row, col, max_distance):
    
    def within_distance(other_row, other_col):
        return np.linalg.norm([row - other_row, col - other_col]) <= max_distance
    
    boundary_indices = {(i, j) for i, j in grid_neighbour_indices(row, col, len(grid), len(grid[0]))
                        if within_distance(i, j)}
    neighbours = set()
    while boundary_indices:
        neighbours.update(boundary_indices)
        boundary_indices = {(i, j) for k, l in boundary_indices for i, j in
                            grid_neighbour_indices(k, l, len(grid), len(grid[0]))
                            if within_distance(i, j)} - neighbours
    return neighbours


def grid_neighbours_within_distance(grid, row, col, max_distance):
    indices = grid_neighbour_indices_within_distance(grid, row, col, max_distance)
    return (grid[i][j] for i, j in indices)


def colour_distance(first_cell, second_cell):
    return np.linalg.norm(first_cell.colour - second_cell.colour)


def weighted_colour_distance(first_cell, second_cell):
    return (first_cell.weight + second_cell.weight) * colour_distance(first_cell, second_cell)
