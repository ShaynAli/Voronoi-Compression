def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Modified from https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a, which was written by Aubrey Taylor
    """
    import sys
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '|' * (bar_length - filled_length)

    sys.stdout.write(f'\r{prefix} {bar}  {percents}% {suffix}')

    if iteration == total:
        sys.stdout.write('\n')

    sys.stdout.flush()


def weighted_vector_average(vectors, weights):
    import numpy as np
    return np.average(vectors, weights=weights, axis=0)
