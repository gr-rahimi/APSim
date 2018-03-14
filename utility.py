from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def draw_matrix(file_to_save, matrix, boundries):
    """

    :param file_to_save:
    :param matrix:
    :param boundries:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 10})
    color_boundries = boundries[::-1]
    colors_map = np.array(color_boundries, ndmin= 2).transpose()
    colors_map = colors_map.repeat(3, axis = 1) # make RGB gray scale

    cmap = colors.ListedColormap(colors_map)
    norm = colors.BoundaryNorm(boundries, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0)
    ax.set_xticks(range(0, len(matrix), 10))
    ax.set_yticks(range(0, len(matrix[0]), 10))

    plt.gca().invert_yaxis()
    plt.savefig(file_to_save)
    plt.clf()

def generate_diagonal_route(size, diagonal_width):
    routing_matrix = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        routing_matrix[i][i] = 1
        for j in range(1, diagonal_width + 1):
            if i - j >= 0:
                routing_matrix[i][i - j] = 1

            if i + j < size:
                routing_matrix[i][i + j] = 1

    return routing_matrix

