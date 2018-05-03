from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import igraph

def draw_matrix(file_to_save, matrix, boundries, **kwargs):
    """

    :param file_to_save:
    :param matrix:
    :param boundries:
    :return:
    """
    matplotlib.rcParams.update({'font.size': 7})
    plt.rcParams['axes.labelweight'] = 'bold'
    fig, ax = plt.subplots()
    draw_matrix_on_ax(ax,matrix, boundries)
    plt.savefig(file_to_save,**kwargs)
    plt.close()

def draw_matrix_on_ax(ax, matrix, boundries):
    color_boundries = boundries[::-1]
    colors_map = np.array(color_boundries, ndmin= 2).transpose()
    colors_map = colors_map.repeat(3, axis = 1) # make RGB gray scale
    cmap = colors.ListedColormap(colors_map)
    norm = colors.BoundaryNorm(boundries, cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0)
    ax.set_xticks(range(0, len(matrix), 15))
    ax.set_yticks(range(0, len(matrix[0]), 15))
    ax.invert_yaxis()


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

def generate_semi_diagonal_route(basic_block_size, one_dir_copy):
    routing_matrix = [[0 for _ in range(256)] for _ in range(256)]
    for i in range(0, 256, basic_block_size):
        for j1 in range(basic_block_size):
            for j2 in range(basic_block_size):
                routing_matrix[i+j1][i+j2] = 1
        for c in range(1, one_dir_copy + 1):
            for j1 in range(basic_block_size):
                for j2 in range(basic_block_size):
                    if (i + c * basic_block_size + j1) < 256 and  (i + j2) < 256:
                        routing_matrix[i + c * basic_block_size + j1][i + j2] = 1
                    if (i + c * basic_block_size + j2) < 256 and (i + j1) < 256:
                        routing_matrix[i + j1][i + c * basic_block_size + j2] = 1
    return routing_matrix



def minimize_automata(automata, merge_reports = False, same_residuals_only = False, same_report_code = False):
    original_node_count = automata.get_number_of_nodes(True)

    while True:
        current_node_cont = automata.get_number_of_nodes(True)
        print current_node_cont
        if merge_reports:
            automata._combine_finals_with_same_symbol_set(same_residuals_only=same_residuals_only,
                                                          same_report_code=same_report_code )
        automata.left_merge(merge_reports , same_residuals_only , same_report_code )
        automata.right_merge(merge_reports, same_residuals_only, same_report_code)
        automata.combine_symbol_sets()
        new_node_count = automata.get_number_of_nodes(True)
        assert new_node_count<= current_node_cont, "it should always be smaller"
        if new_node_count == current_node_cont:
            break
    final_node_count = automata.get_number_of_nodes(True)

    print "saved %d nodes"%(original_node_count- final_node_count,)






def generate_squared_routing(size, basic_block_WH, overlap):
    routing_matrix = [[0 for _ in range(size)] for _ in range(size)]

    def generate_pattern(start, size):
        if size == basic_block_WH:
            for i in range(size):
                for j in range(size):
                    routing_matrix[start + i][start + j] = 1

        else:
            generate_pattern(start + size / 2, size / 2)
            generate_pattern(start + overlap, size / 2)
            for i in range(size):
                for j in range(overlap):
                    routing_matrix[start + i][start + j] = 1
                    routing_matrix[start + j][start + i] = 1

    generate_pattern(0, size)

    def draw_pattern():

        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        ax.imshow(routing_matrix, cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0)
        ax.set_xticks(range(0, size, 15))
        ax.set_yticks(range(0, size, 15))

        plt.gca().invert_yaxis()
        plt.savefig("pattern.png")
        plt.clf()

    draw_pattern()
    return routing_matrix

def cut_switch_box(sb, l):

    for r_idx, r in enumerate(sb[:l]):
        sb[r_idx] = r[:l]

    return sb[:l]




def get_graph_from_matrix(routing_matrix, is_igraph):
    R, C = len(routing_matrix), len(routing_matrix[0])

    assert R==C, "matrix should be squared"
    if is_igraph:
        G = igraph.Graph(directed = True)
    else:
        G= nx.DiGraph()

    for i in range(R):
        if is_igraph:
            G.add_vertices(i)
        else:
            G.add_node(i)

    for i in range(R):
        for j in range(R):
            if routing_matrix[i][j]:

                G.add_edge(i, j)

    return G

def get_switch_count(switch_layout):
    sum_list = map(lambda x:sum(x), switch_layout)
    return sum(sum_list)


def get_star_symbol_set(stride_val):
    if stride_val == 1:
        return (0,255)
    else:
        return (get_star_symbol_set(stride_val/2),get_star_symbol_set(stride_val/2))




def generate_input(automaton, input_len, file_name):
    buf= bytearray()

    with open(file_name, 'wb') as f:
        for i in range(0,input_len, automaton.get_stride_value()):
            pass

def symbol_range_border_extractor(symbol_range, map):
    """

    :param symbol_range: a symbol range like ((0,5),(0,255)) =>0*
    :param map: a bit wise mapping to extract boundry => [0,1] => [0,255], [1,1] => [5,255] etc.
    :return: border point
    """
    l = len(map)
    assert _get_symbol_dim(symbol_range) == l, "length of map should be half of the input range"


    if l == 1:
        return [symbol_range[map[0]]]
    else:
        out_list = symbol_range_border_extractor(symbol_range[0], map[0:l/2])
        out_list.extend(symbol_range_border_extractor(symbol_range[1], map[l/2:]))
        return out_list





def _get_symbol_dim(input_symbol):
    import collections
    if not isinstance(input_symbol, collections.Sequence):
        return 0.5
    else:
        return int(2 * _get_symbol_dim(input_symbol[0]))



def _is_symbol_set_sorted(symbol_set):
    if not symbol_set:  # fake root has None symbol set
        return  True
    for  prev_pt, next_pt in zip(symbol_set[:-1], symbol_set[1:]):
        if next_pt< prev_pt:
            return False
    return  True