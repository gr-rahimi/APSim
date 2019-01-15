from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import igraph
from automata.elemnts import ElementsType
from automata.elemnts.element import FakeRoot
from automata.elemnts.ste import PackedInput, PackedInterval, PackedIntervalSet
import logging

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



def minimize_automata(automata,
                      merge_reports = True, same_residuals_only = True,
                      same_report_code = True, left_merge = True, right_merge = True,
                      combine_symbols = True):
    assert automata.is_homogeneous, 'minimization only works for homogeneous representation'
    original_node_count = automata.nodes_count

    for ste in automata.nodes:
        if ste.type is not ElementsType.FAKE_ROOT:
            ste.symbols.prone()

    while True:
        current_node_cont = automata.nodes_count
        logging.debug("minimization, current count{}".format(current_node_cont))
        if merge_reports:
            logging.debug("start report merge")
            automata.combine_finals_with_same_symbol_set(same_residuals_only=same_residuals_only,
                                                          same_report_code=same_report_code)
        if left_merge:
            logging.debug("start left merge")
            automata.left_merge(merge_reports , same_residuals_only , same_report_code )
        if right_merge:
            logging.debug("start right merge")
            automata.right_merge(merge_reports, same_residuals_only, same_report_code)
        if combine_symbols:
            logging.debug("combine symbol set")
            automata.combine_symbol_sets(merge_reports, same_residuals_only, same_report_code)
        new_node_count = automata.nodes_count
        assert new_node_count<= current_node_cont, "it should always be smaller"
        if new_node_count == current_node_cont:
            break
    final_node_count = automata.nodes_count

    #print "saved %d nodes"%(original_node_count- final_node_count,)






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


def multi_byte_stream(file_path, chunk_size):
    with open(file_path,'rb') as f:
        for read_bytes in iter(lambda: f.read(chunk_size), b''):
            yield bytearray(read_bytes)

def draw_symbols_len_histogram(atm):
    """
    this function draws a histogram based on the length of the symbols of all the nodes in the input automata
    :param atm: the under process automata
    :return: None
    """

    all_nodes = filter( lambda n : n.id != FakeRoot.fake_root_id, atm.nodes) # filter fake root
    all_nodes_symbols_len_count = [len(n.symbols) for n in all_nodes]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    #fig, ax = plt.subplots(2, 3)

    ax_symbol = fig.add_subplot(231)
    n, bins, patches = ax_symbol.hist(x=all_nodes_symbols_len_count, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    ax_symbol.grid(axis='y', alpha=0.75)
    ax_symbol.set_xlabel('Symbol size')
    ax_symbol.set_ylabel('Frequency')
    max_sym_count = n.max()
    ax_symbol.set_ylim(ymax=np.ceil(max_sym_count / 10) * 10 if max_sym_count % 10 else max_sym_count + 10)

    all_nodes_fan_in = atm.get_STEs_in_degree()
    ax_fan_in = fig.add_subplot(232)
    n, bins, patches = ax_fan_in.hist(x=all_nodes_fan_in, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    ax_fan_in.grid(axis='y', alpha=0.75)
    ax_fan_in.set_xlabel('fan in size')
    ax_fan_in.set_ylabel('Frequency')
    max_fanin_count = n.max()
    ax_fan_in.set_ylim(ymax=np.ceil(max_fanin_count / 10) * 10 if max_fanin_count % 10 else max_fanin_count + 10)

    all_nodes_fan_out = atm.get_STEs_out_degree()
    ax_fan_out = fig.add_subplot(233)
    n, bins, patches = ax_fan_out.hist(x=all_nodes_fan_out, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    ax_fan_out.grid(axis='y', alpha=0.75)
    ax_fan_out.set_xlabel('fan out size')
    ax_fan_out.set_ylabel('Frequency')
    max_fan_out_count = n.max()
    ax_fan_out.set_ylim(ymax=np.ceil(max_fan_out_count / 10) * 10 if max_fan_out_count % 10 else max_fan_out_count + 10)


    sym_fan_in_ax = fig.add_subplot(234, projection='3d')
    hist, xedges, yedges = np.histogram2d(all_nodes_symbols_len_count, all_nodes_fan_in, bins=20,
                                          range=[[0, max(all_nodes_symbols_len_count)], [0, max(all_nodes_fan_in)]])
    xpos, ypos = np.meshgrid(xedges[:-1] + 2, yedges[:-1] + 2, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 5 * np.ones_like(zpos)
    dz = hist.ravel()
    sym_fan_in_ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    sym_fan_in_ax.set_xlabel('symbol size')
    sym_fan_in_ax.set_ylabel('fan in')

    sym_fan_out_ax = fig.add_subplot(235, projection='3d')
    hist, xedges, yedges = np.histogram2d(all_nodes_symbols_len_count, all_nodes_fan_out, bins=20,
                                          range=[[0, max(all_nodes_symbols_len_count)], [0, max(all_nodes_fan_out)]])
    xpos, ypos = np.meshgrid(xedges[:-1] + 2, yedges[:-1] + 2, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 5 * np.ones_like(zpos)
    dz = hist.ravel()
    sym_fan_out_ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    sym_fan_out_ax.set_xlabel('symbol size')
    sym_fan_out_ax.set_ylabel('fan out')

    fan_in_fan_out_ax = fig.add_subplot(236, projection='3d')
    hist, xedges, yedges = np.histogram2d(all_nodes_fan_in, all_nodes_fan_out, bins=20,
                                          range=[[0, max(all_nodes_fan_in)], [0, max(all_nodes_fan_out)]])
    xpos, ypos = np.meshgrid(xedges[:-1] + 1, yedges[:-1] + 1, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 2 * np.ones_like(zpos)
    dz = hist.ravel()
    fan_in_fan_out_ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

    fan_in_fan_out_ax.set_xlabel('fan in')
    fan_in_fan_out_ax.set_ylabel('fan out')


    plt.show()


def get_equivalent_symbols(atms_list):
    '''
    this function receives an input list of automatas and returns list of sets witk equivalnet symbols in the same set
    :param atms_list: list of automatas
    :return: list of sets of equivalent symbols in the same set
    '''
    assert all((atm.stride_value == atms_list[0].stride_value for atm in atms_list))

    symbol_map = {}
    size = 0
    for atm in atms_list:
        assert atm.is_homogeneous
        for q in atm.nodes:
            if q.type == ElementsType.FAKE_ROOT:
                continue
            buffer = {}
            for pt in q.symbols.points:
                current_map = symbol_map.get(pt, 0)
                if current_map not in buffer:
                    size += 1
                    buffer[current_map] = size

                symbol_map[pt] = buffer[current_map]


    # optimal range assignment
    optimal_dics = [] # keep tracks of compressed labels for each ste
    assigned_dic = {} # keeps the last assignment in order
    new_dic = {}

    for atm in atms_list:
        optimal_dic = {}
        for q in atm.nodes:
            if q.type == ElementsType.FAKE_ROOT:
                continue
            for pt in q.symbols.points:
                orig_label = symbol_map[pt]
                if orig_label not in assigned_dic:
                    assigned_dic[orig_label] = len(assigned_dic) + 1

                new_dic[pt] = assigned_dic[orig_label]
                optimal_dic.setdefault(q, set()).add(assigned_dic[orig_label])

        optimal_dics.append(optimal_dic)

    return new_dic, optimal_dics


def replace_equivalent_symbols(symbol_dictionary_list, atms_list):
    '''
    :param atms: list of auotmatas
    :param symbol_dictionary_list: a dictionary from nodes to list of numbers
    :return: a list with replaces symbols
    '''

    for atm, sym_dic in zip(atms_list, symbol_dictionary_list):
        for q in atm.nodes:

            if q.type == ElementsType.FAKE_ROOT:
                continue
            new_symbol_set = PackedIntervalSet([]);

            new_symbols_list = sorted(list(sym_dic[q]))
            new_start = prev_val = new_symbols_list[0]

            for new_sym in new_symbols_list[1:]:
                if new_sym == prev_val + 1:
                    prev_val = new_sym
                    continue
                else:
                    new_symbol_set.add_interval(PackedInterval(PackedInput((new_start,)), PackedInput((prev_val,))))
                    new_start = prev_val = new_sym

            new_symbol_set.add_interval(PackedInterval(PackedInput((new_start,)), PackedInput((prev_val,))))
            new_symbol_set.prone()

            q.symbols = new_symbol_set

        atm.stride_value = 1



def get_binary_val(val , bits_count):
    '''
    this function receives an integer and returns back a binary value with bit width as bit counts
    :param val: the input value
    :param bits_ount: number of bits to be converted
    :return: a reversed iterator
    '''

    result=[]
    for _ in range(bits_count):
        result.append(val % 2)
        val /= 2

    assert val == 0, 'wrong bit counts'
    result.reverse()
    return result















