import logging
import math
import random
import operator
from itertools import product
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import igraph
from deap import algorithms, base, creator, tools
from automata.elemnts import ElementsType
from automata.elemnts.element import FakeRoot
from automata.elemnts.ste import PackedInput, PackedInterval, PackedIntervalSet


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
    colors_map = np.array(color_boundries, ndmin=2).transpose()
    colors_map = colors_map.repeat(3, axis=1) # make RGB gray scale
    cmap = colors.ListedColormap(colors_map)
    norm = colors.BoundaryNorm(boundries, cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0)
    ax.set_xticks(range(0, len(matrix), 64))
    ax.set_yticks(range(0, len(matrix[0]), 64))
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
                      merge_reports=True, same_residuals_only=True,
                      same_report_code=True, left_merge=True, right_merge=True,
                      combine_symbols=True, combine_equal_syms_only=False, add_all_input=True):
    logging.debug("minimization started...")
    assert automata.is_homogeneous, 'minimization only works for homogeneous representation'


    original_node_count = automata.nodes_count


    automata.set_all_symbols_mutation(mutation_value=True)

    automata.prone_all_symbol_sets()

    while True:
        current_node_cont = automata.nodes_count
        logging.debug("minimization iteration ..., current count {}".format(current_node_cont))
        if merge_reports:
            logging.debug("start report merge...")
            automata.combine_finals_with_same_symbol_set(same_residuals_only=same_residuals_only,
                                                          same_report_code=same_report_code)
            logging.debug("start report merge done!")
        if left_merge:
            logging.debug("start left merge...")
            automata.left_merge(merge_reports, same_residuals_only , same_report_code )
            logging.debug("start left merge done!")
        if right_merge:
            logging.debug("start right merge")
            automata.right_merge(merge_reports, same_residuals_only, same_report_code)
            logging.debug("start right merge done!")
        if combine_symbols:
            logging.debug("combine symbol set...")
            automata.combine_symbol_sets(merge_reports, same_residuals_only, same_report_code, combine_equal_syms_only)
            logging.debug("combine symbol set done!")
        if add_all_input:
            logging.debug("add all start...")
            automata.add_all_start_node()
            logging.debug("add all start done!")

        new_node_count = automata.nodes_count

        logging.debug("minimization iteration done, new node count {}".format(new_node_count))
        assert new_node_count <= current_node_cont, "it should always be smaller"
        if new_node_count == current_node_cont:
            break

    logging.debug("minimization done!")




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

class InputDistributer(object):
    def __init__(self, is_file, file_path=None, max_stride_size = 1, single_input_size = 1):
        max_stride_size = max(2,max_stride_size)
        '''
        this function read input stream and distribute it amnog readers
        :param is_file: True if we must read from file
        :param file_path: path of the input file
        :param max_stride_size: the maximum of stride size of all automatas that will read input
        :param single_input_size: size of each input in byte. it is important when reading from file
        '''
        self._circ_buffer = [0 for _ in range(max_stride_size)]
        self._head = 0
        self._batch_size = max_stride_size
        self._file = None
        self._is_file = is_file
        self._single_input_size = single_input_size
        if is_file:
            self._file = open(file_path, 'rb')

    def __del__(self):
        if self._is_file:
            self._file.close()

    def _get_input(self):
        if self._is_file:
            result = 0
            for _ in range(self._single_input_size):
                result = result * 256 + bytearray(self._file.read(1))[0]
            return result
        else:
            return  input('please eneter a number from 0 to {}'.format(pow(256, self._single_input_size)))

    def get_stream(self, stride_val):
        my_tail = 0
        b=[0 for _ in range(stride_val)]
        while True:
            for i in range(stride_val):
                if my_tail == self._head:
                    self._circ_buffer[self._head] = self._get_input()
                    self._head = (self._head + 1) % self._batch_size

                b[i] = self._circ_buffer[my_tail]
                my_tail = (my_tail + 1) % self._batch_size

            yield b

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

def _replace_equivalent_symbols(symbol_dictionary_list, atms_list, max_val):
    '''
    :param atms: list of auotmatas
    :param symbol_dictionary_list: a dictionary from nodes to set of 1D numbers
    :return: a list of automatas with replaces symbols
    '''


    for atm, sym_dic in zip(atms_list, symbol_dictionary_list):
        node_edge_iter = atm.nodes if atm.is_homogeneous else atm.get_edges()

        for ne in node_edge_iter:

            if atm.is_homogeneous and ne.type == ElementsType.FAKE_ROOT:
                continue
            sym_set = ne.symbols if atm.is_homogeneous else ne[2]['symbol_set']

            new_symbol_set = PackedIntervalSet([])

            sym_set.mutable = False
            ivls = get_interval(list(sym_dic[sym_set]))
            sym_set.mutable = True

            for l, r in ivls:
                left_pt = PackedInput((l,))
                right_pt = PackedInput((r,))
                new_symbol_set.add_interval(PackedInterval(left_pt, right_pt))

            new_symbol_set.prone()
            new_symbol_set.merge() # this is not necessary. TODO remove it

            if atm.is_homogeneous:
                ne.symbols = new_symbol_set
            else:
                ne[2]['symbol_set'] = new_symbol_set

        atm.stride_value = 1
        atm.max_val_dim = max_val


def get_equivalent_symbols(atms_list, replace=True, use_random_assignment=False, is_input_homogeneous=False):
    '''
    this function receives an input list of automatas and returns list of sets witk equivalnet symbols in the same set
    :param atms_list: list of automatas
    :param replace: True/False, if True, replaces the original autoamtaon symbol set
    :param use_random_assignment: if true, assign codes randomly else use an optimal policy
    :param is_input_homogeneous: if True it means all the input is in the same set as symbols
    :return: list of sets of equivalent symbols in the same set
    '''
    assert atms_list
    assert all((atm.stride_value == atms_list[0].stride_value and atm.max_val_dim == atms_list[0].max_val_dim for atm in atms_list))

    symbol_map = {}
    size = 0
    for atm in atms_list:
        atm.set_all_symbols_mutation(mutation_value=True)
        node_edge_iter = atm.nodes if atm.is_homogeneous else atm.get_edges()
        for ne in node_edge_iter:
            if atm.is_homogeneous and ne.type == ElementsType.FAKE_ROOT:
                continue
            buffer = {}
            sym_set = ne.symbols if atm.is_homogeneous else ne[2]['symbol_set']
            for pt in sym_set.points:
                current_map = symbol_map.get(pt, 0)
                if current_map not in buffer:
                    size += 1
                    buffer[current_map] = size

                symbol_map[pt] = buffer[current_map]

    # optimal range assignment
    optimal_dics = [] # keep tracks of compressed labels for each ste
    assigned_dic = {} # keeps the last assignment in order
    new_dic = {}

    assert pow(atms_list[0].max_val_dim + 1, atms_list[0].stride_value) >= len(symbol_map), 'This conndition should be satisfied'
    need_extra_code = is_input_homogeneous is False and pow(atms_list[0].max_val_dim + 1, atms_list[0].stride_value) > len(symbol_map) # check if the automatas cover all the input cases

    if use_random_assignment is False:
        sym_graph = nx.MultiGraph()

    for atm in atms_list:
        optimal_dic = {}
        node_edge_iter = atm.nodes if atm.is_homogeneous else atm.get_edges()
        for ne in node_edge_iter:
            if atm.is_homogeneous and ne.type == ElementsType.FAKE_ROOT:
                continue
            sym_set = ne.symbols if atm.is_homogeneous else ne[2]['symbol_set']
            sym_set = sym_set.clone()
            sym_set.mutable = False
            for pt in sym_set.points:
                orig_label = symbol_map[pt]
                if orig_label not in assigned_dic:
                    assigned_dic[orig_label] = len(assigned_dic) + (1 if need_extra_code else 0)
                new_dic[pt] = assigned_dic[orig_label]
                optimal_dic.setdefault(sym_set, set()).add(assigned_dic[orig_label])

            if use_random_assignment is False:
                # first adding nodes
                for new_s in optimal_dic[sym_set]:
                    if new_s not in sym_graph:
                        sym_graph.add_node(new_s)
                # second adding edges
                for src_s in optimal_dic[sym_set]:
                    for dst_s in optimal_dic[sym_set]:
                        if src_s != dst_s:
                            sym_graph.add_edge(src_s, dst_s)
        optimal_dics.append(optimal_dic)


    if use_random_assignment is False:
        new_map = {}
        while sym_graph.nodes:
            min_degree, min_node = None, None
            for n in sym_graph.nodes:
                if min_degree == None:
                    min_degree, min_node = sym_graph.degree(n), n
                elif sym_graph.degree(n) < min_degree:
                    min_degree, min_node = sym_graph.degree(n), n

            new_map[min_node] = len(new_map) + (1 if need_extra_code else 0)

            current_node = min_node

            while current_node is not None and sym_graph.neighbors(current_node):

                best_degree, best_node = None, None

                for neighb in sym_graph.neighbors(current_node):
                    if best_degree == None:
                        best_degree, best_node = sym_graph.number_of_edges(current_node, neighb), neighb
                    elif sym_graph.number_of_edges(current_node, neighb) > best_degree:
                        best_degree, best_node = sym_graph.number_of_edges(current_node, neighb), neighb

                sym_graph.remove_node(current_node)
                current_node = best_node
                if best_node is not None:
                    new_map[best_node] = len(new_map) + (1 if need_extra_code else 0)

            if current_node is not None:
                sym_graph.remove_node(current_node)

        for opt_dic in optimal_dics:
            for sym_set in opt_dic:
                old_set = opt_dic[sym_set]
                new_set=set()
                for ch in old_set:
                    new_set.add(new_map[ch])
                new_set = set([new_map[ch] for ch in old_set])

                opt_dic[sym_set] = new_set

    if replace:
        _replace_equivalent_symbols(symbol_dictionary_list=optimal_dics, atms_list=atms_list, max_val=len(new_map) + (-1 if need_extra_code is False else 0))

    if use_random_assignment is False:
        return {pt:new_map[pt_map] for pt, pt_map in new_dic.iteritems()}
    else:
        return new_dic

def get_interval(inp_list):
    '''

    :param inp_list: a list of integers
    :return: a list of intervals [(x1,x2), (x3,x4),.....]
    '''

    inp_list.sort()

    assert inp_list, 'empty list'

    result =[]

    new_start = prev_val = inp_list[0]
    for new_sym in inp_list[1:]:
        if new_sym == prev_val + 1:
            prev_val = new_sym
            continue
        else:
            result.append((new_start, prev_val))
            new_start = prev_val = new_sym

    result.append((new_start, prev_val))

    return result


def get_binary_val(val , bits_count, left_first = True):
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
    if left_first:
        result.reverse()
    return result


def is_there_a_binary_path(atm, src, dst, val, bits_count):
    if not src in atm.nodes or not  dst in atm.nodes:
        return False
    bit_val = get_binary_val(val, bits_count)
    def back_track(curr_node, curr_depth):
        if curr_depth == bits_count and curr_node == dst:
            return True
        elif curr_depth == bits_count:
            return False
        for _, curr_dst, data in atm.get_out_edges(curr_node, data=True, keys=False):
            sym_set = data['symbol_set']
            if sym_set.can_accept(PackedInput((bit_val[curr_depth],))):
                ca = back_track(curr_dst, curr_depth + 1)
                if ca:
                    return True
                else:
                    continue
        return False

    return back_track(src, 0)




def _get_alphabet_list(atm):
    '''
    this function returns set of unique symbols of all STEs in an automata
    the input automata should have stride 1 and homogeneous
    :param atm: the input automata
    :return: a set of unque integers
    '''
    assert atm.stride_value==1 and atm.is_homogeneous
    has_star = False
    pt_set=set()
    for node in atm.nodes:
        if node.type == ElementsType.FAKE_ROOT:
            continue

        if node.symbols.is_star(atm.max_val_dim):
            has_star = True
            continue

        for pt in node.symbols.points:
            pt_set.add(pt[0])

    return list(pt_set), has_star

def get_unified_symbol(atm, is_input_homogeneous, replace):
    '''
    this function receives a single stride homogemneous automata  and replace the symbols with integers starting from 0
    and the last integers for start case
    :param replace:
    :param atm: input atm
    :return: None
    '''
    assert atm.is_homogeneous and atm.stride_value==1

    def get_sym_dictionary(atm, pt_dic):
        '''
        this function receives an automata and a point dictionary. it returns a new dictionary whcih keys are nodes and values
        are a set of new symbols
        :param atm: under process automata
        :param pt_dic: a dictionary from integer points to new points (new points should start from 0)
        :return: a new dictionary (keys=nodes, values=set of new symbols)
        '''
        assert atm.stride_value == 1
        out_dic = {}
        codes_max = max(pt_dic.itervalues())

        for node in atm.nodes:
            if node.type == ElementsType.FAKE_ROOT:
                continue

            if node.symbols.is_star(max_val=atm.max_val_dim):
                out_dic[node.symbols] = set(range(codes_max + 1))
            else:
                val_set = out_dic.setdefault(node.symbols, set())
                for pt in node.symbols.points:
                    val_set.add(pt_dic[pt[0]])

        return out_dic

    alphabet_list, has_star = _get_alphabet_list(atm)
    if replace is True:
        pt_dic = {ch: (ch_idx + (1 if is_input_homogeneous is False and len(alphabet_list) < (atm.max_val_dim + 1) and has_star is False else 0))for ch_idx, ch in enumerate(alphabet_list)}
        sym_dict = get_sym_dictionary(atm, pt_dic=pt_dic)
        _replace_equivalent_symbols(symbol_dictionary_list=[sym_dict], atms_list=[atm], max_val=max(pt_dic.itervalues()))

    return alphabet_list


def pact_interconnect(base_size=256, combines_count=4, image_name=None):
    '''

    :param base_size:basic block size
    :param combines_count: combination_count
    :param image_name: if a string, it will save the switch image to that name
    :return:
    '''

    out_ports = [(192, 255), (256, 287), (480, 511), (512, 543), (736, 767), (768, 831)]
    in_ports = [(192, 255), (256, 287), (480, 511), (512, 543), (736, 767), (768, 831)]

    assert base_size % combines_count == 0
    perbase_i_con = (base_size / combines_count) / 2

    total_count = base_size * combines_count

    template = np.zeros((total_count, total_count), dtype=np.int8)

    for i in range(combines_count):
        base_start = i * base_size
        template[base_start:base_start + base_size, base_start:base_start + base_size] = 1


    for in_port, out_port in product(in_ports, out_ports):
        in_start, in_end = in_port
        out_start, out_end = out_port

        template[in_start:in_end + 1, out_start:out_end + 1] =1




    if image_name:
        draw_matrix(file_to_save=image_name, matrix=template,
                    boundries=[float(i) / 256 for i in range(257)], dpi=100)

    return template

def ga_routing(atms_list, routing_template, available_rows, draw_file=None):
    '''

    :param atms_list: list of automatons that need to be placed
    :param routing_template: the template as a 2d matrix. when there is a 1, we have a switch there
    :param available_rows: list of integers which represents the available rows
    :return: list of dictionaies. Each dictionary belongs to one of atms in the order they were recived.
    '''

    total_nodes = sum(map(lambda atm:atm.nodes_count, atms_list))
    assert total_nodes <= len(available_rows)

    node_start_index = 0
    node_dic = {}
    nodes_list = [None] * total_nodes
    for atm in atms_list:
        local_node_dic = atm.get_BFS_label_dictionary(start_from_root=True, set_nodes_idx=True, start_index=node_start_index)
        node_start_index += atm.nodes_count

        for node, index in local_node_dic.iteritems():
            node_dic[(atm, node)] = index

            nodes_list[index] = (atm, node)

    assert None not in nodes_list

    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    def interval_sampler(numbers, mean_interval, q_size):
        def get_next_interval_size():
            while True:
                number = np.random.poisson(mean_interval)
                yield number

        intervals = []
        current_start = 0

        for ivl_size in get_next_interval_size():
            intervals.append(numbers[current_start:current_start + ivl_size])
            current_start += ivl_size
            if current_start >= len(numbers):
                break

        #sample = random.sample(intervals, len(intervals))
        sample = []
        q = intervals[:q_size]
        for ivl in intervals[q_size:]:
            sample_index = random.randrange(q_size)
            sample.append(q[sample_index])
            del q[sample_index]
            q.append(ivl)

        sample.extend(random.sample(q, len(q)))

        out_sample = [num for ivl in sample for num in ivl]
        return out_sample

    toolbox.register("indices", interval_sampler, available_rows, 16, 3)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.indices)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)

    toolbox.register("mate", tools.cxOrdered)

    def mutlocal_shuffle(individual, indpb, move):
        size = len(individual)
        for i in xrange(size):
            if random.random() < indpb:
                swap_indx = random.randint(max(0, i - move), min(size - 1, i + move))
                individual[i], individual[swap_indx] = \
                    individual[swap_indx], individual[i]
        return individual,

    toolbox.register("mutate", mutlocal_shuffle, indpb=0.05, move=3)

    def evaluation(individual):
        cost = 0

        for node_idx, node_assignee in enumerate(individual):
            if node_idx >= total_nodes:
                break
            current_atm, current_node = nodes_list[node_idx]
            for neighb in current_atm.get_neighbors(current_node):
                neighb_idx = node_dic[(current_atm, neighb)]
                neighb_assignee = individual[neighb_idx]
                if routing_template[node_assignee, neighb_assignee] != 1:
                    cost += 1

        return cost,

    toolbox.register("evaluate", evaluation)
    toolbox.register("select", tools.selTournament, tournsize=50)

    fit_stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
    fit_stats.register('mean', np.mean)
    fit_stats.register('min', np.min)

    pop_size = 1024
    pop = toolbox.population(n=pop_size)

    pop.insert(0, creator.Individual(range(len(available_rows))))  # adding bfs solution as an initial guess

    result, log = algorithms.eaSimple(pop, toolbox,
                                      cxpb=0.5, mutpb=0.1,
                                      ngen=100, verbose=False,
                                      stats=fit_stats)

    best_individual = tools.selBest(result, k=1)[0]
    print('Fitness of the best individual: ', evaluation(best_individual)[0])

    if draw_file:
        plt.figure()
        color_boundries = [i/256.0 for i in range(256)]
        gray_switch_map = routing_template * 0.5
        colors_map = np.array(list(reversed(color_boundries)), ndmin=2).transpose()
        colors_map = colors_map.repeat(3, axis=1)  # make RGB gray scale
        cmap = colors.ListedColormap(colors_map)
        norm = colors.BoundaryNorm(color_boundries, cmap.N)
        plt.imshow(gray_switch_map, cmap=cmap, norm=norm)
        plt.gca().invert_yaxis()

        x_points, y_points = [], []
        for atm, node in nodes_list:
            src_idx = node_dic[(atm, node)]
            for neighb in atm.get_neighbors(node):
                dst_idx = node_dic[(atm, neighb)]
                x_points.append(src_idx)
                y_points.append(dst_idx)
        plt.scatter(x_points, y_points, c='r', s=1, alpha=0.3)

        x_points, y_points = [], []
        for atm, node in nodes_list:
            src_idx = node_dic[(atm, node)]
            for neighb in atm.get_neighbors(node):
                dst_idx = node_dic[(atm, neighb)]
                x_points.append(best_individual[src_idx])
                y_points.append(best_individual[dst_idx])

        plt.scatter(x_points, y_points, c='b', s=1, alpha=0.3)




        plt.savefig(draw_file, dpi=500)
        plt.close()


    plt.figure(figsize=(11, 4))
    plots = plt.plot(log.select('min'), 'c-', log.select('mean'), 'b-')
    plt.legend(plots, ('Minimum fitness', 'Mean fitness'), frameon=True)
    plt.ylabel('Fitness')
    plt.xlabel('Iterations')
    plt.show()


    return None
