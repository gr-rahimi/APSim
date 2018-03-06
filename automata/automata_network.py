from  elemnts.ste import S_T_E
from elemnts.element import StartType
from elemnts.or_elemnt import OrElement
import networkx as nx
import matplotlib.pyplot as plt
from collections import  deque
from tqdm import tqdm
import os
import itertools
import random
import sys
import time
random.seed(a = None)



class Automatanetwork(object):
    known_attributes = {'id','name'}
    _fake_root = "fake_root"



    def __init__(self, id ,is_homogenous = True, stride = 1):
        #self._start_STEs=[]
        self._has_modified = True
        self._my_graph = nx.MultiDiGraph()
        self._node_dict={} # this dictionary helps retrieving nides using id(as a string)
        self._is_homogeneous = is_homogenous
        self._fake_root = S_T_E(start_type = StartType.fake_root, is_report=False,
                                is_marked=False, id=Automatanetwork._fake_root,
                                symbol_set=None)
        self.add_element(self._fake_root)  # This is not areal node. It helps for simpler striding code
        self._stride = stride
        self._id = id
        #self._node_id = 0





    def _get_new_id(self):
        new_random_id = str(random.randint(1, sys.maxint))
        while new_random_id in self._node_dict:
            new_random_id = str(random.randint(1, sys.maxint))
        return new_random_id


    @classmethod
    def _from_graph(cls, id ,is_homogenous, graph ,stride):
        assert is_homogenous, "graph should be in homogenous state"

        automata = Automatanetwork(id = id,is_homogenous = True, stride = stride) # this will create a initial graph but we do not need it

        automata._node_dict = {}
        automata._my_graph = graph
        automata._fake_root = S_T_E(start_type=StartType.fake_root, is_report=False,
                                is_marked=False, id=Automatanetwork._fake_root,
                                symbol_set=None)
        automata.add_element(automata._fake_root)

        for node in list(graph.nodes()):
            if node.get_start() == StartType.all_input or\
                node.get_start() == StartType.start_of_data:
                automata.add_edge(automata._fake_root, node)
            if node.get_start()!= StartType.fake_root: # fake root has already been added to dictinonary
                assert not node.get_id() in automata._node_dict
                automata._node_dict[node.get_id()] = node
        return automata





    @classmethod
    def from_xml(cls, xml_node):

        Automatanetwork._check_validity(xml_node)

        graph_ins = cls( id = xml_node.attrib['id'])


        for child in xml_node:
            if child.tag == 'state-transition-element':
                ste = S_T_E.from_xml_node(child)
                graph_ins.add_element(ste)
            elif child.tag == 'or':
                or_gate = OrElement.from_xml_node(child)
                graph_ins.add_element(to_add_element=or_gate, connect_to_fake_root= False)
            elif child.tag == 'description': # not important
                continue
            else:
                raise RuntimeError('unsupported child of automata-network-> ' + child.tag )


        for src_node_id, src_node in graph_ins._node_dict.iteritems():
            if src_node_id == Automatanetwork._fake_root:
                continue
            for dst_id in src_node.get_adjacency_list():
                dst_node = graph_ins._node_dict[dst_id]
                graph_ins.add_edge(src_node, dst_node)

            src_node.delete_adjacency_list()

        return graph_ins

    def get_stride_value(self):
        """
        return number of chars eating per iteration
        :return:
        """
        return self._stride

    def get_nodes(self):
        return self._my_graph.nodes()

    def get_filtered_nodes(self,lambda_func):
        return filter(lambda_func, self._my_graph.nodes)


    @classmethod
    def from_graph(cls, graph, id):
        """

        :param graph: original graph structure
        :param id: string representing id of graph
        :return: an instance of type Automatanetwork
        """
        graph_ins = cls()
        graph_ins._id = id
        graph_ins._my_graph = graph

        return graph_ins


    @staticmethod
    def _check_validity(xml_node):
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(Automatanetwork.known_attributes)

    def unmark_all_nodes(self):
        for _,n in self._node_dict.iteritems():
            n.set_marked(False)

    # def _rebuild_graph_properties(self):
    #     """
    #     This function rebuild all  peripheral properties of graph
    #     :return:
    #     """
    #     self._start_STEs=[]
    #     for node in self._my_graph.nodes:
    #         if node.is_start():
    #             self._start_STEs.append(node)
    #
    #     self._has_modified =False


    def add_element(self, to_add_element, connect_to_fake_root = True):

        """
        :param to_add_element: Add a ste to the graph
        :return:
        """
        assert to_add_element.get_id() not in self._node_dict
        self._my_graph.add_node(to_add_element)
        self._node_dict[to_add_element.get_id()] = to_add_element
        self._has_modified = True

        if self.is_homogeneous() and to_add_element.is_start() and connect_to_fake_root: # only for homogenous graphs
            self.add_edge(self._fake_root, to_add_element) # add an esge from fake root to all start nodes


    def get_STE_by_id(self, id):
        """
        this function returns the STE instance using the id
        :param id: id of the ste
        :return:
        """
        return self._node_dict.get(id, None)


    # def get_start_STEs(self):
    #     """
    #     get a list of start STEs.
    #     :return: list of start
    #     """
    #     if not self._has_modified:
    #         return tuple(self._start_STEs)
    #
    #     else:
    #         self._rebuild_graph_properties()
    #         return tuple(self._start_STEs)


    def get_number_of_nodes(self):
        return len(self._my_graph) -1

    def add_edge(self, src, dest,**kwargs):
        if not 'label' in kwargs:
            kwargs['label'] = dest.get_symbols()
        if not 'start_type' in kwargs:
            kwargs['start_type'] = dest.get_start()

        self._my_graph.add_edge(src, dest, **kwargs)
        self._has_modified = True


    def get_single_stride_graph(self):
        """
        This function make a new graph with single stride
        It assumes that the graph in cyrrent state is a homogeneous graph
        :return: a graph with a single step stride
        """
        assert self.is_homogeneous() and not self.does_have_all_input(), "Automata should be in homogenous mode and without all input nodes"
        dq = deque()
        self.unmark_all_nodes()
        strided_graph = Automatanetwork(id = self._id + "_S1",is_homogenous= False, stride= self.get_stride_value()*2)
        self._fake_root.set_marked(True)
        ###
        dq.appendleft(self._fake_root)

        while dq:
            #print len(dq)
            current_ste = dq.pop()

            for l1_neigh in self._my_graph.neighbors(current_ste):
                for l2_neigh in self._my_graph.neighbors(l1_neigh):
                    if not l2_neigh.is_marked():

                        temp_ste = S_T_E(start_type = StartType.unknown, is_report= l2_neigh.is_report(),
                                                    is_marked=False, id = l2_neigh.get_id(), symbol_set= None)
                        strided_graph.add_element(temp_ste)
                        l2_neigh.set_marked(True)
                        dq.appendleft(l2_neigh)

                    strided_graph.add_edge(strided_graph.get_STE_by_id(current_ste.get_id()), strided_graph.get_STE_by_id(l2_neigh.get_id()),
                                           label=tuple((span1, span2) for span1 in l1_neigh.get_symbols() for span2 in l2_neigh.get_symbols()),
                                           start_type = l1_neigh.get_start() if current_ste.get_start() == StartType.fake_root else StartType.non_start)

        #strided_graph.draw_graph("mid_graph", draw_edge_label= True)
        #strided_graph.make_homogenous()
        return strided_graph


    def get_number_of_start_nodes(self):
        """
        number of start nodes (fake root neighbors) in a graph
        :return:
        """
        return len(list(self._my_graph.neighbors(self._fake_root)))

    def get_number_of_report_nodes(self):
        """
        number of report nodes in a graph
        :return:
        """
        count = 0
        for node in self._my_graph.nodes():
            if node.get_start() == StartType.fake_root:
                continue
            elif node.is_report():
                count+=1
        return  count

    def delete_node(self, node):
        self._my_graph.remove_node(node)
        del self._node_dict[node.get_id()]

    def make_homogenous(self):
        """
        :return:
        """
        self.unmark_all_nodes()
        assert not self.is_homogeneous() # only works for non-homogeneous graph
        dq = deque()
        self._fake_root.set_marked(True)
        dq.appendleft(self._fake_root)

        while dq:
            #print len(dq)
            current_ste = dq.pop()
            #print "porcessing" , current_ste
            if current_ste.get_start() == StartType.fake_root: # fake root does need processing
                for neighb in self._my_graph.neighbors(current_ste):
                    assert not neighb.is_marked()
                    neighb.set_marked(True)
                    dq.appendleft(neighb)
                continue # process next node from the queue

            for neighb in self._my_graph.neighbors(current_ste):
                if not neighb.is_marked():
                    neighb.set_marked(True)
                    dq.appendleft(neighb)

            src_dict_non_start = {}
            src_dict_all_start = {}
            src_dict_start_of_data = {}

            #src_nodes = list(self._my_graph.predecessors(current_ste))
            #edges = self._my_graph.edges(src_nodes, data = True, keys = False)
            edges = self._my_graph.in_edges(current_ste, data = True, keys = False)


            for edge in edges:

                label = edge[2]['label']
                start_type = edge[2]['start_type']

                if start_type == StartType.non_start:
                    src_dict_non_start.setdefault(edge[0], set()).update(label)
                elif start_type == StartType.start_of_data:
                    src_dict_start_of_data.setdefault(edge[0], set()).update(label)
                elif start_type == StartType.all_input:
                    src_dict_all_start.setdefault(edge[0], set()).update(label)
                else:
                    assert False # It should not happen

            new_nodes = []
            new_all_input_nodes = self._make_homogenous_node(curr_node = current_ste, connectivity_dic = src_dict_all_start,
                                       start_type = StartType.all_input )
            new_nodes.extend(new_all_input_nodes)

            new_start_of_data_nodes = self._make_homogenous_node(curr_node=current_ste, connectivity_dic=src_dict_start_of_data,
                                       start_type=StartType.start_of_data)
            new_nodes.extend(new_start_of_data_nodes)

            new_non_start_nodes =  self._make_homogenous_node(curr_node=current_ste, connectivity_dic=src_dict_non_start,
                                       start_type=StartType.non_start)
            new_nodes.extend(new_non_start_nodes)

            if current_ste in self._my_graph.neighbors(current_ste): # handling self loop nodes

                for neighb, on_edge_char_set in src_dict_non_start.iteritems():
                    if neighb == current_ste:
                        self_loop_handler = S_T_E(start_type = StartType.non_start, is_report= current_ste.is_report(),
                                                  is_marked= True, id =  self._get_new_id(),
                                                  symbol_set= on_edge_char_set) #self loop handlers are always non start nodes
                        self.add_element(self_loop_handler)
                        self.add_edge(self_loop_handler, self_loop_handler)
                        for node in new_nodes:
                            self.add_edge(node, self_loop_handler)

                        out_edges = self._my_graph.out_edges(current_ste, data= True, keys = False)
                        for edge in out_edges:
                            if edge[0] == edge[1]: # self loop node
                                continue
                            self.add_edge(self_loop_handler, edge[1], label = edge[2]['label'], start_type = edge[2]['start_type'])


            self.delete_node(current_ste)

        self._is_homogeneous = True


    def is_homogeneous(self):
        return self._is_homogeneous


    def draw_graph(self, file_name, draw_edge_label = False):
        """

        :param file_name: name of the png file
        :param draw_edge_label: True if writing edge labels is required
        :return:
        """
        pos = nx.spring_layout(self._my_graph, k =0.2)
        node_color = [node.get_color() for node in self._my_graph.nodes()]
        nx.draw(self._my_graph, pos, node_size = 1, width = 0.2, arrowsize = 2, node_color= node_color)

        if draw_edge_label: # draw with edge lable
            edge_lables = nx.get_edge_attributes(self._my_graph, 'label')

            nx.draw_networkx_edge_labels(self._my_graph, pos, node_size = 2, width = 0.1, arrowsize = 2,
                                         node_color= node_color, font_size= 1 )

        plt.savefig(file_name, dpi=1000)
        #self.does_have_self_loop()
        plt.clf()




    def _make_homogenous_node(self, curr_node, connectivity_dic, start_type):

        new_nodes = []

        for neighb, on_edge_char_set in connectivity_dic.iteritems():
            if curr_node != neighb:
                new_node = S_T_E(start_type=start_type, is_report=curr_node.is_report(), is_marked=True,
                                 id=self._get_new_id(),
                                 symbol_set=on_edge_char_set)

                self.add_element(new_node, connect_to_fake_root= False) # it will not be coonected to fake_root since the graph is not homogeneous at the moment
                new_nodes.append(new_node)
                self.add_edge(neighb, new_node, label = new_node.get_symbols(), start_type = new_node.get_start())
                out_edges = self._my_graph.out_edges(curr_node, data = True, keys = False)

                for edge in out_edges:
                    if edge[1] != edge[0]:
                        self.add_edge(new_node, edge[1], label = edge[2]['label'], start_type = edge[2]['start_type'])
                    else:
                        continue # not necessary for self loops
            else:
                assert start_type == StartType.non_start, "self lopps should be in non start category"
                continue # self-loops node will be processed later

        return new_nodes


    def does_have_self_loop(self):
        """
        check if there is a node in the graph that have self loop
        :return: True if there is a node otherwise return False
        """
        found = False
        for node in self._my_graph.nodes():
            if node in self._my_graph.neighbors(node):
                print node.get_id()
                found = True

        return found
    def does_STE_has_self_loop(self, ste):
        return ste in self._my_graph.neighbors(ste)

    def print_summary(self, print_detailed_final_states = False):
        print"********************Automata Report********************"
        print "report for", self._id
        print "Number of nodes: ", self.get_number_of_nodes()
        print "Number of start nodes", self.get_number_of_start_nodes()
        print "Number of report nodes", self.get_number_of_report_nodes()
        print "does have all_input? ", self.does_have_all_input()
        print "does have special element?", self.does_have_special_elements()
        print "is Homogenous?", self.is_homogeneous()
        print "stride value = ", self.get_stride_value()
        if print_detailed_final_states:
            self._print_final_states_detail()

        print "#######################################################"


    def combile_finals_with_same_symbol_set(self):
        """
        This function merges all the final nodes that
        does not have self loop and out edge, but with the same symbol sets
        :return:
        """
        symbol_set_dict = {}
        final_nodes = self.get_filtered_nodes(lambda ste: ste.is_report())
        for idx_fnode, f_node in enumerate(final_nodes):
            found_symbol_set_key = False
            for key in symbol_set_dict:
                if f_node.is_symbolset_a_subsetof_self_symbolset(key.get_symbols()) and \
                        key.is_symbolset_a_subsetof_self_symbolset(f_node.get_symbols()):
                    found_symbol_set_key = True
                    symbol_set_dict[key].append(f_node)
                    break
            if not found_symbol_set_key:
                symbol_set_dict[f_node] = []

        for _ , nodes in symbol_set_dict.iteritems():
            to_bejoined_nodes =[]
            for node in nodes:
                if not self.does_STE_has_self_loop(node) and len(list(self._my_graph.neighbors(node))) == 0:
                    to_bejoined_nodes.append(node)

            if len(to_bejoined_nodes) > 1:
                anchor_node = to_bejoined_nodes[0]
                for other_node in to_bejoined_nodes[1:]:
                    for pred in self._my_graph.predecessors(other_node):
                        self.add_edge(pred, anchor_node)
                    self.delete_node(other_node)






    def _print_final_states_detail(self):
        final_states_with_self_loop = 0
        final_states_with_back_connection = 0
        symbol_set_dict = {}

        final_nodes = self.get_filtered_nodes(lambda ste: ste.is_report())

        for idx_fnode,f_node in enumerate(final_nodes):
            if self.does_STE_has_self_loop(f_node):
                final_states_with_self_loop+=1

            for neighb in self._my_graph.neighbors(f_node):
                if not neighb.is_report():
                    final_states_with_back_connection += 1
                    break
            found_symbol_set_key = False
            for key in symbol_set_dict:
                if f_node.is_symbolset_a_subsetof_self_symbolset(key.get_symbols()) and \
                        key.is_symbolset_a_subsetof_self_symbolset(f_node.get_symbols()):
                    found_symbol_set_key = True
                    symbol_set_dict[key] += 1
                    break
            if not found_symbol_set_key:
                symbol_set_dict[f_node] = 1

        print "number of final nodes with self connection:", final_states_with_self_loop
        print "number of final nodes with back connection:", final_states_with_back_connection
        print "number of different symbol sets", len(symbol_set_dict)






    def split(self):
        """
        split the current automata and rturn both of them
        :return:
        """

        left_automata = Automatanetwork(id = self._id+"_split1",is_homogenous= self.is_homogeneous(), stride= self.get_stride_value()/2)
        right_automata = Automatanetwork(id = self._id+"_split2", is_homogenous=self.is_homogeneous(), stride=self.get_stride_value()/2)
        self.unmark_all_nodes()
        self._fake_root.set_marked(True) # fake root has been added in the constructor for both splited graphs

        self._split_node(self._fake_root, left_automata = left_automata, right_automata= right_automata)

        return left_automata, right_automata

    def _split_node(self, node, left_automata, right_automata):
        """

        :param node: the node taht is going to be splitted
        :param left_automata: the first automata to put the first split
        :param right_automata: the second automata to put the second split
        :return:
        """

        for neighb in self._my_graph.neighbors(node):
            if not neighb.is_marked():
                neighb.set_marked(True)
                left_symbols, right_symbols = neighb.split_symbols()

                left_ste = S_T_E( start_type = neighb.get_start(), is_report = neighb.is_report(),
                                 is_marked = True, id = neighb.get_id(), symbol_set= left_symbols)
                left_automata.add_element(left_ste)

                right_ste = S_T_E( start_type = neighb.get_start(), is_report = neighb.is_report(),
                                 is_marked=True, id=neighb.get_id(), symbol_set=right_symbols)
                right_automata.add_element(right_ste)

                self._split_node(neighb, left_automata= left_automata, right_automata=right_automata)

            left_ste_src = left_automata.get_STE_by_id(node.get_id())
            left_ste_dst = left_automata.get_STE_by_id(neighb.get_id())
            left_automata.add_edge(left_ste_src, left_ste_dst, label = left_ste_dst.get_symbols(), start_type = left_ste_dst.get_start())

            right_ste_src = right_automata.get_STE_by_id(node.get_id())
            right_ste_dst = right_automata.get_STE_by_id(neighb.get_id())
            right_automata.add_edge(right_ste_src, right_ste_dst, label=left_ste_dst.get_symbols(),
                                   start_type=left_ste_dst.get_start())

    def _find_next_states(self, current_active_states, input):
        """

        :param current_active_states: a set of current active states
        :param input: an iterable symbol set
        :return: (True/False) if there is a report element in new states, (Set) new states
        """
        new_active_states = set()
        is_report = False

        for act_st in current_active_states:
            if self.is_homogeneous():
                for neighb in self._my_graph.neighbors(act_st):
                    can_accept, temp_is_report = neighb.can_accept(input=input)
                    is_report = is_report or temp_is_report
                    if can_accept:
                        new_active_states.add(neighb)
            else:
                out_edges = self._my_graph.out_edges(act_st, data=True, keys=False)
                for edge in out_edges:
                    can_accept, temp_is_report = edge[1].can_accept(input=input,
                                                                    on_edge_symbol_set=edge[2]['label'])
                    is_report = is_report or temp_is_report
                    if can_accept:
                        new_active_states.add(edge[1])

        return is_report, new_active_states

    def feed_input(self, input_stream, offset = 0, jump = 0):

        my_stride = self.get_stride_value()
        assert offset < (jump), "this condition should be met"

        temp_g = (itertools.islice(input_stream, offset + i, len(input_stream), jump ) for i in range(my_stride))
        g = itertools.izip(*temp_g)


        active_states = set([self._fake_root])

        if self.is_homogeneous():
            all_start_states = [all_start_neighb for all_start_neighb in self._my_graph.neighbors(self._fake_root)
                                if all_start_neighb.get_start() == StartType.all_input]
        else:
            all_start_edges = [all_start_edge for all_start_edge in
                               self._my_graph.out_edges(self._fake_root, data=True, keys=False)
                               if all_start_edge[2]['start_type'] == StartType.all_input]

        for input in tqdm(g, total = len(input_stream) / jump, unit="symbol"):

            is_report, new_active_states = self._find_next_states(current_active_states = active_states, input = input)

            if self.is_homogeneous():
                for all_start_state in all_start_states:
                    can_accept, temp_is_report = all_start_state.can_accept(input=input)
                    is_report = is_report or temp_is_report
                    if can_accept:
                        new_active_states.add(all_start_state)
            else:
                for all_start_edge in all_start_edges:
                    can_accept, temp_is_report = all_start_edge[1].can_accept(input=input,
                                                                    on_edge_symbol_set=all_start_edge[2][
                                                                        'label'])
                    is_report = is_report or temp_is_report
                    if can_accept:
                        new_active_states.add(all_start_edge[1])

            active_states = new_active_states
            yield active_states, is_report







    def feed_file(self, input_file):
        """

        :param input_file: file to be feed to the input
        :return:
        """
        active_states = set([self._fake_root])
        if self.is_homogeneous():
            all_start_states = [all_start_neighb for all_start_neighb in self._my_graph.neighbors(self._fake_root)
                            if all_start_neighb.get_start() == StartType.all_input]
        else:
            all_start_edges = [all_start_edge for all_start_edge in self._my_graph.out_edges(self._fake_root, data=True, keys = False)
                                if all_start_edge[2]['start_type'] == StartType.all_input]

        with open(input_file, 'rb') as f:
            file_size = os.path.getsize(input_file)
            for input in tqdm(iter(lambda: f.read(self.get_stride_value()),b''), total= file_size/ self.get_stride_value(), unit = "symbol"):

                input = bytearray(input)

                is_report, new_active_states = self._find_next_states(current_active_states=active_states, input=input)

                if self.is_homogeneous():
                    for all_start_state in all_start_states:
                        can_accept, temp_is_report = all_start_state.can_accept(input=input)
                        is_report = is_report or temp_is_report
                        if can_accept:
                            new_active_states.add(all_start_state)
                else:
                    for all_start_edge in all_start_edges:
                        can_accept, temp_is_report = all_start_edge[1].can_accept(input=input,
                                                                        on_edge_symbol_set=all_start_edge[2]['label'])
                        is_report = is_report or temp_is_report
                        if can_accept:
                            new_active_states.add(all_start_edge[1])


                active_states = new_active_states
                yield active_states, is_report


    def remove_ors(self):
        to_be_deleted_ors= []
        for node in self._my_graph.nodes:
            if node == self._fake_root:
                continue
            if node.is_OR():
                for pre_childs in self._my_graph.predecessors(node):
                    if pre_childs == node :
                        continue

                    for post_childs in self._my_graph.neighbors(node):
                        if post_childs == node:
                            continue

                        self.add_edge(pre_childs, post_childs)

                to_be_deleted_ors.append(node)

        for or_node in to_be_deleted_ors:
            self.delete_node(or_node)

    def does_have_all_input(self):
        """
        check if there is a node that has an all input property
        :return: True if there is.
        """
        for node in self._my_graph.neighbors(self._fake_root):
            if node.get_start() == StartType.all_input:
                return True
        return  False

    def remove_all_start_nodes(self):
        """
        this funstion add a new node that accepts Dot Kleene start and connect it to all "all_input nodes"
        :return: a graph taht does not have any start node with all_input condition
        """

        assert self.is_homogeneous() and self.get_stride_value() == 1,\
            "Graph should be in homogenous state to handle this situation and alaso it should be single stride"

        if not self.does_have_all_input():
            return

        star_node = S_T_E(start_type = StartType.start_of_data, is_report = False, is_marked = False,
                          id = "all_input_handler", symbol_set={(0, 255)}, adjacent_S_T_E_s = [])

        self.add_element(to_add_element = star_node, connect_to_fake_root = True)

        self.add_edge(star_node, star_node)

        temp_var = list(self._my_graph.neighbors(star_node))


        for node in self._my_graph.neighbors(self._fake_root):

            if node == star_node:
                continue
            if node.get_start() == StartType.all_input:
                node.set_start(StartType.start_of_data)
                self.add_edge(star_node,node)


    def get_connected_components_size(self):
        start_time = time.time()
        undirected_graph= self._my_graph.to_undirected()
        undirected_graph.remove_node("fake_root")
        print "componnent size calculation took:", time.time()-start_time
        return tuple(len(g) for g in sorted(nx.connected_components(undirected_graph), key=len, reverse=True))

    def get_connected_components_as_automatas(self):
        assert not self.does_have_special_elements(), "This function does not support automatas with special elements"
        assert self.is_homogeneous(), "Graph should be in homogeneous state"
        undirected_graph = self._my_graph.to_undirected()
        undirected_graph.remove_node("fake_root")
        ccs =  nx.connected_components(undirected_graph)
        splitted_automatas = []

        for cc_idx, cc in enumerate(ccs):
            sg = self._my_graph.subgraph(cc)
            new_graph = nx.MultiDiGraph(sg)
            new_autoama = Automatanetwork._from_graph(id = self._id + str(cc_idx),is_homogenous=True, graph = new_graph, stride= self.get_stride_value())
            splitted_automatas.append(new_autoama)


        return splitted_automatas








    def does_have_special_elements(self):
        for nodes in self._my_graph.nodes():
            if nodes.is_special_element():
                return True
        return  False



    def left_merge(self):
        assert self.is_homogeneous(), "This function is working only for homogeneous case!"
        self.unmark_all_nodes()
        dq = deque()

        self._fake_root.set_marked(True)
        dq.appendleft(self._fake_root)


        while dq:

            current_node = dq.pop()


            for children in list(self._my_graph.neighbors(current_node)):
                if children.is_marked():
                    continue

                if not children in self._node_dict: # this children has been deleted
                    continue

                if children == current_node:
                    continue # self loop
                for second_children in list(self._my_graph.neighbors(current_node)):
                    if not second_children in self._node_dict:
                        continue # deleted node
                    if second_children == children :
                        continue #comparing the same node
                    if second_children.is_marked():
                        continue

                    if self._can_left_merge_stes(children, second_children):
                        first_children_neighb = set(self._my_graph.neighbors(children))
                        for second_children_neigh in self._my_graph.neighbors(second_children):
                            if second_children_neigh in first_children_neighb:
                                continue
                            self.add_edge(children, second_children_neigh)

                        self.delete_node(second_children)

            for children in self._my_graph.neighbors(current_node):
                if not children.is_marked():
                    children.set_marked(True)
                    dq.appendleft(children)


    def right_merge(self):

        assert self.is_homogeneous(), "This function is working only for homogeneous case!"
        self.unmark_all_nodes()
        dq = deque()

        report_nodes = self.get_filtered_nodes(lambda n: n.is_report())
        for report_node in report_nodes:
            report_node.set_marked(True)
            dq.appendleft(report_node)


        while dq:

            current_node = dq.pop()

            for parent in list(self._my_graph.predecessors(current_node)):

                if parent.is_marked():
                    continue

                if not parent in self._node_dict: # this parent has been deleted
                    continue

                if parent == current_node:
                    continue # self loop

                for second_parent in list(self._my_graph.predecessors(current_node)):
                    if not second_parent in self._node_dict:
                        continue # deleted node
                    if second_parent == parent :
                        continue #comparing the same node
                    if second_parent.is_marked():
                        continue

                    if self._can_right_merge_stes(parent, second_parent):
                        first_children_parents = set(self._my_graph.predecessors(parent))
                        for second_children_neigh in self._my_graph.predecessors(second_parent):
                            if second_children_neigh in first_children_parents:
                                continue
                            self.add_edge(second_children_neigh, parent)

                        self.delete_node(second_parent)

            for parent in self._my_graph.predecessors(current_node):
                if not parent.is_marked():
                    parent.set_marked(True)
                    dq.appendleft(parent)




    def combine_symbol_sets(self):
        """
        this function combines the symbol sets of two stes with a same parent and a same child
        :return:
        """
        self.unmark_all_nodes()

        dq = deque()
        self._fake_root.set_marked(True)
        dq.appendleft(self._fake_root)

        while dq:
            current_node = dq.pop()

            for first_neighb_node in list(self._my_graph.neighbors(current_node)):
                if not first_neighb_node in self._node_dict:
                    continue #deleted node
                for sec_neighb_node in list(self._my_graph.neighbors(current_node)):
                    if not sec_neighb_node in self._node_dict:
                        continue # deleted node
                    if first_neighb_node == sec_neighb_node:
                        continue
                    if self._can_combine_symbol_set(fst_ste=first_neighb_node, sec_ste= sec_neighb_node):
                        print "miow"
                        for symbol in sec_neighb_node.get_symbols():
                            first_neighb_node.add_symbol(symbol)
                        self.delete_node(sec_neighb_node)

            for node in self._my_graph.neighbors(current_node):
                if not node.is_marked():
                    node.set_marked(True)
                    dq.appendleft(node)










    def _can_combine_symbol_set(self, fst_ste, sec_ste):
        if fst_ste.get_start() != sec_ste.get_start():
            return  False

        if fst_ste.is_report() != sec_ste.is_report():
            return  False

        if self.does_STE_has_self_loop(fst_ste) != self.does_STE_has_self_loop(sec_ste):
            return  False

        fst_ste_neighbors = list(self._my_graph.neighbors(fst_ste))

        if fst_ste in fst_ste_neighbors:
            fst_ste_neighbors.remove(fst_ste)

        if len(fst_ste_neighbors) != 1:
            return False

        sec_ste_neighbors = list(self._my_graph.neighbors(sec_ste))

        if sec_ste in sec_ste_neighbors:
            sec_ste_neighbors.remove(sec_ste)

        if len(sec_ste_neighbors) != 1:
            return False

        if sec_ste_neighbors != fst_ste_neighbors:
            return False

        fst_ste_predecessors = list(self._my_graph.predecessors(fst_ste))

        if fst_ste in fst_ste_predecessors:
            fst_ste_predecessors.remove(fst_ste)

        if len(fst_ste_predecessors) != 1:
            return False

        sec_ste_predecessors = list(self._my_graph.predecessors(sec_ste))

        if sec_ste in sec_ste_predecessors:
            sec_ste_predecessors.remove(sec_ste)

        if len(sec_ste_predecessors) != 1:
            return False

        if sec_ste_predecessors != fst_ste_predecessors:
            return False

        return True




    def _can_left_merge_stes(self,fst_ste, sec_ste):

        if fst_ste.get_start() != sec_ste.get_start():
            return  False

        if fst_ste.is_report() != sec_ste.is_report():
            return  False

        if not fst_ste.is_symbolset_a_subsetof_self_symbolset(sec_ste.get_symbols()) or not \
                sec_ste.is_symbolset_a_subsetof_self_symbolset(fst_ste.get_symbols()):
            return False

        if self.does_STE_has_self_loop(fst_ste) != self.does_STE_has_self_loop(sec_ste):
            return  False

        fst_ste_parents = set(self._my_graph.predecessors(fst_ste))
        sec_ste_parents = set(self._my_graph.predecessors(sec_ste))

        if sec_ste in fst_ste_parents and fst_ste in sec_ste_parents:
            fst_ste_parents.remove(sec_ste)
            sec_ste_parents.remove(fst_ste)

        try:
            fst_ste_parents.remove(fst_ste)
            sec_ste_parents.remove(sec_ste)
        except KeyError:
            pass

        return fst_ste_parents == sec_ste_parents

    def _can_right_merge_stes(self,fst_ste, sec_ste):

        if fst_ste.get_start() != sec_ste.get_start():
            return  False

        if fst_ste.is_report() != sec_ste.is_report():
            return  False

        if not fst_ste.is_symbolset_a_subsetof_self_symbolset(sec_ste.get_symbols()) or not \
                sec_ste.is_symbolset_a_subsetof_self_symbolset(fst_ste.get_symbols()):
            return False

        if self.does_STE_has_self_loop(fst_ste) != self.does_STE_has_self_loop(sec_ste):
            return  False

        fst_ste_children = set(self._my_graph.neighbors(fst_ste))
        sec_ste_children = set(self._my_graph.neighbors(sec_ste))

        if sec_ste in fst_ste_children and fst_ste in sec_ste_children:
            fst_ste_children.remove(sec_ste)
            sec_ste_children.remove(fst_ste)

        try:
            fst_ste_children.remove(fst_ste)
            sec_ste_children.remove(sec_ste)
        except KeyError:
            pass

        return fst_ste_children == sec_ste_children

    def get_STEs_out_degree(self):
        out_degree_list = []
        for node in self.get_nodes():
            if node.get_start() == StartType.fake_root:
                continue

            out_degree_list.append(self._my_graph.out_degree(node))

        return tuple(out_degree_list)
















def compare_strided(only_report, file_path,*automatas ):
    with open(file_path, 'rb') as f:
        file_content = f.read()

    byte_file_content = bytearray(file_content)

    strides = [sum(stride for stride in map(lambda x:x.get_stride_value(), automata)) for automata in automatas]
    max_stride = max(strides)
    gens =[]

    for idx, automata in enumerate(automatas):
        sum_stride_value = 0
        strided_gen =[]
        for strided_automata in automata:
            strided_gen.append(strided_automata.feed_input(byte_file_content, offset= sum_stride_value, jump = strides[idx]))
            sum_stride_value += strided_automata.get_stride_value()
        gens.append(strided_gen)


    stopped_automata = 0

    while stopped_automata == 0:

        first_automata_ste_set = None
        first_automata_report = False
        for idx, generator_set in enumerate(gens):

            try:
                for _ in range(max_stride/strides[idx]):

                    temp_result_set_list=[]
                    for gen in generator_set:
                        result_set, _ = next(gen)
                        temp_result_set_list.append(result_set)

                    intersection_set = set(temp_result_set_list[0])
                    for result_set in temp_result_set_list[1:]:
                        intersection_set = intersection_set.intersection(result_set)

                    for result_set in temp_result_set_list:
                        result_set.clear()
                        result_set.update(intersection_set)

                temp_is_report = False
                for ste in intersection_set:
                    if ste.is_report():
                        temp_is_report = True
                        break

                if idx == 0:
                    first_automata_ste_set = intersection_set
                    first_automata_report = temp_is_report
                else:

                    assert temp_is_report == first_automata_report
                    if not only_report:
                        assert first_automata_ste_set == intersection_set


            except StopIteration:
                stopped_automata +=1

    if stopped_automata == len(automatas):
        print "they are equal"
    else:
        print "something is wrong with the rate of consumption"























def compare_input(only_report, file_path, *automatas):
    gens = []
    result = [() for i in range(len(automatas))]
    max_stride = max([sv.get_stride_value() for sv in automatas])

    for a in automatas:
        g = a.feed_file(file_path)
        gens.append(g)

    try:
        while True:
            for idx_g, (g, automata) in enumerate(zip(gens,automatas)):
                assert max_stride % automata.get_stride_value() == 0
                for _ in range(max_stride / automata.get_stride_value()):
                    temp_active_states, temp_is_report = next(g)
                result[idx_g] =(temp_active_states, temp_is_report)

            for active_state, report_state in result[1:]:
                #print active_state, report_state, "* correct = ",result[0]
                assert report_state==result[0][1] # check report states
                if not only_report:
                    assert active_state == result[0][0] # check current states

    except StopIteration:
        print "They are equal"









