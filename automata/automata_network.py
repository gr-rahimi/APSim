from  elemnts.ste import S_T_E, StartType
import networkx as nx
import matplotlib.pyplot as plt
from collections import  deque
from sets import Set

class Automatanetwork(object):
    known_attributes = {'id','name'}
    _fake_root = "fake_root"



    def __init__(self, is_homogenous = True, stride = 1):
        #self._start_STEs=[]
        self._has_modified = True
        self._my_graph = nx.MultiDiGraph()
        self._node_dict={} # this dictionary helps retrieving nides using id(as a string)
        self._is_homogeneous = is_homogenous
        self._fake_root = S_T_E(start_type = StartType.fake_root, is_report=False,
                                is_marked=False, id=Automatanetwork._fake_root,
                                symbol_set=None)
        self.add_STE(self._fake_root)  # This is not areal node. It helps for simpler striding code
        self._stride = stride


    @classmethod
    def from_xml(cls, xml_node):

        Automatanetwork._check_validity(xml_node)
        graph_ins = cls()
        graph_ins._id = xml_node.attrib['id']

        for child in xml_node:
            if child.tag == 'state-transition-element':
                ste = S_T_E.from_xml_node(child)
                graph_ins.add_STE(ste)
            else:
                raise RuntimeError('unsupported child of automata-network')


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


    def add_STE(self, to_add_STE, connect_to_fake_root = True):

        """
        :param to_add_STE: Add a ste to the graph
        :return:
        """
        assert to_add_STE.get_id() not in self._node_dict
        self._my_graph.add_node(to_add_STE)
        self._node_dict[to_add_STE.get_id()] = to_add_STE
        self._has_modified = True

        if self.is_homogeneous() and to_add_STE.is_start() and connect_to_fake_root: # only for homogenous graphs
            self.add_edge(Automatanetwork._fake_root, to_add_STE) # add an esge from fake root to all start nodes


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
        return len(self._my_graph)

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

        dq = deque()
        self.unmark_all_nodes()
        strided_graph = Automatanetwork(is_homogenous= False, stride= self.get_stride_value()*2)
        strided_graph._id = self._id + "_S1"
        self._fake_root.set_marked(True)
        ###
        dq.appendleft(self._fake_root)

        while dq:

            current_ste = dq.pop()

            for l1_neigh in self._my_graph.neighbors(current_ste):
                for l2_neigh in self._my_graph.neighbors(l1_neigh):
                    if not l2_neigh.is_marked():

                        temp_ste = S_T_E(start_type = StartType.unknown, is_report= l2_neigh.is_report(),
                                                    is_marked=False, id = l2_neigh.get_id(), symbol_set= None)
                        strided_graph.add_STE(temp_ste)
                        l2_neigh.set_marked(True)
                        dq.appendleft(l2_neigh)

                    strided_graph.add_edge(strided_graph.get_STE_by_id(current_ste.get_id()), strided_graph.get_STE_by_id(l2_neigh.get_id()),
                                           label=tuple((span1, span2) for span1 in l1_neigh.get_symbols() for span2 in l2_neigh.get_symbols()),
                                           start_type = l1_neigh.get_start() if current_ste.get_start() == StartType.fake_root else StartType.non_start)

        #strided_graph.draw_graph("mid_graph", draw_edge_label= True)
        strided_graph.make_homogenous()
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
                    src_dict_non_start.setdefault(edge[0], Set()).update(label)
                elif start_type == StartType.start_of_data:
                    src_dict_start_of_data.setdefault(edge[0], Set()).update(label)
                elif start_type == StartType.all_input:
                    src_dict_all_start.setdefault(edge[0], Set()).update(label)
                else:
                    assert False # It should not happen

            self._make_homogenous_node(curr_node = current_ste, connectivity_dic = src_dict_all_start,
                                       start_type = StartType.all_input )
            self._make_homogenous_node(curr_node=current_ste, connectivity_dic=src_dict_non_start,
                                       start_type=StartType.non_start)
            self._make_homogenous_node(curr_node=current_ste, connectivity_dic=src_dict_start_of_data,
                                       start_type=StartType.start_of_data)
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
        nx.draw(self._my_graph, pos, node_size = 2, width = 0.1, arrowsize = 2, node_color= node_color)

        if draw_edge_label: # draw with edge lable
            edge_lables = nx.get_edge_attributes(self._my_graph, 'label')

            nx.draw_networkx_edge_labels(self._my_graph, pos, node_size = 2, width = 0.1, arrowsize = 2,
                                         node_color= node_color, font_size= 3 )

        plt.savefig(file_name, dpi=1000)
        plt.clf()




    def _make_homogenous_node(self, curr_node, connectivity_dic, start_type):

        new_nodes = []
        self_loop_node = None

        for neighb, on_edge_char_set in connectivity_dic.iteritems():
            new_node = S_T_E(start_type=start_type, is_report= curr_node.is_report(), is_marked= True,
                             id = neighb.get_id()+"_"+curr_node.get_id()+"_" +str(on_edge_char_set), symbol_set = on_edge_char_set)
            self.add_STE(new_node)
            if curr_node == neighb:
                assert start_type == StartType.non_start
                self_loop_node =new_node # we need to make an edge from every other node to this node
            else:
                new_nodes.append(new_node)
                self.add_edge(neighb, new_node, label = new_node.get_symbols(), start_type = new_node.get_start())

            out_edges = self._my_graph.out_edges(curr_node, data = True, keys = False)
            for edge in out_edges:
                self.add_edge(new_node, edge[1], label = edge[2]['label'], start_type = edge[2]['start_type'])

        if self_loop_node:
            for node in new_nodes:
                self.add_edge(node, self_loop_node, label = self_loop_node.get_symbols(), start_type = self_loop_node.get_start())
            self.add_edge(self_loop_node,self_loop_node, label = self_loop_node.get_symbols(), start_type = self_loop_node.get_start())


    def print_summary(self):
        print "Number of nodes: ", self.get_number_of_nodes()
        print "Number of start nodes", self.get_number_of_start_nodes()
        print "Number of report nodes", self.get_number_of_report_nodes()



    def split(self):
        """
        split the current automata and rturn both of them
        :return:
        """

        left_automata = Automatanetwork(is_homogenous= self.is_homogeneous(), stride= self.get_stride_value()/2)
        right_automata = Automatanetwork(is_homogenous=self.is_homogeneous(), stride=self.get_stride_value()/2)
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
                left_automata.add_STE(left_ste)

                right_ste = S_T_E( start_type = neighb.get_start(), is_report = neighb.is_report(),
                                 is_marked=True, id=neighb.get_id(), symbol_set=right_symbols)
                right_automata.add_STE(right_ste)

                self._split_node(neighb, left_automata= left_automata, right_automata=right_automata)

            left_ste_src = left_automata.get_STE_by_id(node.get_id())
            left_ste_dst = left_automata.get_STE_by_id(neighb.get_id())
            left_automata.add_edge(left_ste_src, left_ste_dst, label = left_ste_dst.get_symbols(), start_type = left_ste_dst.get_start())

            right_ste_src = right_automata.get_STE_by_id(node.get_id())
            right_ste_dst = right_automata.get_STE_by_id(neighb.get_id())
            right_automata.add_edge(right_ste_src, right_ste_dst, label=left_ste_dst.get_symbols(),
                                   start_type=left_ste_dst.get_start())


    def feed_file(self, input_file):
        """

        :param input_file: file to be feed to the input
        :return:
        """
        active_states = Set([self._fake_root])

        all_start_states = [all_start_neighb for all_start_neighb in self._my_graph.neighbors(self._fake_root)
                            if all_start_neighb.get_start() == StartType.all_input]
        with open(input_file, 'rb') as f:
            for input in iter(lambda: f.read(self.get_stride_value()),b''):
                new_active_states = Set()
                for act_st in active_states:
                    for neighb in self._my_graph.neighbors(act_st):
                        if neighb.can_accept(input=input):
                            new_active_states.add(neighb)

                for state in all_start_states:
                    if state.can_accept(input):
                        new_active_states.add(state)

                active_states = new_active_states
                yield active_states



    def get_connected_components_size(self):
        undirected_graph = self._my_graph.copy()
        undirected_graph= undirected_graph.to_undirected()
        undirected_graph.remove_node("fake_root")
        return tuple(len(g) for g in sorted(nx.connected_components(undirected_graph), key=len, reverse=True))





