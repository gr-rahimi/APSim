from  elemnts.ste import S_T_E, StartType
import networkx as nx
import re
from collections import  deque

class Automatanetwork(object):
    known_attributes = {'id'}
    fake_root = "fake_root"



    def __init__(self):
        self._start_STEs=[]
        self._has_modified = True
        self._my_graph = nx.DiGraph()
        self._my_graph.add_node(Automatanetwork.fake_root) # This is not areal node. It helps for simpler striding code
        self._node_dict={} # this dictionary helps retrieving nides using id(as a string)

        pass

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
            for dst_id in src_node.get_adjacency_list():
                dst_node = graph_ins._node_dict[dst_id]
                graph_ins.add_edge(src_node, dst_node)

            src_node.delete_adjacency_list()

        return graph_ins


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
        for n in self._my_graph.nodes():
            n.set_marked(False)

    def _rebuild_graph_properties(self):
        """
        This function rebuild all  peripheral properties of graph
        :return:
        """
        self._start_STEs=[]
        for node in self._my_graph.nodes:
            if node.get_start() == StartType.all_input or\
                            node.get_start() == StartType.start_of_data:
                self._start_STEs.append(node)

        self._has_modified =False


    def add_STE(self, to_add_STE):
        """

        :param to_add_STE: Add a ste to the graph
        :return:
        """
        assert to_add_STE.get_id() not in self._node_dict
        self._my_graph.add_node(to_add_STE)
        self._node_dict[to_add_STE.get_id()] = to_add_STE
        self._has_modified = True

        if to_add_STE.get_start() != StartType.non_start:
            self.add_edge(Automatanetwork.fake_root, to_add_STE) # add an esge from fake root to all start nodes


    def get_STE_by_id(self, id):
        """
        this function returns the STE instance using the id
        :param id: id of the ste
        :return:
        """
        return self._node_dict.get(id, None)


    def get_start_STEs(self):
        """
        get a list of start STEs.
        :return: list of start
        """
        if not self._has_modified:
            return tuple(self._start_STEs)

        else:
            self._rebuild_graph_properties()
            return tuple(self._start_STEs)


    def get_number_of_nodes(self):
        return len(self._my_graph)

    def add_edge(self, src, dest,**kwargs):
        self._my_graph.add_edge(src, dest, **kwargs)
        self._has_modified = True


    def get_single_stride_graph(self):
        """
        This function make a new graph with single stride
        :return: a graph with a single step stride
        """

        dq = deque()
        self.unmark_all_nodes()

        strided_graph = Automatanetwork()
        strided_graph_dict = {}
        strided_graph._id = self._id + "_S1"
        ###



        ###
        for n in self.get_start_STEs():
            #TODO We may not need to process start nodes that has already been processed

            temp_ste = S_T_E(start_type = n.get_start(), is_report= n.is_report(), is_marked=n.is_marked(), id = n.get_id(), symbol_set= n.get_symbols())
            strided_graph.add_STE(temp_ste)
            strided_graph_dict[n.get_id()] = temp_ste
            n.set_marked(True)
            dq.appendleft(n)


        while dq:

            current_ste = dq.pop()

            for l1_neigh in self._my_graph.neighbors(current_ste):
                for l2_neigh in self._my_graph.neighbors(l1_neigh):
                    if type(l2_neigh) == str:
                        print "Error"
                    if not l2_neigh.is_marked():
                        assert l2_neigh.get_start() ==StartType.non_start # it should not be a start node
                        temp_ste = S_T_E(start_type = l2_neigh.get_start(), is_report= l2_neigh.is_report(),
                                                    is_marked=False, id = l2_neigh.get_id(), symbol_set= None)
                        strided_graph.add_STE(temp_ste)
                        strided_graph_dict[l2_neigh.get_id()] = temp_ste
                        l2_neigh.set_marked(True)
                        dq.appendleft(l2_neigh)

                    strided_graph.add_edge(strided_graph_dict[current_ste.get_id()], strided_graph_dict[l2_neigh],
                                           label=((span1, span2) for span1 in l1_neigh.get_symbols() for span2 in l2_neigh.get_symbols))

        return strided_graph



    def make_homogenous(self):
        """

        :return:
        """
        pass























































