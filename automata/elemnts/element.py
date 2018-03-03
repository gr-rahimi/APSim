from enum import Enum

class StartType(Enum):
    non_start = 0
    start_of_data = 1
    all_input = 2
    unknown = 3 # for non homogeneous graphs, yes but will be determined
    fake_root = 4

class BaseElement(object):

    def __init__(self, is_report, is_marked = False, id = None):
        self._is_report = is_report
        self._marked = is_marked
        self._id = id

        '''
        when we make the automata form XML node,
        we need this to build the graph after parsing the whole xmldelete_adjacency_list
        '''
        self._adjacent_S_T_Es = []

    @classmethod
    def get_element_type(cls):
        return cls.__name__

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        return self._id == str(other)

    def __str__(self):
        return  self._id

    def __repr__(self):
        return self._id

    def is_marked(self):
        return self._marked

    def set_marked(self, m):
        self._marked = m

    def is_report(self):
        return self._is_report

    def get_id(self):
        return self._id


    def get_adjacency_list(self):
        return self._adjacent_S_T_Es

    def delete_adjacency_list(self):
        """
        To save memory and remove dual copy of structure
        :return:
        """
        del self._adjacent_S_T_Es

    @classmethod
    def from_xml_node_to_dict(cls, xml_node, attrib_dic):

        # find state id
        attrib_dic['id'] = xml_node.attrib['id']


    @classmethod
    def check_validity(cls, xml_node):
        assert 'id' in xml_node.attrib

    def is_S_T_E(self):
        return  False

    def is_OR(self):
        return False

    def is_special_element(self):
        return False



    def is_start(self):
        return  False # currently only STEs can be start elemnts

    def get_symbols(self):
        return tuple()

    def get_start(self):
        return StartType.non_start

