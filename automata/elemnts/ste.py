import networkx as nx
import CPP.VASim as VASim
from sets import Set
from enum import Enum

class StartType(Enum):
    non_start = 0
    start_of_data = 1
    all_input = 2
    unknown = 3 # for non homogeneous graphs, yes but will be determined
    fake_root = 4

class S_T_E(object):
    known_attributes = {'start', 'symbol-set', 'id'}
    aom_known_attributes = {'element'}
    report_on_match_attributes = {'reportcode'}



    def __init__(self, start_type, is_report, is_marked = False, id = None, symbol_set= None):
        self._start_type = start_type
        self._is_report = is_report
        self._marked = is_marked
        self._id = id
        self._symbol_set = Set(symbol_set)



    @classmethod
    def from_xml_node(cls, xml_node):
        S_T_E._check_validity(xml_node)

        new_ste = cls(start_type = StartType.non_start, is_report = False)

        '''
        when we make the automata form XML node,
        we need this to build the graph after parsing the whole xmldelete_adjacency_list
        '''
        new_ste._adjacent_S_T_Es = []

        # find state id
        assert 'id' in xml_node.attrib
        new_ste._id = xml_node.attrib['id']


        # find if start state
        if 'start' in xml_node.attrib:
            if xml_node.attrib['start'] == 'start-of-data':
                new_ste._start_type = StartType.start_of_data
            elif xml_node.attrib['start'] == 'all-input':
                new_ste._start_type = StartType.all_input
            else:
                raise RuntimeError('Unknown value for start attribute')


        # find symbol set
        assert 'symbol-set' in xml_node.attrib # all STEs should have symbol set
        symbol_set = VASim.parseSymbolSet(str(xml_node.attrib['symbol-set']))
        start = False
        start_idx = -1
        for idx_b, ch in enumerate(symbol_set):
            if ch == "1" and start == False:
                start = True
                start_idx = idx_b
            elif ch == "0" and start == True:
                start = False
                new_ste._symbol_set.add((start_idx, idx_b - 1))
                start_idx = -1

        if start == True: # this is necessary if the last iteration was 1
            new_ste._symbol_set.add((start_idx, idx_b))
            start = False # not necessary
            start_idx = -1 # not necessary


        for child in xml_node:
            if child.tag == 'activate-on-match':
                S_T_E._check_validity_aom(child)
                assert 'element' in child.attrib
                new_ste._adjacent_S_T_Es.append(child.attrib['element'])
            elif child.tag == 'report-on-match':
                S_T_E._check_validity_rom(child)
                #TODO we should consider reportcode
                new_ste._is_report = True
            else:
                raise RuntimeError('unsupported children of STE')

        return new_ste


    def get_adjacency_list(self):
        return self._adjacent_S_T_Es

    def delete_adjacency_list(self):
        """
        To save memory and remove dual copy of structure
        :return:
        """
        del self._adjacent_S_T_Es


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
    def get_start(self):
        return  self._start_type
    def is_report(self):
        return self._is_report
    def get_id(self):
        return self._id
    def is_start(self):
        return self._start_type == StartType.all_input or\
               self._start_type == StartType.start_of_data


    # check if the ste element has any attribute that I have not considered yet
    @staticmethod
    def _check_validity(xml_node):
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(S_T_E.known_attributes)

    # check if the active-on-match has any new attribute
    @staticmethod
    def _check_validity_aom(xml_node):
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(S_T_E.aom_known_attributes)

    # check if the report-on-match has any new attribute
    @staticmethod
    def _check_validity_rom(xml_node):
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(S_T_E.report_on_match_attributes)

    def get_symbols(self):
        return tuple(self._symbol_set)

    def set_symbols(self, symbols):
        self._symbol_set = Set(symbols)

    def add_symbol(self, symbol):
        self._symbol_set.add(symbol)







