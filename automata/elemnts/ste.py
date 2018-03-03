import networkx as nx
import CPP.VASim as VASim
from element import BaseElement, StartType

class S_T_E(BaseElement):
    known_attributes = {'start', 'symbol-set', 'id'}
    aom_known_attributes = {'element'}
    report_on_match_attributes = {'reportcode'}



    def __init__(self, start_type, is_report, is_marked = False, id = None, symbol_set= set(), adjacent_S_T_E_s = []):
        super(S_T_E, self).__init__(is_report = is_report, is_marked = is_marked, id = id)
        self._start_type = start_type
        #assert _is_symbol_set_sorted(symbol_set), "symbol set should be sorted"
        self._symbol_set = symbol_set
        self._adjacent_S_T_Es = adjacent_S_T_E_s


    @classmethod
    def from_xml_node(cls, xml_node):
        assert cls.__name__ == S_T_E.__name__ # this function should not be called from child classes.
        cls.check_validity(xml_node)
        parameter_dict = {}
        super(S_T_E, cls).from_xml_node_to_dict(xml_node = xml_node, attrib_dic= parameter_dict)

        # find if start state
        if 'start' in xml_node.attrib:
            if xml_node.attrib['start'] == 'start-of-data':
                parameter_dict['start_type'] = StartType.start_of_data
            elif xml_node.attrib['start'] == 'all-input':
                parameter_dict['start_type'] = StartType.all_input
            else:
                raise RuntimeError('Unknown value for start attribute')
        else:
            parameter_dict['start_type'] = StartType.non_start

        # find symbol set
        assert 'symbol-set' in xml_node.attrib  # all STEs should have symbol set
        symbol_str = str(xml_node.attrib['symbol-set'])
        symbol_list = VASim.parseSymbolSet(symbol_str)
        symbol_list = symbol_list[::-1]  # reverse the string

        symbol_set = set()

        start = False
        start_idx = -1
        for idx_b, ch in enumerate(symbol_list):
            if ch == "1" and start == False:
                start = True
                start_idx = idx_b
            elif ch == "0" and start == True:
                start = False
                symbol_set.add((start_idx, idx_b - 1))
                start_idx = -1

        if start == True:  # this is necessary if the last iteration was 1
            symbol_set.add((start_idx, idx_b))
            start = False  # not necessary
            start_idx = -1  # not necessary

        parameter_dict['symbol_set'] = symbol_set

        adjacent_S_T_E_s = []
        is_report = False

        for child in xml_node:
            if child.tag == 'activate-on-match':
                S_T_E._check_validity_aom(child)
                assert 'element' in child.attrib
                adjacent_S_T_E_s.append(child.attrib['element'])
            elif child.tag == 'report-on-match':
                S_T_E._check_validity_rom(child)
                # TODO we should consider reportcode
                is_report = True
            elif child.tag == 'layout':
                continue  # Elaheh said it is not important
            else:
                raise RuntimeError('unsupported children of STE->' + child.tag)

        parameter_dict['adjacent_S_T_E_s'] = adjacent_S_T_E_s
        parameter_dict['is_report'] = is_report

        return S_T_E(**parameter_dict)

    @classmethod
    def from_xml_node_to_dict(cls, xml_node, attrib_dic):
        pass

    def get_start(self):
        return  self._start_type

    def set_start(self, start):
        self._start_type = start

    def is_start(self):
        return self._start_type == StartType.all_input or\
               self._start_type == StartType.start_of_data


    # check if the ste element has any attribute that I have not considered yet
    @classmethod
    def check_validity(cls, xml_node):
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(S_T_E.known_attributes)
        super(S_T_E, cls).check_validity(xml_node)

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
        #assert _is_symbol_set_sorted(symbols), "symbol set should be sorted"
        self._symbol_set = set(symbols)

    def add_symbol(self, symbol):
        self._symbol_set.add(symbol)

    def get_color(self):

        if self.get_start() == StartType.fake_root:
            return  (0,0,0,1) # Black
        elif self.get_start() == StartType.start_of_data:
            return (0,1,0,1) # Green
        elif self.get_start() == StartType.all_input:
            return (0,1,0,0.5) # Light Green
        elif self.is_report():
            return (0,0,1,1) # Blue
        elif self.get_start() == StartType.unknown:
            return (1,1,0,1)  # Yellow
        else:
            return (1,0,0,1) # Red

    def split_symbols(self):
        left_set = set()
        right_set = set()

        for left_symbol, right_symbol in self.get_symbols():
            left_set.add(left_symbol)
            right_set.add(right_symbol)

        return tuple(left_set), tuple(right_set)

    def can_accept(self, input, on_edge_symbol_set = None):
        """

        :param input: the input bytes
        :return: (acceptance True/False, is_reported True/False)
        """
        symbol_set = on_edge_symbol_set if on_edge_symbol_set else self._symbol_set
        for symbol in symbol_set:
            if self._check_interval(input, symbol):
                return (True, self.is_report())

        return (False, False)

    def _check_interval(self, input, symbol_set):
        assert len(symbol_set) == 2
        if len(input) ==1:
            left_margin , right_margin = symbol_set
            can_accept = left_margin<=input[0] and input[0]<= right_margin
            return  can_accept
        else:
            return self._check_interval(input[:len(input)/2],symbol_set[0]) and\
                   self._check_interval(input[len(input)/2:],symbol_set[1])

    def is_S_T_E(self):
        return True

    def is_symbolset_a_subsetof_self_symbolset(self, other_symbol_set):
        """

        :param other_symbol_set: the symbol set that is going to be checked
        :return: true if the input symbol set is a subset of the current subset
        """
        my_symbol_set = sorted(self.get_symbols())
        other_symbol_set = sorted(other_symbol_set)

        dim_size = _get_symbol_dim(other_symbol_set[0])
        assert dim_size == _get_symbol_dim(self.get_symbols()[0]), "dimesnsions should be equal"

        def cube_point_generator(interval):
            from itertools import product

            product_generator = product([0,1], repeat = dim_size)

            def point_calculator(x ,p , d):
                if d == 1:
                    return [x[p[0]]]
                else:
                    return point_calculator(x[0], p[0:len(p)/2], d/2) + (point_calculator(x[1], p[len(p)/2:], d/2))

            for p in product_generator:
                yield point_calculator(interval, p, dim_size)

        for interval in other_symbol_set:
            g = cube_point_generator(interval)
            for point in g:
                can_accept , _ =self.can_accept(point)
                if can_accept:
                    continue
                else:
                    return False
        return  True














def _is_symbol_set_sorted(symbol_set):
    if not symbol_set:  # fake root has None symbol set
        return  True
    for  prev_pt, next_pt in zip(symbol_set[:-1], symbol_set[1:]):
        if next_pt< prev_pt:
            return False
    return  True

def _get_symbol_dim(input_symbol):
    import collections
    if not isinstance(input_symbol, collections.Sequence):
        return 0.5
    else:
        return int(2 * _get_symbol_dim(input_symbol[0]))







