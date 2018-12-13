import CPP.VASim as VASim
from .element import BaseElement, StartType
from . import ElementsType
from itertools import chain, product, izip
import utility
from heapq import heappush, heappop
import point_comperator


CYTHON_CAN_ACCEPT_FUNC =  point_comperator.cython_can_accept

class ComparableMixin(object):
  def __eq__(self, other):
    return not self<other and not other<self
  def __ne__(self, other):
    return self<other or other<self
  def __gt__(self, other):
    return other<self
  def __ge__(self, other):
    return not self<other
  def __le__(self, other):
    return not other<self

class PackedInput(ComparableMixin):

    def __init__(self, alphabet_point):
        self._point = bytearray(alphabet_point)
        self._iter_idx = -1
        self._dim = len(self._point)

    def __iter__(self):
        return self._point.__iter__()

    def __len__(self):
        return len(self._point)

    def __lt__(self, other):
        return self.point < other.point

    def __str__(self):
        int_list=[int(i) for i in self.point]
        return str(int_list)

    def __getitem__(self, item):
        return self._point[item]

    @property
    def dim(self):
        return self._dim

    @property
    def point(self):
        return self._point


class PackedInterval(object):

    _pos_stat, _neg_stat = 0, 0
    def __init__(self, p1, p2):
        self._left_pt = p1
        self._right_pt = p2

    @property
    def left(self):
        return self._left_pt

    @property
    def right(self):
        return self._right_pt

    @property
    def dim(self): # dimension
        return self._left_pt.dim

    def get_corner(self, min_max_filter):
        assert len(min_max_filter) == self.dim
        return PackedInput(tuple((l if m == 0 else r for l, m, r in
                                  zip(self._left_pt, min_max_filter, self._right_pt))))

    def can_interval_accept(self, point):
        return (self.left <= point <= self.right)
        #assert point.dim == self.dim
        acceptance_result = CYTHON_CAN_ACCEPT_FUNC(self.left.point, point.point, self.right.point)
        assert le_res == acceptance_result

        # python_result = True
        # for l,p,r in izip(self.left, point, self.right):
        #     if l <= p <= r:
        #         continue
        #     else:
        #         python_result = False
        #         break
        #
        # assert python_result == acceptance_result


        if acceptance_result:
            PackedInterval._pos_stat+=1
        else:
            PackedInterval._neg_stat+=1

        if(PackedInterval._pos_stat + PackedInterval._neg_stat) % 100000 == 0:
            print "pos={},neg={}".format(PackedInterval._pos_stat,
                                         PackedInterval._neg_stat)
        return acceptance_result


    def __lt__(self, other):
        return self.left < other.left

    def __str__(self):
        return '[' + str(self.left) + ',' + str(self.right) + ']'

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def __ne__(self, other):
        return not self.__eq__(other)


class PackedIntervalSet(object):
    def __init__(self, packed_interval_set):
        self._interval_set = packed_interval_set

    def __iter__(self):
        return self._interval_set.__iter__()

    def __len__(self):
        return len(self._interval_set)

    def __getitem__(self, item):
        return self._interval_set[item]

    def is_splittable(self):
        for p in self._split_corner_gen():
            if not self.can_accept(p):
                return False
        return True

    def __eq__(self, other):
        return self._interval_set == other._interval_set

    def __ne__(self, other):
        return not self.__eq__(other)

    def _split_corner_gen(self):
        intervals = []
        for d in range(self.dim):

            ranges = [(ivl.left[d], ivl.right[d]) for ivl in self._interval_set]
            ranges = sorted(set(ranges))
            curr_time = 0
            processed_range_idx = -1
            finish_list = []
            last_covered_time = -1
            test_points = []

            while curr_time < ranges[-1][1] or finish_list:

                if len(finish_list) == 0:
                    processed_range_idx +=1
                    curr_time = ranges[processed_range_idx][0]
                    last_covered_time = ranges[processed_range_idx][1]
                    heappush(finish_list, ranges[processed_range_idx][1])
                    continue

                if processed_range_idx + 1 < len(ranges) and \
                    ranges[processed_range_idx+1][0] < last_covered_time and \
                        ranges[processed_range_idx+1][0] < finish_list[0]:
                    processed_range_idx +=1

                    if last_covered_time < ranges[processed_range_idx][1]:
                        last_covered_time = ranges[processed_range_idx][1]

                    heappush(finish_list, ranges[processed_range_idx][1])

                    test_points.append(int((curr_time + ranges[processed_range_idx][0])/2))
                    curr_time = ranges[processed_range_idx][0]
                else:
                    new_time = heappop(finish_list)

                    test_points.append(int((curr_time + new_time)/2))
                    curr_time = new_time

            test_points = sorted(set(test_points))
            intervals.append(test_points)


        for corner in product(*intervals):
            yield PackedInput(corner)




            #####
        #
        #     merged = [False] * len(ranges)
        #     for range_idx, range_val in enumerate(ranges[:-1]):
        #         if not merged[range_idx]:
        #             for cand_idx, cand_ivl in enumerate(ranges[range_idx+1:]):
        #                 if cand_ivl[0]<range_val[1]< cand_ivl[1]:
        #                     merged[range_idx+1+cand_idx] = True
        #                     range_val[1] = cand_ivl[1]
        #                 elif range_val[1] > cand_ivl[1]:
        #                     merged[range_idx + 1 + cand_idx] = True
        #                 else:
        #                     break
        #     flat_list=[]
        #     flat_list.extend([range_ivl[0] for range_idx, range_ivl in enumerate(ranges) if not merged[range_idx]])
        #     flat_list.extend([range_ivl[1] for range_idx, range_ivl in enumerate(ranges) if not merged[range_idx]])
        #     flat_list=sorted(set(flat_list)) # remove repeated elements
        #     intervals.append(flat_list)
        #
        # for corner in product(*intervals):
        #     yield PackedInput(corner)


    @classmethod
    def get_star(cls, dim):
        left_pt = PackedInput(tuple(0 for _ in range(dim)))
        right_pt = PackedInput(tuple(255 for _ in range(dim)))
        packed_interval = PackedInterval(p1=left_pt, p2=right_pt)
        return PackedIntervalSet([packed_interval])

    @property
    def dim(self):
        if not self._interval_set:
            #TODO log
            raise RuntimeWarning("empty symbol set does not have dimension yet")
        else:
            return self._interval_set[0].dim

    def add_interval(self, interval):
        import bisect
        assert isinstance(interval, PackedInterval), "argument should be an instance of PackedInterval"
        bisect.insort(self._interval_set, interval)
        return



        temp_symbol_set = PackedIntervalSet([interval])
        if not self.is_symbolset_a_subset(temp_symbol_set):
            # for ivl in self._interval_set:
            #     left_pt = ivl.left
            #     right_pt = ivl.right
            #     if interval.can_accept(left_pt) and interval.can_accept(right_pt):
            #         self._interval_set.remove(ivl)
            #         continue #check for next intervals

            #self._interval_set.append(interval)
            #self._interval_set.sort()
            bisect.insort(self._interval_set, interval)

        else:
            return

    def is_symbolset_a_subset(self, other_symbol_set):
        """
        :param other_symbol_set: the symbol set that is going to be checked
        :return: true if the input symbol set is a subset of the current subset
        """
        if len(self):
            assert self.dim == other_symbol_set.dim

        start_idx = 0
        for input_interval in other_symbol_set:
            left_point = input_interval.left
            right_point = input_interval.right
            can_accept = False
            for dst_interval in self._interval_set[start_idx:]:
                can_accept = dst_interval.can_interval_accept(left_point) and\
                             dst_interval.can_interval_accept(right_point)

                if can_accept:
                    break
                else:
                    start_idx += 1
                    continue

            if not can_accept:
                return False
        return True


    @classmethod
    def combine(cls, left_set, right_set):
        #TODO I think it is better to not use add_interval.
        #  Can we say that this new intervals are independent?

        assert left_set.dim == right_set.dim, "This condition is necessary"
        new_packed_list = []
        for l_int in left_set:
            for r_int in right_set:
                left_pt = PackedInput(tuple(l for l in chain(l_int.left, r_int.left)))
                right_pt = PackedInput(tuple (r for r in chain(l_int.right, r_int.right)))
                new_packed_list.append(PackedInterval(left_pt, right_pt))

        #temp_list = new_packed_list[:]
        #temp_ret = PackedIntervalSet(temp_list)
        #temp_ret.prone()

        return PackedIntervalSet(new_packed_list)

    def clone(self):
        return PackedIntervalSet([interval for interval in self._interval_set])

    def can_accept(self, input_pt):
        assert isinstance(input_pt, PackedInput)
        for intvl in self._interval_set:
            if intvl.can_interval_accept(input_pt):
                return True

        return False

    def __str__(self):

        if len(self._interval_set) == 0:
            return ''
        else:
            to_return_str = str(self._interval_set[0])
            for item in self._interval_set[1:]:
                to_return_str = to_return_str + ',' + str(item)

            return "*" + to_return_str + "*"

    def prone(self):
        """
        This function removes already covered intervals from the list
        :return:
            None
        """
        if self._interval_set:

            to_be_deleted = [] # keeps inexes of items that are going to be deleted

            current_master = 0 # this is the index of the item that others will be compared to

            for i in range(1, len(self._interval_set)):

                if self._interval_set[current_master].can_interval_accept(self._interval_set[i].left) and \
                        self._interval_set[current_master].can_interval_accept(self._interval_set[i].right):
                    to_be_deleted.append(i)

                elif self._interval_set[i].can_interval_accept(self._interval_set[current_master].left) and \
                        self._interval_set[i].can_interval_accept(self._interval_set[current_master].right):
                    to_be_deleted.append(current_master)
                    current_master = i

            for idx, item in enumerate(to_be_deleted):
                del self._interval_set[item-idx]



class S_T_E(BaseElement):
    known_attributes = {'start', 'symbol-set', 'id'}
    aom_known_attributes = {'element'}
    report_on_match_attributes = {'reportcode'}

    def __init__(self, start_type, is_report, is_marked , id , symbol_set, adjacent_S_T_E_s ,
                 report_residual , report_code, **kwargs ):
        super(S_T_E, self).__init__(is_report=is_report, is_marked=is_marked, id=id, start_type=start_type,
                                    adjacent_nodes=adjacent_S_T_E_s, report_residual=report_residual,
                                    report_code=report_code, **kwargs)
        #TODO rename to symbols_set
        self._symbol_set = symbol_set

    def get_mark_idx(self):
        return self._mark_idx

    @classmethod
    def from_xml_node(cls, xml_node , id):
        assert cls.__name__ == S_T_E.__name__ # this function should not be called from child classes.
        cls.check_validity(xml_node)
        parameter_dict = {"id": id}
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
        symbol_list=symbol_list[::-1]

        symbol_set = PackedIntervalSet([])

        start = False
        start_idx = -1
        for idx_b, ch in enumerate(symbol_list):
            if ch == "1" and start == False:
                start = True
                start_idx = idx_b
            elif ch == "0" and start == True:
                start = False
                left_pt = PackedInput((start_idx,))
                right_pt = PackedInput((idx_b-1,))
                interval = PackedInterval(left_pt, right_pt)
                symbol_set.add_interval(interval)
                start_idx = -1

        if start == True:  # this is necessary if the last iteration was 1
            left_pt = PackedInput((start_idx,))
            right_pt = PackedInput((idx_b,))
            interval = PackedInterval(left_pt, right_pt)
            symbol_set.add_interval(interval)


        parameter_dict['symbol_set'] = symbol_set

        adjacent_S_T_E_s = []
        is_report = False
        report_code = None

        for child in xml_node:
            if child.tag == 'activate-on-match':
                S_T_E._check_validity_aom(child)
                assert 'element' in child.attrib
                adjacent_S_T_E_s.append(child.attrib['element'])
            elif child.tag == 'report-on-match':
                S_T_E._check_validity_rom(child)
                is_report = True
                if "reportcode" in child.attrib:
                    report_code = child.attrib['reportcode']
            elif child.tag == 'layout':
                continue  # Elaheh said it is not important
            else:
                #TODO log here
                raise RuntimeError('unsupported children of STE->' + child.tag)

        parameter_dict['adjacent_S_T_E_s'] = adjacent_S_T_E_s
        parameter_dict['is_report'] = is_report
        parameter_dict['report_code'] = report_code
        parameter_dict['report_residual'] = 0 if is_report else -1 # we assume the anml files are single strided
        parameter_dict['is_marked']= False

        return S_T_E(**parameter_dict)

    def is_symbolset_splitable(self):
        return self.symbols.is_splittable()

    @classmethod
    def from_xml_node_to_dict(cls, xml_node, attrib_dic):
        pass


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

    @property
    def symbols(self):
        return self._symbol_set

    @symbols.setter
    def symbols(self, symbols):
        self._symbol_set = symbols

    def add_symbol(self, interval):
        self._symbols.add_interval(interval)



    def split_symbols(self):
        left_set = PackedIntervalSet(packed_interval_set=[])
        right_set = PackedIntervalSet(packed_interval_set=[])
        dim = self.symbols.dim
        #assert dim > 1 and dim%2 == 0
        for ivl in self.symbols:
            left_pt = ivl.left
            right_pt = ivl.right

            left_set.add_interval(PackedInterval(PackedInput(left_pt[:dim/2]), PackedInput(right_pt[:dim/2])))
            right_set.add_interval(PackedInterval(PackedInput(left_pt[dim/2:]), PackedInput(right_pt[dim/2:])))

        return left_set, right_set

    @property
    def type(self):
        return ElementsType.STE






























