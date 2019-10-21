import CPP.VASim as VASim
from .element import BaseElement, StartType
from . import ElementsType
from itertools import chain, product, izip
from heapq import heappush, heappop
#import point_comperator
import bisect
from sortedcontainers import SortedSet

#CYTHON_CAN_ACCEPT_FUNC =  point_comperator.cython_can_accept

symbol_type = 1 #cl


def get_Symbol_type(is_naive = False):
    if symbol_type == 1:
        return PackedIntervalSet
    else:
        if is_naive is not True:
            return GeneratorSymbolSet
        else:
            return NaiveSymbolSet


class GeneratorSymbolSet(object):
    def __init__(self, initial_list):
        self._generator_list = initial_list

    def __iter__(self):
        for left_l, right_l in self._internal_iter():
            left_pt = PackedInput(left_pt)
            right_pt = PackedInput(right_pt)

            yield PackedInterval(left_pt, right_pt)

    def _internal_iter(self):

        left_result , right_result = [], []

        def my_generator():
            #varibales_indexmap
            fhpl = 0
            shpl = 1
            fhpr = 2
            shpr = 3
            fhi = 4
            shi = 5
            shg = 6

            def get_next_tuple(first_half_pt_left, first_half_pt_right,
                               first_half_i, sec_half_i, sec_half_gen):
                try:
                    sec_half_pt_left, sec_half_pt_right = next(sec_half_i)
                    return (first_half_pt_left, sec_half_pt_left, first_half_pt_right, sec_half_pt_right, first_half_i,
                               sec_half_i, sec_half_gen)
                except StopIteration:
                        try:
                            first_half_pt_left, first_half_pt_right = next(first_half_i)
                        except StopIteration:
                                return None
                        sec_half_i = iter(sec_half_gen)
                        sec_half_pt_left, sec_half_pt_right = next(sec_half_i)
                        return (first_half_pt_left, sec_half_pt_left, first_half_pt_right, sec_half_pt_right,
                                    first_half_i, sec_half_i, sec_half_gen)

            i_list = []
            for first_half_gen, sec_half_gen in self._generator_list:
                first_half_i = iter(first_half_gen)
                sec_half_i = iter(sec_half_gen)
                first_half_pt_left, first_half_pt_right = next(first_half_i)
                sec_half_pt_left, sec_half_pt_right = next(sec_half_i)
                bisect.insort((first_half_pt_left,
                               sec_half_pt_left,
                               first_half_pt_right,
                               sec_half_pt_right,
                               first_half_i,
                               sec_half_i,
                               sec_half_gen))

            current_hit = i_list.pop(0)

            next_hit = get_next_tuple(current_hit[fhpl], current_hit[shpl],
                                      current_hit[fhpr], current_hit[shi], current_hit[shg])
            if next_hit is not None:
                i_list.append(next_hit)

            while i_list:

                next_tuple = i_list.pop(0)
                next_hit = get_next_tuple(next_tuple[fhpl], next_tuple[shpl], next_tuple[fhpr], next_tuple[shi],
                                          next_tuple[shg])
                if next_hit is not None:
                    i_list.append(next_hit)

                #check if the new point is bigger
                assert first_half_pt_left <= next_hit[fhpl] and sec_half_pt_left <= next_hit[shpl]

                if first_half_pt_left == next_hit[0] and sec_half_pt_left == next_hit[1]:
                    current_hit = next_hit
                    continue

                l = len(current_hit[fhpl])

                left_result[:l] = current_hit[fhpl]
                left_result[l:] = current_hit[shpl]

                right_result[:l] = current_hit[fhpr]
                right_result[l:] = current_hit[shpr]

                yield left_result, right_result

    @classmethod
    def combine(cls, left_set, right_set):
        pass


class NaiveSymbolSet(GeneratorSymbolSet):
    def __init__(self, intervals):
        '''

        :param intervals:intervals as list of tuples. [(left1,right1), (left, right2), ...]
        '''
        GeneratorSymbolSet.__init__(self, None)
        #TODO we assume here that the initial list does not need to be proned
        self._naive_list = [(p.left.point[0], p.right.point[0]) for p in intervals].sort()

    def add_interval(self, point):

        bisect.insort(self._naive_list, (point.left[0], point.right[0]))

    def __iter__(self):
        def my_generator():
            for left_pt, right_pt in self._naive_list:
                yield PackedInterval(PackedInterval(PackedInput((left_pt, )), PackedInput((right_pt, ))))

    def _internal_iter(self):
        def my_generator():
            left_list, right_list = [None], [None]
            for left_pt, right_pt in self._naive_list:
                left_list[0] = left_pt
                right_list[0] = right_pt
                yield left_list, right_list

        return my_generator()


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
        self._point = alphabet_point
        self._iter_idx = -1
        self._dim = len(self._point)

    def __iter__(self):
        return self._point.__iter__()

    def __len__(self):
        return self._dim

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

    def __hash__(self):
        return hash(self._point)


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

    def __lt__(self, other):
        return self.left < other.left

    def __str__(self):
        return '[' + str(self.left) + ',' + str(self.right) + ']'

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right

    def __ne__(self, other):
        return not self.__eq__(other)

    def is_interval_star(self, max_val):
        '''
        check if an 1d interval is star
        :param max_val: maximum value of star. for 8 bit, it is 255
        :return: True if it is star
        '''

        for d in range(self.dim):
            if self.left[d] == 0 and self.right[d] == max_val:
                continue
            else:
                return False

        return True

    def __hash__(self):
        return hash((self.left, self.right))




class PackedIntervalSet(object):
    def __init__(self, packed_interval_set):
        self.__mutable = True
        self._set_interval_set(packed_interval_set)
        self._set_points = None

    def _set_interval_set(self, packed_interval_set):
        if self.mutable:
            self._interval_set = SortedSet(packed_interval_set)
        else:
            new_sorted_set = SortedSet(packed_interval_set)
            if new_sorted_set[0].left < self._interval_set[0].left:
                raise RuntimeError()
            else:
                self._interval_set = new_sorted_set


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

    def __hash__(self):
        if self.__mutable:
            raise RuntimeError()
        else:
            return hash(self._interval_set[0].left)

    def __eq__(self, other):
        if len(self._interval_set)!= len(other._interval_set):
            return False
        for s,o in zip(self._interval_set, other._interval_set):
            if s!=o:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def initialize_set_points(self):
        self._set_points = set()

        for pt in self.points:
            self._set_points.add(pt)

    def _split_corner_gen(self):
        intervals = []
        for d in range(self.dim):

            ranges = [(ivl.left[d], ivl.right[d]) for ivl in self._interval_set]
            ranges = sorted(set(ranges))
            curr_time = -1
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


    @classmethod
    def get_star(cls, dim, max_val):
        left_pt = PackedInput(tuple(0 for _ in range(dim)))
        right_pt = PackedInput(tuple(max_val for _ in range(dim)))
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
        if self.__mutable:
            #import bisect
            assert isinstance(interval, PackedInterval), "argument should be an instance of PackedInterval"
            #bisect.insort(self._interval_set, interval)
            self._interval_set.add(interval)
            return
        else:
            if interval.left < self._interval_set[0].left:
                raise RuntimeError()
            else:
                assert isinstance(interval, PackedInterval), "argument should be an instance of PackedInterval"
                # bisect.insort(self._interval_set, interval)
                self._interval_set.add(interval)

    @property
    def mutable(self):
        return self.__mutable

    @mutable.setter
    def mutable(self, mutable_val):
        self.__mutable = mutable_val

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

        # temp_list = new_packed_list[:]
        # temp_ret = PackedIntervalSet(temp_list)
        # temp_ret.prone()

        # assert len(temp_ret) == len (new_packed_list)

        return PackedIntervalSet(new_packed_list)

    def clone(self):
        return PackedIntervalSet(self._interval_set)

    def can_accept(self, input_pt, fast_mode=False):
        if fast_mode is False:
            assert isinstance(input_pt, PackedInput)
            for intvl in self._interval_set:
                if intvl.can_interval_accept(input_pt):
                    return True
            return False
        else:
            return input_pt.point in self._set_points


    def __repr__(self):
        pass

    def __str__(self):

        if len(self._interval_set) == 0:
            return ''
        else:
            to_return_str = str(self._interval_set[0])
            for item in self._interval_set[1:]:
                to_return_str = to_return_str + ',' + str(item)

            return "*" + to_return_str + "*"

    def merge(self):
        if self.__mutable is False:
            raise RuntimeError()

        if self.dim > 1:
            return
        new_sym_set = []

        # we know that the current symbol is sorted
        curr_left = self._interval_set[0].left[0]
        curr_right = self._interval_set[0].right[0]

        for ivl in self._interval_set[1:]:
            if ivl.left[0] - 1 <= curr_right:
                curr_right = ivl.right[0]
            else:
                new_sym_set.append(PackedInterval(PackedInput((curr_left,)), PackedInput((curr_right,))))
                curr_left, curr_right = ivl.left[0], ivl.right[0]

        new_sym_set.append(PackedInterval(PackedInput((curr_left,)), PackedInput((curr_right,))))
        self._set_interval_set(new_sym_set)

    def prone(self):
        """
        This function removes already covered intervals from the list
        :return:
            None
        """

        if self.__mutable is False:
            raise RuntimeError()

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
                else:
                    current_master = i
            to_be_deleted.sort()
            for idx, item in enumerate(to_be_deleted):

                del self._interval_set[item-idx]

    @property
    def points(self):
        for interval in self._interval_set:
            counter = [[interval.left[i], interval.left[i], interval.right[i]] for i in range(self.dim)]
            while True:
                yield tuple(counter[i][1] for i in range(self.dim))

                d = self.dim - 1

                while d >= 0:
                    if counter[d][1] == counter[d][2]:
                        counter[d][1] = counter[d][0]
                        d -= 1
                        continue
                    else:
                        counter[d][1] += 1
                        break

                if d == -1:
                    break

                # yield tuple(c[1] for c in counter)

        return

    def points_on_dim(self, d, array_len):
        '''
        this function iterate over numbers in a specific dimnesion
        :param d: the dimension iteratating will happen
        :param array_len: length of the array that this vector will be places
        :return: iterator
        '''
        assert d < self.dim, 'this access is out of range'
        out_array = [0] * array_len

        for ivl in self._interval_set:
            left_d = ivl.left[d]
            right_d = ivl.right[d]

            out_array[left_d: right_d + 1] = [1] * (right_d + 1 - left_d)

        return tuple(out_array)

    def clear(self):
        if self.__mutable is False:
            raise RuntimeError()

        del self._interval_set[:]


    def is_star(self, max_val):
        '''
        check if the symbol set is star
        :Param maximum acceptable value
        :return: True if it is
        '''
        for ivl in self:
            if ivl.is_interval_star(max_val=max_val):
                return True
        return False

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
        #inplace symbol replacer
        self.symbols.clear()
        for s in symbols:
            self._symbol_set.add_interval(s)

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

    @property
    def is_fake(self):
        return False


