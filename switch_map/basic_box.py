from __future__ import  division
from enum import Enum
import copy

class SwitchAvailability(Enum):
    SwitchAlreadyUsed = -1
    NoSwitch = 0
    SwitchAvailable = 1



class BaseSwitch(object):

    def __init__(self, size):

        self._switch_array = [[0 for _ in range(size[1])] for _ in range(size[0])]
        self._size = size
        self._total_switches = 0
        self._used_switches = 0
        self._max_used_index = -1


    def get_size(self):
        return self._size

    def get_copy(self):
        return copy.deepcopy(self)

    def get_utilization(self):
        return self._used_switches / self._total_switches

    def set_switch(self, row, col):

        assert (row, col) < self._size, "trying to set wrong location in switch box"
        #print row, col ,self._size, len(self._switch_array), len(self._switch_array[row])
        if self._switch_array[row][col] == -1 or\
            self._switch_array[row][col] == 0:
            print "Trying to enable a non switch location. Chheck availability first"
            return False

        self._switch_array[row][col] = -1
        self._used_switches += 1
        self._max_used_index = max(self._max_used_index, row, col)
        return True

    def get_remaining_capacity(self):
        return min(self._size) -1 - self._max_used_index

    def unset_switch(self, row, col):
        assert False, "This function is not safe yet"
        assert (row, col) < self._size , "trying to set wrong location in switch box"
        if self._switch_array[row][col] == 0 or\
            self._switch_array[row][col] == 1:
            print "Trying to free a non switch location. Chheck availability first"
            return False

        self._switch_array[row][col] = 1
        self._used_switches -= 1
        return True

    def get_switch_info(self, row, col):
        if self._switch_array[row][col] == -1:
            return SwitchAvailability.SwitchAlreadyUsed
        elif self._switch_array[row][col] == 0:
            return SwitchAvailability.NoSwitch
        elif self._switch_array[row][col] == 1:
            return SwitchAvailability.SwitchAvailable
        else:
            assert False, "What kind of switch is this?"

    def _get_switches_statistics(self):
        total_switches = 0
        used_switches = 0
        max_index = -1
        for row_idx, row in enumerate(self._switch_array):
            for item_idx, item in enumerate(row):
                if item == 1 or item == -1:
                    total_switches += 1
                    if item == -1:
                        used_switches += 1
                        max_index = max(max_index, row_idx, item_idx)

        return total_switches, used_switches, max_index




    def set_raw_switch(self, switch_array):

        del self._switch_array

        self._switch_array = copy.deepcopy(switch_array)
        self._size = (len(switch_array), len(switch_array[0]))

        self._total_switches, self._used_switches, self._max_used_index = self._get_switches_statistics()





    def get_switch_array(self):
        return self._switch_array

    class RowIterator(object):
        def __init__(self, original_sb, row):
            self._original_sb = original_sb
            self._row = row

        def __getitem__(self, item):
            return self._original_sb.get_switch_info(self._row, item)



    def __getitem__(self, item):
        return BaseSwitch.Row_Iterator(self, item)

    def add_switch_box_layout(self, sb):
        offset = self._max_used_index + 1

        for row_idx, row in enumerate(sb):
            for item_idx, item in enumerate(row):
                if item == 1:
                    result = self.set_switch(offset + row_idx, offset + item_idx)
                    assert result






































