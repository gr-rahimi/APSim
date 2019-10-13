from enum import Enum
from abc import ABCMeta, abstractproperty
from . import ElementsType

class StartType(Enum):
    non_start = 0
    start_of_data = 1
    all_input = 2
    unknown = 3 # for non homogeneous graphs, yes but will be determined
    fake_root = 4



class BaseElement(object):
    known_attributes = {'id'}

    __metaclass__ = ABCMeta

    def __init__(self, is_report, is_marked , id, start_type,
                 report_residual, adjacent_nodes, report_code, original_id=None ):
        self._is_report = is_report
        self._marked = is_marked
        self._id = id
        self._start_type = start_type

        '''
        when we make the automata form XML node,
        we need this to build the graph after parsing the whole xmldelete_adjacency_list
        '''
        #TODO renme this from STE to something general
        self._adjacent_S_T_Es = adjacent_nodes
        assert not is_report or report_residual >= 0, "for report states, report residual should be a valid number"
        self._report_residual = report_residual
        self._report_code = report_code
        self.original_id = original_id

        #TODO is this property still necessary? better to be removed or move to a base class
        #this is used for labeling
        self._mark_idx = -1


    @property
    def mark_index(self):
        return self._mark_idx

    @mark_index.setter
    def mark_index(self, idx):
        self._mark_idx = idx

    @abstractproperty
    def type(self):
        pass
    @abstractproperty
    def is_fake(self):
        pass

    def __hash__(self):
        return hash(self.id)

    #TODO should we check the instance of other? Now only networkx can use it
    def __eq__(self, other):
        return self.id == other

    def __str__(self):
        if self.report:
            return str(self.id)+','+\
                   str(self.report_code)+','+\
                   str(self.report_residual)
        else:
            return str(self.id)

    def __repr__(self):
        return str(self.id)

    @property
    def marked(self):
        return self._marked

    @marked.setter
    def marked(self, marked):
        self._marked = marked

    #TODO rename to is_report
    @property
    def report(self):
        return self._is_report

    @report.setter
    def report(self, is_report):
        self._is_report= is_report

    @property
    def id(self):
        return self._id

    @property
    def start_type(self):
        return self._start_type

    @start_type.setter
    def start_type(self, start_type):
        self._start_type = start_type

    @property
    def report_residual(self):
        return self._report_residual

    @report_residual.setter
    def report_residual(self, report_residual):
        self._report_residual = report_residual

    @property
    def report_code(self):
        return self._report_code

    @report_code.setter
    def report_code(self, report_code):
        self._report_code = report_code

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
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(cls.known_attributes)
        # find state id
        attrib_dic['original_id'] = xml_node.attrib['id']


    @classmethod
    def check_validity(cls, xml_node):
        #TODO this need to be fixed. It brings ste attr that is not subset
        return True
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(BaseElement.known_attributes)

    def is_special_element(self):
        if self.type == ElementsType.OR:
            return True
        else:
            return False

    def is_start(self):
        return self.start_type == StartType.all_input or\
               self.start_type == StartType.start_of_data

    def get_color(self):

        if self.start_type == StartType.fake_root:
            return  (0,0,0,1) # Black
        elif self.start_type == StartType.start_of_data:
            return (0,1,0,1) # Green
        elif self.start_type == StartType.all_input:
            return (0,1,0,0.5) # Light Green
        elif self.report:
            return (0,0,1,1) # Blue
        elif self.start_type == StartType.unknown:
            return (1,1,0,1)  # Yellow
        else:
            return (1,0,0,1) # Red


class FakeRoot(BaseElement):
    fake_root_id = 0

    def __init__(self):
        super(FakeRoot, self).__init__(is_report=False, is_marked=False, id=FakeRoot.fake_root_id,
                                       start_type=StartType.fake_root, report_residual=None,
                                       adjacent_nodes=None,report_code=None)

    @property
    def type(self):
        return ElementsType.FAKE_ROOT

    @property
    def is_fake(self):
        return True





