from .element import BaseElement, StartType
from . import ElementsType
import logging

class OrElement(BaseElement):
    known_attributes = {'id'}

    def __init__(self, id, is_report, **kwargs):
        super(OrElement, self).__init__(is_report=is_report, is_marked = False, id = id,
                                        start_type=StartType.non_start,**kwargs)

    @classmethod
    def from_xml_node(cls, xml_node, id):
        assert cls.__name__ == OrElement.__name__  # this function should not be called from child classes.
        cls.check_validity(xml_node)

        parameter_dict = {"id": id}

        super(OrElement, cls).from_xml_node_to_dict(xml_node=xml_node, attrib_dic=parameter_dict)

        is_report = False
        for child in xml_node:
            if child.tag == 'report-on-high':
                is_report = True
                report_code = child.attrib['reportcode']
            else:
                logging.error("Unsupported children of OR element: %s", child.tag)
                raise RuntimeError('unsupported children of OR element -> ' + child.tag)

        parameter_dict['is_report'] = is_report
        parameter_dict['report_residual'] = 0
        parameter_dict['report_code'] = report_code

        #TODO is this correct?
        parameter_dict['adjacent_nodes'] = [] # in current anml files, there is no fan out for or gates
        return OrElement(**parameter_dict)

    @classmethod
    def check_validity(cls, xml_node):
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(OrElement.known_attributes)
        super(OrElement, cls).check_validity(xml_node)

    @property
    def type(self):
        return ElementsType.OR

    @property
    def is_fake(self):
        return False











