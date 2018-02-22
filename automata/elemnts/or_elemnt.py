from element import BaseElement

class OrElement(BaseElement):
    known_attributes = { 'id'}

    def __init__(self, id, is_report):
        super(OrElement, self).__init__(is_report, is_marked = False, id = id)


    @classmethod
    def from_xml_node(cls, xml_node):
        assert cls.__name__ == OrElement.__name__  # this function should not be called from child classes.
        cls.check_validity(xml_node)

        parameter_dict = {}
        super(OrElement, cls).from_xml_node_to_dict(xml_node=xml_node, attrib_dic=parameter_dict)

        is_report = False
        for child in xml_node:
            if child.tag == 'report-on-high':
                #TODO report code should be handeled
                is_report = True
            else:
                raise RuntimeError('unsupported children of STE -> ' + child.tag)

        parameter_dict['is_report'] = is_report
        return OrElement(**parameter_dict)



    # check if the ste element has any attribute that I have not considered yet
    @classmethod
    def check_validity(cls, xml_node):
        attr_set = set(xml_node.attrib)
        assert attr_set.issubset(OrElement.known_attributes)
        super(OrElement, cls).check_validity(xml_node)

    def is_OR(self):
        return True











