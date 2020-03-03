import xml.etree.cElementTree as ET
from .automata_network import Automatanetwork


def parse_anml_file(file_path):
    anml_tree = ET.parse(file_path)
    anml_root = anml_tree.getroot()

    #TODO make checking cleaner
    if anml_root.tag == 'automata-network':
        automata = Automatanetwork.from_xml(anml_root)
    else:
        automata_network = anml_tree.findall('automata-network')
        assert len(automata_network) == 1
        automata = Automatanetwork.from_xml(automata_network[0])

    return automata

def generate_anml_file(file, automata):
    """ generate an ANML network output file """

    # The supported start types
    start_types = {0: None, 1: 'start-of-data', 2: 'all-input'}

    # Header for an ANML file
    string = "<anml version=\"1.0\"  xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">\n"
    string += "\t<automata-network id=\"" + automata.id + "\">\n"

    # Iteratre through all nodes in the automata
    for node in automata.nodes:

        # Throw away the fake
        if not node.is_fake:

            # Generate ANML-compatible symbol set
            character_set =  expand_symbol_set(node._symbol_set)

            # Add an STE element with a given node id and character set
            string += "\t\t<state-transition-element id=\"" + str(node.id) + "\" " + \
                "symbol-set=\"" + character_set + "\" "

            # Add start type if a start STE
            start_type = start_types[node._start_type]
            if start_type:
                string += "start=\"" + start_type+ "\">\n"
            else:
                string += ">\n"
            
            # Add reporting + reportcode
            if node._is_report:
                string += "\t\t\t<report-on-match reportcode=\"" +\
                    str(node._report_code) + "\"/>\n"

            for n in automata.get_neighbors(node):
                string += "\t\t\t<activate-on-match element=\"" + \
                    str(automata.get_STE_by_id(n).id) + "\"/>\n"
            string += "\t\t</state-transition-element>\n"

    # Close the automata network
    string += '\t</automata-network>\n'
    string += '</anml>\n'

    # Write the generated string out to a file
    with open(file, 'w') as f:
        f.write(string)


def expand_symbol_set(symbol_set):
    """
        This function generates an ANML-compatible symbol-set string
    """

    character_set = ""

    # Go through each interval and combine symbols
    for ss in symbol_set._interval_set._list:
        lower = ss._left_pt[0]
        upper = ss._right_pt[0]

        # If the range is one symbol wide, add that one symbol
        if lower == upper:
           character_set += r"\x%02X" % int(lower)
        else:
            # If it's more than one symbol wide, represent a range
            character_set += r"\x%02X-\x%02X" % (int(lower), int(upper))

    # Return the generated character-set string
    return character_set