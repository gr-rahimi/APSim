import xml.etree.cElementTree as ET
from automata_network import  Automatanetwork

def parse_anml_file(file_path):

    anml_tree = ET.parse(file_path)
    anml_root = anml_tree.getroot()

    #TODO: support more generic anml files
    if len(anml_root) > 1:
        raise RuntimeError("anml file has more than one automata-network tag. Not supported yet!!!")


    for automata_network in anml_root:
        automata = Automatanetwork.from_xml(automata_network)
        return  automata












