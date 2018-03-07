import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle

automata = atma.parse_anml_file(anml_path[AnmalZoo.EntityResolution])
print "Finished processing from anml file. Here is the summary"

automata.remove_ors()


orig_automatas = automata.get_connected_components_as_automatas()
atm = orig_automatas[0]
print atm.max_STE_in_degree()
atm.set_max_fan_in(8)
print atm.max_STE_in_degree()

orig_automatas = automata.get_connected_components_as_automatas()

compare_input(True, input_path[AnmalZoo.EntityResolution], orig_automatas[0], atm)



















