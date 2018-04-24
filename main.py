import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata


automata1 = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
automata1.remove_ors()

automata2 = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
automata2.remove_ors()

atma.compare_input(True, input_path[AnmalZoo.Snort], automata1, automata2)





