from __future__ import division
import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
import os, shutil
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import utility
from automata.elemnts.element import StartType


automata = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
print "Finished processing from anml file. Here is the summary"

automata.remove_ors()

orig_automatas = automata.get_connected_components_as_automatas()
current_automata = orig_automatas[0]

all_nodes = current_automata.get_nodes()
set1 = set()
set2 = set()
set3 = set()

for node_idx, node in enumerate(all_nodes):
    if node.get_start() == atma.StartType.fake_root:
        continue
    if node_idx % 3 == 0:
        set1.add(node)
    elif node_idx % 3 == 1:
        set2.add(node)
    else:
        set3.add(node)


in_list, out_list = current_automata.count_interconnect_activity(input_path[AnmalZoo.Snort], [set1, set2], [set2, set3])


print in_list, out_list