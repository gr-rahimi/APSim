from __future__ import division
from automata.utility.utility import pact_interconnect
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
import os, shutil
from tqdm import tqdm
import numpy as np
import pickle
from automata.utility import utility

routing_template = pact_interconnect(image_name='esi.png')
number_of_automatas = 200
draw_individually = False
max_stride = 2

for anml in [
             AnmalZoo.Snort]:



    automata = atma.parse_anml_file(anml_path[anml])
    automata.remove_ors()
    #utility.minimize_automata(automata)
     # accumulative switch map
    ccs = automata.get_connected_components_as_automatas()

    for stride in range(max_stride, max_stride + 1): # one more for the original automata

        same_incon_list, curr_node_count = [], 0
        for cc_idx, cc in enumerate(ccs[:number_of_automatas]):

            print cc.get_summary(logo="original")

            for _ in range(stride):
                cc = cc.get_single_stride_graph()

            if cc.is_homogeneous is False:
                cc.make_homogenous()

            utility.minimize_automata(cc)
            cc.fix_split_all()

            #print cc.get_summary()

            if curr_node_count + cc.nodes_count > 1024:
                print "started genetic"
                utility.ga_routing(atms_list=same_incon_list,
                                   routing_template=routing_template,
                                   available_rows=range(1024), draw_file='reza.png')
                exit(0)
                same_incon_list, curr_node_count = [cc], cc.nodes_count
            else:
                same_incon_list.append(cc)
                curr_node_count += cc.nodes_count




















