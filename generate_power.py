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
from switch_map.basic_box import  BaseSwitch



#diagonal_routing = utility.generate_diagonal_route(256,10)
diagonal_routing = utility.generate_semi_diagonal_route(1,12)

full_mesh_routing = [[1 for _ in range(256)] for _ in range(256)]

result_dict = {}
unsupported_anmls = [AnmalZoo.Synthetic_CoreRings, AnmalZoo.ClamAV]

for anml in AnmalZoo:
    if anml in unsupported_anmls:
        continue

    full_mesh_sb_list, linear_sb_list = [], []
    full_mesh_combined ,linear_combined = [], []


    automata = atma.parse_anml_file(anml_path[anml])
    print "start processing", anml
    automata.remove_ors()
    total_edges = automata.get_number_of_edges()
    ccs = automata.get_connected_components_as_automatas()
    ccs.sort(key = lambda cc : cc.get_number_of_nodes(True), reverse=True)

    unsupported = False
    for cc in tqdm(ccs):
        number_of_nodes = cc.get_number_of_nodes(True)
        if number_of_nodes > 256:
            unsupported = True
            break # we only support less tahn 256
        #first check if it can be fitted in linear mode or not
        bfs_cost, bfs_label_dictionary = cc.bfs_rout(diagonal_routing, None)

        if bfs_cost == 0: # it can be supported in a linear case, let's find a used one that has enough capacity or make a new one
            sb_list, combined_list = linear_sb_list, linear_combined
        else:
            sb_list, combined_list = full_mesh_sb_list, full_mesh_combined

        min_remained_capacity = 256
        sb_candidate = None
        for sb, ca in zip(sb_list, combined_list):
            capacity = sb.get_remaining_capacity()
            if capacity>= number_of_nodes and\
                capacity < min_remained_capacity:
                min_remained_capacity = capacity
                sb_candidate = sb
                ca_candidate = ca

        if sb_candidate:
            actual_sb = cc.get_connectivity_matrix(bfs_label_dictionary)
            sb_candidate.add_switch_box_layout(actual_sb)
            ca_candidate.add_automata(cc)
        else: # make a new linear switch box
            actual_sb = cc.get_connectivity_matrix(bfs_label_dictionary)
            new_sb = BaseSwitch((256, 256))
            new_sb.set_raw_switch(diagonal_routing if bfs_cost ==0 else full_mesh_routing)
            new_sb.add_switch_box_layout(actual_sb)
            sb_list.append(new_sb)
            combined_list.append(cc)

    if unsupported:
        unsupported_anmls.append(anml)
        continue

    print "start feeding input for ", anml
    linear_count = 0
    full_count = 0

    new_automata = atma.Automatanetwork(id="temp", is_homogenous=True, stride=1, max_val=255)
    print "start combining automatas for linear case", anml
    linear_set=[]
    for ca in tqdm(linear_combined):
        new_stes = new_automata.add_automata(ca)
        linear_set.append(new_stes)
    print "start feeding input for linear"
    inb_counter_linear, _= new_automata.count_interconnect_activity(input_path[anml],
                                                       inbound_set_list=linear_set, outbound_set_list=[set()])

    new_automata = atma.Automatanetwork(id="temp", is_homogenous=True, stride=1, max_val=255)
    print "start combining automatas for full case", anml
    full_set = []
    for ca in tqdm(full_mesh_combined):
        new_stes = new_automata.add_automata(ca)
        full_set.append(new_stes)
    print "start feeding input for full"
    inb_counter_full, _ = new_automata.count_interconnect_activity(input_path[anml],
                                                                     inbound_set_list=full_set,
                                                                     outbound_set_list=[set()])

    print "result of linear:",inb_counter_linear
    print "result of full:", inb_counter_full

    result_dict[anml] = (sum(inb_counter_linear), sum(inb_counter_full))

for anml in AnmalZoo:
    if not anml in result_dict:
        continue
    #print anml, " total edges = " , res[2], " number of linear meshes = ", len(res[1]), "number of full meshes = ", len(res[0])
    res = result_dict[anml]
    print anml, res[0], "\t", res[1]

print "Unsupported animals: ", unsupported_anmls
























