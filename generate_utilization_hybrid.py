from __future__ import division
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from tqdm import tqdm
from automata.utility import utility
from switch_map.basic_box import  BaseSwitch



#diagonal_routing = utility.generate_diagonal_route(256,10)
diagonal_routing = utility.generate_semi_diagonal_route(1, 5)

full_mesh_routing = [[1 for _ in range(256)] for _ in range(256)]

result_dict = {}
unsupported_anmls = [AnmalZoo.Synthetic_CoreRings, AnmalZoo.ClamAV]

for anml in AnmalZoo:
    if anml in unsupported_anmls:
        continue

    full_mesh_sb_list, linear_sb_list = [], []

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
            sb_list = linear_sb_list
        else:
            sb_list = full_mesh_sb_list

        min_remained_capacity = 256
        sb_candidate = None
        for sb in sb_list:
            capacity = sb.get_remaining_capacity()
            if capacity>= number_of_nodes and\
                capacity < min_remained_capacity:
                min_remained_capacity = capacity
                sb_candidate = sb

        if sb_candidate:
            actual_sb = cc.get_connectivity_matrix(bfs_label_dictionary)
            sb_candidate.add_switch_box_layout(actual_sb)
        else: # make a new linear switch box
            actual_sb = cc.get_connectivity_matrix(bfs_label_dictionary)
            new_sb = BaseSwitch((256, 256))
            new_sb.set_raw_switch(diagonal_routing if bfs_cost ==0 else full_mesh_routing)
            new_sb.add_switch_box_layout(actual_sb)
            sb_list.append(new_sb)

    if unsupported:
        unsupported_anmls.append(anml)
        continue

    for sb in  full_mesh_sb_list:
        print "full mesh", sb.get_utilization()

    for sb in  linear_sb_list:
        print "linear mesh", sb.get_utilization()


    result_dict[anml] = (full_mesh_sb_list, linear_sb_list, total_edges)

for anml in AnmalZoo:
    if not anml in result_dict:
        continue
    #print anml, " total edges = " , res[2], " number of linear meshes = ", len(res[1]), "number of full meshes = ", len(res[0])
    res = result_dict[anml]
    print anml, res[2], "\t", len(res[1]), "\t", len(res[0])

print "Unsupported animals: ", unsupported_anmls
























