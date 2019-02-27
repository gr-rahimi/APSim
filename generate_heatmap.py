from __future__ import division
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
import os, shutil
from tqdm import tqdm
import numpy as np
from automata.utility import utility

diagonal_routing = utility.generate_diagonal_route(256, 10)


for anml in [AnmalZoo.Dotstar03, AnmalZoo.Dotstar06, AnmalZoo.Dotstar09, AnmalZoo.Ranges05,
             AnmalZoo.Ranges1, AnmalZoo.ExactMath, AnmalZoo.Bro217, AnmalZoo.TCP]:

    if anml in [AnmalZoo.ClamAV, AnmalZoo.Synthetic_CoreRings]:
        continue

    if not os.path.exists("Results/"+str(anml)):
        os.makedirs("Results/"+str(anml)) #make directory if it does not exist
    else:
        shutil.rmtree("Results/"+str(anml)) #remove content
        os.makedirs("Results/"+str(anml))  # make directory if it does not exist

    automata = atma.parse_anml_file(anml_path[anml])
    automata.remove_ors()
    #utility.minimize_automata(automata)
    acc_switch_map = None # accumulative switch map
    ccs = automata.get_connected_components_as_automatas()
    for cc_idx, cc in enumerate(tqdm(ccs)):

        bfs_cost, bfs_label_dictionary = cc.bfs_rout(diagonal_routing, None)
        switch_map = cc.draw_native_switch_box("Results/" + str(anml) + "/number_" + str(cc_idx) + "bfs_cost_" + str(bfs_cost), bfs_label_dictionary,
                                               True, dpi=100)
        if not acc_switch_map:
            acc_switch_map = switch_map
        else:
            X = len(switch_map)
            Y = len(switch_map[0])
            assert X == len(acc_switch_map) and Y == len(acc_switch_map[0]), "they should hacve equal size"

            for x in range(X):
                for y in range(Y):
                    acc_switch_map[x][y] += switch_map[x][y]

    heat_map = np.array(acc_switch_map) / cc_idx

    utility.draw_matrix("Results/" + str(anml) + "/heat_map.png", heat_map, [i / 256 for i in range(257)], dpi =500)
    #utility.draw_matrix("Results/" + str(anml) + "/heat_map.png", heat_map, [0,1/(cc_idx+1), 1], dpi=500)




























