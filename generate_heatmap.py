from __future__ import division
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
import os, shutil
from tqdm import tqdm
import numpy as np
from automata.utility import utility


number_of_automatas = 1
draw_individually = False
max_stride = 3
switch_size = 256
diagonal_routing = utility.generate_diagonal_route(switch_size, 10)

for anml in [AnmalZoo.PowerEN]:

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
    acc_switch_map = np.zeros((switch_size, switch_size)) # accumulative switch map
    ccs = automata.get_connected_components_as_automatas()

    for stride in range(max_stride + 1): # one more for the original automata

        print "starting stride ", stride
        for cc_idx, cc in enumerate(ccs[:number_of_automatas]):
            print "processing {} , id {}".format(anml, cc_idx)


            for _ in range(stride):
                cc = cc.get_single_stride_graph()

            if cc.is_homogeneous is False:
                cc.make_homogenous()

            utility.minimize_automata(cc)
            cc.fix_split_all()

            print cc.get_summary()

            bfs_cost, bfs_label_dictionary = cc.bfs_rout(diagonal_routing)

            switch_map = cc.get_connectivity_matrix(node_dictionary=bfs_label_dictionary)

            acc_switch_map += switch_map

        heat_map = acc_switch_map / number_of_automatas

        utility.draw_matrix(file_to_save="Results/" + str(anml) + "/heat_mapS"+str(stride)+".png", matrix=heat_map,
                            boundries=[i / 256 for i in range(257)], dpi=500)




























