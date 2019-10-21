from __future__ import division
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
import os, shutil
from tqdm import tqdm
import numpy as np
from automata.utility import utility


number_of_automatas = 50
draw_individually = False
max_stride = 0
switch_size = 128
routing_template = None

for anml in [
             AnmalZoo.TCP, AnmalZoo.ClamAV]:

    if not os.path.exists("Results/"+str(anml)):
        os.makedirs("Results/"+str(anml)) #make directory if it does not exist
    else:
        pass
        #shutil.rmtree("Results/"+str(anml)) #remove content
        #os.makedirs("Results/"+str(anml))  # make directory if it does not exist

    automata = atma.parse_anml_file(anml_path[anml])
    automata.remove_ors()
    #utility.minimize_automata(automata)
     # accumulative switch map
    ccs = automata.get_connected_components_as_automatas()

    for stride in range(max_stride, max_stride + 1): # one more for the original automata
        acc_switch_map = np.zeros((switch_size, switch_size))

        print "starting stride ", stride
        for cc_idx, cc in enumerate(ccs[:]):
            print "processing {} , id {}".format(anml, cc_idx)
            print cc.get_summary(logo="original")

            for _ in range(stride):
                cc = cc.get_single_stride_graph()

            if cc.is_homogeneous is False:
                cc.make_homogenous()

            utility.minimize_automata(cc)
            cc.fix_split_all()

            print cc.get_summary()

            bfs_cost, bfs_label_dictionary = cc.bfs_rout(routing_template)

            switch_map = cc.get_connectivity_matrix(node_dictionary=bfs_label_dictionary)

            acc_mat_size, _ = acc_switch_map.shape
            new_mat_size, _ = switch_map.shape

            if new_mat_size > acc_mat_size:
                switch_map[:acc_mat_size, :acc_mat_size] += acc_switch_map
                acc_switch_map = switch_map
            else:
                acc_switch_map[:new_mat_size, :new_mat_size] += switch_map

        heat_map = acc_switch_map / 1

        utility.draw_matrix(file_to_save="Results/" + str(anml) + "/heat_mapS"+str(stride)+".png", matrix=heat_map,
                            boundries=[i / 256 for i in range(257)], dpi=500)




























