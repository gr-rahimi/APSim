from __future__ import division
import automata as atma
import matplotlib.pyplot as plt
from automata.utility import utility
import automata
from automata.elemnts.ste import S_T_E
from automata.elemnts.element import StartType

diagonal_routing = utility.generate_diagonal_route(256, 10)



num_nodes = 4
max_fan_in = 3
max_fan_out = 3

def gen_fully_connected_automata(size):
    atm = atma.automata_network.Automatanetwork(id="full automata", is_homogenous=True, stride=1, max_val=255)
    for i in range(size):
        to_add_ste = automata.elemnts.ste.S_T_E(start_type= StartType.start_of_data if i == 0 else StartType.non_start,
                                                is_report = False, is_marked=False, id= str(i),
                                                symbol_set=set())
        atm.add_element(to_add_ste)

    for i in range(size):
        for j in range(size):
            atm.add_edge(atm.get_STE_by_id(str(i)),atm.get_STE_by_id(str(j)))

    return atm


#f,axarr = plt.subplots(4)

atm_orig = gen_fully_connected_automata(num_nodes)

atm_max_fan_in = gen_fully_connected_automata(num_nodes)
atm_max_fan_in.set_max_fan_in(max_fan_in)

atm_max_fan_out = gen_fully_connected_automata(num_nodes)
atm_max_fan_out.set_max_fan_in(max_fan_out)


atm_both_max_fan = gen_fully_connected_automata(num_nodes)

atm_orig_bfs_cost, atm_orig_bfs_label_dictionary = atm_orig.bfs_rout(diagonal_routing, None)
atm_orig_switch_map = atm_orig.get_connectivity_matrix(atm_orig_bfs_label_dictionary)
utility.draw_matrix_on_ax(ax=axarr[0], matrix= atm_orig_switch_map, boundries = [0, 0.5, 1])
axarr[0].set_title("orig automata")

atm_max_fan_in_bfs_cost, atm_max_fan_in_bfs_label_dictionary = atm_max_fan_in.bfs_rout(diagonal_routing, None)
atm_max_fan_in_switch_map = atm_max_fan_in.get_connectivity_matrix(atm_max_fan_in_bfs_label_dictionary)
utility.draw_matrix_on_ax(ax=axarr[1], matrix= atm_max_fan_in_switch_map, boundries = [0, 0.5, 1])
axarr[1].set_title("fan in automata")

atm_max_fan_out_bfs_cost, atm_max_fan_out_bfs_label_dictionary = atm_max_fan_out.bfs_rout(diagonal_routing, None)
atm_max_fan_out_switch_map = atm_max_fan_out.get_connectivity_matrix(atm_max_fan_out_bfs_label_dictionary)
utility.draw_matrix_on_ax(ax=axarr[2], matrix= atm_max_fan_out_switch_map, boundries = [0, 0.5, 1])
axarr[2].set_title("fan out automata")

plt.show()










