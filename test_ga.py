import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
import utility
import networkx as nx
import networkx.algorithms.isomorphism as iso



automata = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
print "Finished processing from anml file. Here is the summary"

automata.remove_ors()

orig_automatas = automata.get_connected_components_as_automatas()
#orig_automatas[0].add_automata(orig_automatas[1])


#print atm.max_STE_in_degree()
#print atm.max_STE_out_degree()

#atm.set_max_fan_in(3)
#atm.set_max_fan_out(3)
#orig_automatas[0].draw_switch_box("snort/atm.png",orig_automatas[0].get_BFS_label_dictionary())

#for atm_idx, atm in enumerate(orig_automatas):
#    bfs_assignment = atm.get_BFS_label_dictionary()
#    atm.draw_switch_box("snort/" + "atm_"+str(atm_idx), bfs_assignment)


#exit(0)

current_automata = orig_automatas[0]
#current_automata.set_max_fan_in(4)
#current_automata.set_max_fan_out(4)
routing_matrix = utility.generate_diagonal_route(256,10)
routing_matrix= utility.cut_switch_box(routing_matrix, current_automata.get_number_of_nodes() + 1)
#routing_matrix = utility.generate_diagonal_route()

routing_matrix_graph = utility.get_graph_from_matrix(routing_matrix, True)

#routing_matrix = utility.generate_squared_routing(256, 8, 4)
current_automata.bfs_rout(routing_matrix, None)
bfs_switch_box = current_automata.draw_native_switch_box("snort/bfs_routing", current_automata.get_BFS_label_dictionary(),True,True)
bfs_switch_box = utility.cut_switch_box(bfs_switch_box ,current_automata.get_number_of_nodes() + 1)
bfs_switch_box_graph = utility.get_graph_from_matrix(bfs_switch_box, True)
#GM = iso.GraphMatcher(routing_matrix_graph, bfs_switch_box_graph)
print routing_matrix_graph.subisomorphic_lad(bfs_switch_box_graph)



#ga_routing_dic = current_automata.ga_route(routing_template = routing_matrix, avilable_rows = range(256))
#current_automata.draw_switch_box("snort/ga_routing", ga_routing_dic)

exit(0)























