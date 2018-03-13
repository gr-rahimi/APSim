import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle

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
def generate_route(size, diagonal_width):
    routing_matrix = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        routing_matrix[size-1-i][i] = 1
        for j in range(diagonal_width):
            if size-1-i-j-1>= 0:
                routing_matrix[size-1-i-j-1][i] =1

            if size-1-i+j+1< size:
                routing_matrix[size-1-i+j + 1][i] =1

    return routing_matrix

current_automata = orig_automatas[1]

routing_matrix = generate_route(current_automata.get_number_of_nodes(),5)
current_automata.bfs_rout(routing_matrix, None)
current_automata.draw_switch_box("snort/bfs_routing",current_automata.get_BFS_label_dictionary())
ga_routing_dic = current_automata.ga_route(routing_template = routing_matrix, avilable_rows = range(current_automata.get_number_of_nodes()))
current_automata.draw_switch_box("snort/ga_routing", ga_routing_dic)

exit(0)

for atm_idx, atm in enumerate(orig_automatas):

    atm.draw_switch_box("snort/" + "atm_"+str(atm_idx))
    if atm_idx == 100:
        break





















