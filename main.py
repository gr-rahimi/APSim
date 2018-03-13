import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors


def generate_route(size, diagonal_width):
    routing_matrix = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        routing_matrix[i][i] = 1
        for j in range(1, diagonal_width + 1):
            if i - j >= 0:
                routing_matrix[i][i - j] = 1

            if i + j < size:
                routing_matrix[i][i + j] = 1

    return routing_matrix


def generate_squared_routing(size, basic_block_WH, overlap):
    routing_matrix = [[0 for _ in range(size)] for _ in range(size)]

    def generate_pattern(start, size):
        if size == basic_block_WH:
            for i in range(size):
                for j in range(size):
                    routing_matrix[start + i][start + j] = 1

        else:
            generate_pattern(start + size / 2, size / 2)
            generate_pattern(start + overlap, size / 2)
            for i in range(size):
                for j in range(overlap):
                    routing_matrix[start + i][start + j] = 1
                    routing_matrix[start + j][start + i] = 1

    generate_pattern(0, size)

    def draw_pattern():

        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig, ax = plt.subplots()
        ax.imshow(routing_matrix, cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0)
        ax.set_xticks(range(0, size, 15))
        ax.set_yticks(range(0, size, 15))

        plt.gca().invert_yaxis()
        plt.savefig("pattern.png")
        plt.clf()

    draw_pattern()
    return routing_matrix



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

for atm_idx, atm in enumerate(orig_automatas):
    bfs_assignment = atm.get_BFS_label_dictionary()
    atm.draw_switch_box("snort/" + "atm_"+str(atm_idx), bfs_assignment)


exit(0)

current_automata = orig_automatas[0]
current_automata.set_max_fan_in(5)
current_automata.set_max_fan_out(5)
#routing_matrix = generate_route(current_automata.get_number_of_nodes(),10)
routing_matrix = generate_squared_routing(256, 8, 4)
current_automata.bfs_rout(routing_matrix, None)
current_automata.draw_switch_box("snort/bfs_routing",current_automata.get_BFS_label_dictionary())
ga_routing_dic = current_automata.ga_route(routing_template = routing_matrix, avilable_rows = range(current_automata.get_number_of_nodes()))
current_automata.draw_switch_box("snort/ga_routing", ga_routing_dic)

exit(0)























