from __future__ import division
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
import matplotlib.pyplot as plt
from automata.utility import utility

diagonal_routing = utility.generate_diagonal_route(256, 10)

automata = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
automata.remove_ors()

ccs = automata.get_connected_components_as_automatas()
while True:
    fst_automata_idx = int(raw_input("Enter first automata "))
    sec_automata_idx = int(raw_input("Enter second automata "))

    fst_automata = ccs[fst_automata_idx]
    sec_automata = ccs[sec_automata_idx]

    f,axarr = plt.subplots(2,3)

    fst_bfs_cost, fst_bfs_label_dictionary = fst_automata.bfs_rout(diagonal_routing, None)
    fst_switch_map = fst_automata.get_connectivity_matrix(fst_bfs_label_dictionary)
    utility.draw_matrix_on_ax(ax=axarr[0][0], matrix= fst_switch_map, boundries = [0, 0.5, 1])
    axarr[0][0].set_title("first automata")

    sec_bfs_cost, sec_bfs_label_dictionary = sec_automata.bfs_rout(diagonal_routing, None)
    sec_switch_map = sec_automata.get_connectivity_matrix(sec_bfs_label_dictionary)
    utility.draw_matrix_on_ax(ax=axarr[1][0], matrix=sec_switch_map, boundries=[0, 0.5, 1])
    axarr[1][0].set_title("second automata")

    #fst_automata.draw_graph("fst_graph.png")
    #sec_automata.draw_graph("fst_graph.png")

    fst_automata.add_automata(sec_automata)
    fst_bfs_cost, fst_bfs_label_dictionary = fst_automata.bfs_rout(diagonal_routing, None)
    fst_switch_map = fst_automata.get_connectivity_matrix(fst_bfs_label_dictionary)
    utility.draw_matrix_on_ax(ax=axarr[0][1], matrix=fst_switch_map, boundries=[0, 0.5, 1])
    axarr[0][1].set_title("simple add")

    fst_automata.left_merge()
    fst_bfs_cost, fst_bfs_label_dictionary = fst_automata.bfs_rout(diagonal_routing, None)
    fst_switch_map = fst_automata.get_connectivity_matrix(fst_bfs_label_dictionary)
    utility.draw_matrix_on_ax(ax=axarr[1][1], matrix= fst_switch_map, boundries = [0, 0.5, 1])
    axarr[1][1].set_title("left merge")


    fst_automata.right_merge()
    fst_bfs_cost, fst_bfs_label_dictionary = fst_automata.bfs_rout(diagonal_routing, None)
    fst_switch_map = fst_automata.get_connectivity_matrix(fst_bfs_label_dictionary)
    utility.draw_matrix_on_ax(ax=axarr[0][2], matrix= fst_switch_map, boundries = [0, 0.5, 1])
    axarr[0][2].set_title("right merge")

    fst_automata.combine_symbol_sets()
    fst_bfs_cost, fst_bfs_label_dictionary = fst_automata.bfs_rout(diagonal_routing, None)
    fst_switch_map = fst_automata.get_connectivity_matrix(fst_bfs_label_dictionary)
    utility.draw_matrix_on_ax(ax=axarr[1][2], matrix= fst_switch_map, boundries = [0, 0.5, 1])
    axarr[1][2].set_title("combine symbols")


    plt.show()






