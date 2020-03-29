import os
import fcntl

import networkx as nx
import networkx.algorithms.matching as matching_alg

import automata as atma
from automata.elemnts.ste import is_there_common_sym, PackedIntervalSet
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo, input_path10M
from automata.utility.utility import minimize_automata, get_approximate_automata, automata_run_stat
from automata.utility import total_reports, reports_per_cycle, total_active_states, actives_per_cycle, reports_in_cycle
from automata.utility.utility import draw_graph

match_graph = nx.Graph()

full_atm = atma.parse_anml_file(anml_path[AnmalZoo.EntityResolution])
full_atm.remove_ors()
atms_list = full_atm.get_connected_components_as_automatas()

curr_atm = atms_list[30]  # automata under test

parent_sym_to_product_sym_dic = {}
for ste in curr_atm.nodes:
    if ste.is_fake:
        continue  # fake root is not considered

    match_graph.add_node(ste)

    # here we find the union of parents and find the version with false posetive versions
    comb_sym_set = PackedIntervalSet([])  # create an empty symbol sets to find union of parents symbol set
    for pred in curr_atm.get_predecessors(ste):
        if pred.is_fake:
            continue
        for ivl in pred.symbols.intervals:
            comb_sym_set.add_interval(ivl)  # finding the union of parents symbol sets

    #  making symbol set smaller
    comb_sym_set.prone()
    comb_sym_set.merge()
    comb_sym_set = comb_sym_set.get_combinatorial_symbol_set()
    parent_sym_to_product_sym_dic[ste] = comb_sym_set

    # now having a graph with no edge, we add edge between nodes if they can be combined with each other
    # Two nodes can be combined if these two conditions are satisfied. First, they should not have a common symbol
    # second, their parrents should also not have common symbols

    for n in match_graph.nodes():
        if n == ste:
            continue  # we do not check a node with itself

        if is_there_common_sym(ste.symbols, n.symbols):
            continue  # first condition has not been met
        elif is_there_common_sym(parent_sym_to_product_sym_dic[ste], parent_sym_to_product_sym_dic[n]):
            continue  # second condition has not been met
        else:  # both condiotions have been met, we can create an edge between ste, n
            match_graph.add_edge(ste, n)


curr_atm.draw_graph("curr_atm.svg")
draw_graph(match_graph, "match_graph.svg")

result = matching_alg.max_weight_matching(match_graph)
curr_atm.draw_graph("curr_atm_with_matching.svg", matching_nodes=result)

print "number of matching edges:", len(result)
print curr_atm.get_summary()











