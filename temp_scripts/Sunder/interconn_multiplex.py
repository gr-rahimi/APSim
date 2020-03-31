import os
import fcntl
import csv
from multiprocessing.dummy import Pool as ThreadPool
import traceback

import networkx as nx
import networkx.algorithms.matching as matching_alg

import automata as atma
from automata.elemnts.ste import is_there_common_sym, PackedIntervalSet
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo, input_path10M
from automata.utility.utility import minimize_automata, get_approximate_automata, automata_run_stat
from automata.utility import total_reports, reports_per_cycle, total_active_states, actives_per_cycle, reports_in_cycle
from automata.utility.utility import draw_graph

atms_count = 100  #
filed_names = ['number of original states', 'number of matched states']
thread_count = 4

def thread_func(ds):
    try:

        full_atm = atma.parse_anml_file(anml_path[ds])
        full_atm.remove_ors()
        atms_list = full_atm.get_connected_components_as_automatas()
        atms_list = atms_list[:atms_count]

        with open(str(ds) + '4bit.csv',
                  'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(filed_names)
            for curr_atm in atms_list:
                b_atm = atma.automata_network.get_bit_automaton(curr_atm, original_bit_width=curr_atm.max_val_dim_bits_len)
                four_b_atm = atma.automata_network.get_strided_automata2(atm=b_atm,
                                                                  stride_value=4,
                                                                  is_scalar=True,
                                                                  base_value=2,
                                                                  add_residual=True)
                curr_atm = four_b_atm.get_single_stride_graph()
                curr_atm.make_homogenous()
                minimize_automata(curr_atm)
                curr_atm.fix_split_all()

                match_graph = nx.Graph()

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

                matching_result = matching_alg.max_weight_matching(match_graph)

                print "number of orignal nodes ", curr_atm.nodes_count
                print "number of matching nodes ", 2 * len(matching_result)
                csv_writer.writerow([curr_atm.nodes_count, 2 * len(matching_result)])
    except Exception as ex:
        print traceback.print_exc()

if __name__ == '__main__':

    ds = [a for a in AnmalZoo]

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(thread_func, ds)
    t_pool.close()
    t_pool.join()