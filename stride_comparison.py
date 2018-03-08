import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle

def minimize_automata(automata):
    automata.combile_finals_with_same_symbol_set()
    while True:
        current_node_cont = automata.get_number_of_nodes()
        automata.left_merge()
        automata.right_merge()
        automata.combine_symbol_sets()
        new_node_count = automata.get_number_of_nodes()
        assert new_node_count<= current_node_cont, "it should always be smaller"
        if new_node_count == current_node_cont:
            break


for automata_name, automata_path in anml_path.iteritems():
    if automata_name == AnmalZoo.EntityResolution or automata_name == AnmalZoo.Synthetic or automata_name == AnmalZoo.RandomForest:
        continue
    automata = atma.parse_anml_file(automata_path)
    print "Finished processing from anml file. Here is the summary", str(automata_name)

    automata.remove_ors()

    orig_atm_nodes_count = []
    fst_stride_nodes_count = []
    sec_stride_nodes_count = []
    third_stride_nodes_count = []

    orig_automatas = automata.get_connected_components_as_automatas()

    for orig_cc in tqdm(orig_automatas, unit="automata"):
        orig_atm_nodes_count.append((orig_cc.get_number_of_nodes(), orig_cc.get_number_of_edges(),
                                     orig_cc.max_STE_out_degree(),orig_cc.max_STE_in_degree() ))

        orig_cc.remove_all_start_nodes()

        fst_st_atm = orig_cc.get_single_stride_graph()
        fst_st_atm.make_homogenous()
        minimize_automata(fst_st_atm)
        fst_stride_nodes_count.append((fst_st_atm.get_number_of_nodes(), fst_st_atm.get_number_of_edges(),
                                       fst_st_atm.max_STE_out_degree(),fst_st_atm.max_STE_in_degree()))

        sec_st_atm = fst_st_atm.get_single_stride_graph()
        sec_st_atm.make_homogenous()
        minimize_automata(sec_st_atm)
        sec_stride_nodes_count.append((sec_st_atm.get_number_of_nodes(), sec_st_atm.get_number_of_edges(),
                                       sec_st_atm.max_STE_out_degree(),sec_st_atm.max_STE_in_degree()))

        thd_st_atm = sec_st_atm.get_single_stride_graph()
        thd_st_atm.make_homogenous()
        minimize_automata(thd_st_atm)
        third_stride_nodes_count.append((thd_st_atm.get_number_of_nodes(), thd_st_atm.get_number_of_edges(),
                                         thd_st_atm.max_STE_out_degree(),thd_st_atm.max_STE_in_degree()))


    pickle.dump((orig_atm_nodes_count,fst_stride_nodes_count,sec_stride_nodes_count,third_stride_nodes_count), open(str(automata_name)+".pkl","wb"))













