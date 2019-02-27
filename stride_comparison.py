import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from tqdm import tqdm
import pickle
from automata.utility.utility import minimize_automata



for automata_name, automata_path in anml_path.iteritems():
    if automata_name != AnmalZoo.Levenshtein:
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
        orig_atm_nodes_count.append((orig_cc.get_number_of_nodes(), orig_cc.edges_count,
                                     orig_cc.max_STE_out_degree(),orig_cc.max_STE_in_degree() ))

        orig_cc.remove_all_start_nodes()

        fst_st_atm = orig_cc.get_single_stride_graph()
        fst_st_atm.make_homogenous()
        minimize_automata(fst_st_atm)
        fst_stride_nodes_count.append((fst_st_atm.get_number_of_nodes(), fst_st_atm.edges_count,
                                       fst_st_atm.max_STE_out_degree(),fst_st_atm.max_STE_in_degree()))

        sec_st_atm = fst_st_atm.get_single_stride_graph()
        sec_st_atm.make_homogenous()
        minimize_automata(sec_st_atm)
        sec_stride_nodes_count.append((sec_st_atm.get_number_of_nodes(), sec_st_atm.edges_count,
                                       sec_st_atm.max_STE_out_degree(),sec_st_atm.max_STE_in_degree()))

        thd_st_atm = sec_st_atm.get_single_stride_graph()
        thd_st_atm.make_homogenous()
        minimize_automata(thd_st_atm)
        third_stride_nodes_count.append((thd_st_atm.get_number_of_nodes(), thd_st_atm.edges_count,
                                         thd_st_atm.max_STE_out_degree(),thd_st_atm.max_STE_in_degree()))


    pickle.dump((orig_atm_nodes_count,fst_stride_nodes_count,sec_stride_nodes_count,third_stride_nodes_count), open(str(automata_name)+".pkl","wb"))













