import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata



for automata_name, automata_path in anml_path.iteritems():
    if automata_name != AnmalZoo.Levenshtein:
        continue
    automata = atma.parse_anml_file(automata_path)
    print "Finished processing from anml file. Here is the summary", str(automata_name)

    automata.remove_ors()

    #automata.re_label_automata_states(4)


    orig_atm_nodes_count, fst_stride_nodes_count, sec_stride_nodes_count,\
    third_stride_nodes_count, fourth_stride_nodes_count = [], [], [], [], []


    orig_automatas = automata.get_connected_components_as_automatas()

    for orig_cc in tqdm(orig_automatas, unit="automata"):

        minimize_automata(orig_cc)
        orig_cc.remove_all_start_nodes()
        print "Original CC number of nodes = %d" % (
            orig_cc.get_number_of_nodes(without_fake_root=True))
        orig_atm_nodes_count.append((orig_cc.get_number_of_nodes(without_fake_root=True), orig_cc.get_number_of_edges(),
                                     orig_cc.max_STE_out_degree(),orig_cc.max_STE_in_degree()))




        fst_st_atm = orig_cc.get_single_stride_graph()
        fst_st_atm.make_homogenous()
        minimize_automata(fst_st_atm)
        print "Striding phase 1 finished. Number of nodes = %d" % (
            fst_st_atm.get_number_of_nodes(without_fake_root=True))
        fst_stride_nodes_count.append((fst_st_atm.get_number_of_nodes(without_fake_root=True), fst_st_atm.get_number_of_edges(),
                                       fst_st_atm.max_STE_out_degree(),fst_st_atm.max_STE_in_degree()))

        sec_st_atm = fst_st_atm.get_single_stride_graph()
        sec_st_atm.make_homogenous()
        minimize_automata(sec_st_atm)
        print "Striding phase 2 finished. Number of nodes = %d" % (
            sec_st_atm.get_number_of_nodes(without_fake_root=True))
        sec_stride_nodes_count.append((sec_st_atm.get_number_of_nodes(without_fake_root=True), sec_st_atm.get_number_of_edges(),
                                       sec_st_atm.max_STE_out_degree(),sec_st_atm.max_STE_in_degree()))

        thd_st_atm = sec_st_atm.get_single_stride_graph()
        thd_st_atm.make_homogenous()
        minimize_automata(thd_st_atm)
        print "Striding phase 3 finished. Number of nodes = %d" % (
            thd_st_atm.get_number_of_nodes(without_fake_root=True))
        third_stride_nodes_count.append((thd_st_atm.get_number_of_nodes(without_fake_root=True), thd_st_atm.get_number_of_edges(),
                                         thd_st_atm.max_STE_out_degree(),thd_st_atm.max_STE_in_degree()))

        fourth_st_atm = thd_st_atm.get_single_stride_graph()
        fourth_st_atm.make_homogenous()
        minimize_automata(fourth_st_atm)
        print "Striding phase 4 finished. Number of nodes = %d" % (
            fourth_st_atm.get_number_of_nodes(without_fake_root=True))
        fourth_stride_nodes_count.append((fourth_st_atm.get_number_of_nodes(without_fake_root=True), fourth_st_atm.get_number_of_edges(),
                                          fourth_st_atm.max_STE_out_degree(), fourth_st_atm.max_STE_in_degree()))


    pickle.dump((orig_atm_nodes_count,fst_stride_nodes_count,sec_stride_nodes_count,third_stride_nodes_count,fourth_stride_nodes_count), open(str(automata_name)+".pkl","wb"))













