import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata



for automata_name, automata_path in anml_path.iteritems():
    if automata_name != AnmalZoo.Custom:
        continue
    automata = atma.parse_anml_file(automata_path)
    print "Finished processing from anml file. Here is the summary", str(automata_name)

    automata.remove_ors()

    #automata.re_label_automata_states(4)

    orig_automatas = automata.get_connected_components_as_automatas()

    for orig_cc in tqdm(orig_automatas, unit="automata"):

        orig_cc.remove_all_start_nodes()
        minimize_automata(orig_cc)
        print "original automata:"
        orig_cc.print_summary()
        fst_st_atm = orig_cc.get_single_stride_graph()
        sec_st_atm = fst_st_atm.get_single_stride_graph()
        thd_st_atm = sec_st_atm.get_single_stride_graph()
        thd_st_atm.make_homogenous()
        thd_st_atm.print_summary()
        #frt_st_atm = thd_st_atm.get_single_stride_graph()

        #frt_st_atm.make_homogenous()

        #print "fourth stride automata:"
        #frt_st_atm.print_summary()
        #minimize_automata(frt_st_atm)

        #print max([cc.get_number_of_nodes(True) for cc in frt_st_atm.get_connected_components_as_automatas()])






        atma.compare_input(True,input_path[automata_name], orig_cc, thd_st_atm)















