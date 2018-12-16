import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream


# automata1 = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
# automata1.remove_ors()
# automata1.print_summary()
# automatas = automata1.get_connected_components_as_automatas()
#
# to_save_automatas = automatas[:10]
#
# pickle.dump(to_save_automatas, open("snort1-10.pkl", "wb"))
#
# automata = pickle.load(open('split_check.pkl', 'rb'))
# l_atm, r_atm = automata.split()
# llatm,lratm = l_atm.split()
# rlatm,rrratm = r_atm.split()
#
# # automata.draw_graph('automata.svg', draw_edge_label = False, use_dot = True, write_node_labels = True)
# # l_atm.draw_graph('l_atm.svg', draw_edge_label = False, use_dot = True, write_node_labels = True)
# # r_atm.draw_graph('r_atm.svg', draw_edge_label = False, use_dot = True, write_node_labels = True)
# compare_strided(False, input_path[AnmalZoo.Snort], (automata,), (llatm,lratm,rlatm,rrratm))
#
# exit(0)


#automatas = pickle.load(open('atm61.pkl','rb'))

automata1 = atma.parse_anml_file(anml_path[AnmalZoo.Hamming])
automata1.remove_ors()
automata1.print_summary()
automatas = automata1.get_connected_components_as_automatas()
faulty_automats =[]

for atm_idx, atm in enumerate(automatas):

    print "idx=", atm_idx
    atm.remove_all_start_nodes()
    atm.remove_ors()
    atm.print_summary()



    #atm.draw_graph(file_name='1.svg', draw_edge_label=True, use_dot=True, write_node_labels=True)
    atm2=atm.get_single_stride_graph()
    #atm2.draw_graph(file_name='2.svg', draw_edge_label=True, use_dot=True, write_node_labels=True)
    atm2.make_homogenous()
    #atm2.draw_graph(file_name='1-5.svg', draw_edge_label=True, use_dot=True, write_node_labels=True)
    #minimize_automata(atm2, merge_reports=True, same_residuals_only=True, same_report_code=True,
    #                  combine_symbols=False)
    #atm2.draw_graph(file_name='2H.svg', draw_edge_label=True, use_dot=True, write_node_labels=True)

    atm2=atm2.get_single_stride_graph()

    #atm2.draw_graph(file_name='4.svg', draw_edge_label=True, use_dot=True, write_node_labels=True)
    atm2.make_homogenous()
    atm2.print_summary()
    #atm2.draw_graph(file_name='4H.svg', draw_edge_label=True, use_dot=True, write_node_labels=True)
    minimize_automata(atm2,merge_reports=True,same_residuals_only=True,same_report_code=True,
                      combine_symbols=False)
    atm2.print_summary()
    #atm2.draw_graph(file_name='4HM.svg', draw_edge_label=True, use_dot=True, write_node_labels=True)
    #atm2l, atm2r = atm2.split()

    for n in atm2.nodes:
        if n.start_type==StartType.fake_root:
            continue
        s=n.is_symbolset_splitable()
        if not s:
            print s
            faulty_automats.append(atm_idx)
            break

    #compare_input(True,True,input_path[AnmalZoo.Snort], atm2,atm)
    #compare_strided(False,input_path[AnmalZoo.Snort],(atm2l,atm2r), (atm2,))

print faulty_automats











