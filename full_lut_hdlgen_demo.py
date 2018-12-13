import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream
import automata.HDL.hdl_generator as hd_gen
import csv

#csv_file = open('stride_result.csv', mode='w')
#csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

strided_automatas = []
automatas = pickle.load(open('snort1-20.pkl', 'rb'))
#automatas = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
#automatas.remove_ors()
#automatas = automatas.get_connected_components_as_automatas()



for atm_idx, atm in enumerate(automatas):
    atm.remove_all_start_nodes()
    atm.remove_ors()
    atm.print_summary()

    atm2 = atm.get_single_stride_graph()
    atm4 = atm2.get_single_stride_graph()
    atm8 = atm4.get_single_stride_graph()

    atm2.make_homogenous()
    minimize_automata(atm2, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=True)

    atm2.print_summary()



    atm4.make_homogenous()
    minimize_automata(atm4, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=True)
    atm4.print_summary()

    atm8.make_homogenous()
    minimize_automata(atm8, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=True)
    atm8.print_summary()

    raw_input("press a key for next")
    #csv_writer.writerow([atm.nodes_count, atm2.nodes_count, atm4.nodes_count, atm8.nodes_count])

    del atm2, atm4, atm8


#csv_file.close()

    #strided_automatas.append(atm8)
    #atm2.draw_graph(file_name='atm'+str(atm_idx)+'.svg', draw_edge_label=False, use_dot=True, write_node_labels=False)
    #atm8.print_summary()

    #exit(0)

#hd_gen.generate_full_lut(strided_automatas, capture_symbol=False,single_file=False, folder_name='snort108S_nocap')
#hd_gen.generate_full_lut(strided_automatas, capture_symbol=True,single_file=False, folder_name='snort108S_cap')


