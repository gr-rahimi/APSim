import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream
import automata.HDL.hdl_generator as hd_gen



strided_automatas = []
#automatas = pickle.load(open('snort1-10.pkl', 'rb'))
automatas = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()
for atm_idx, atm in enumerate(automatas):
    atm.remove_all_start_nodes()
    atm.remove_ors()
    atm.print_summary()

    atm2=atm.get_single_stride_graph()
    atm2.make_homogenous()
    minimize_automata(atm2, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=True)
    strided_automatas.append(atm2)
    atm2.draw_graph(file_name='atm'+str(atm_idx)+'.svg', draw_edge_label=False, use_dot=True, write_node_labels=False)
    atm2.print_summary()

hd_gen.generate_full_lut(strided_automatas, single_file=False, folder_name='snortall')



