import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream, draw_symbols_len_histogram
import automata.HDL.hdl_generator as hd_gen
import csv
import logging

logging.getLogger().setLevel(logging.WARNING)

#csv_file = open('stride_result.csv', mode='w')
#csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

strided_automatas = []
automatas = pickle.load(open('../snort1-20.pkl', 'rb'))
# automatas = atma.parse_anml_file(anml_path[AnmalZoo.TCP])
# automatas.remove_ors()
# automatas = automatas.get_connected_components_as_automatas()

for atm_idx, atm in enumerate(automatas[2:]):
    atm.remove_all_start_nodes()
    atm.remove_ors()
    #print atm.get_summary()

    atm2 = atm.get_single_stride_graph()
    atm2.make_homogenous()
    atm4 = atm2.get_single_stride_graph()
    # atm4.make_homogenous()
    #atm8 = atm4.get_single_stride_graph()
    #atm16 = atm8.get_single_stride_graph()
    atm4.make_homogenous()
    minimize_automata(atm4, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=False)
    print atm4.get_summary()
    for n in atm4.nodes:
        if n.start_type==StartType.fake_root:
            continue
        s = n.is_symbolset_splitable()
        if not s:
            print atm_idx, n.symbols

