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
#automatas = pickle.load(open('snort1-20.pkl', 'rb'))
automatas = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()



for atm_idx, atm in enumerate(automatas[:400]):
    atm.remove_all_start_nodes()
    atm.remove_ors()
    print atm.get_summary()

    atm2 = atm.get_single_stride_graph()
    # atm2.make_homogenous()
    atm4 = atm2.get_single_stride_graph()
    # atm4.make_homogenous()
    atm8 = atm4.get_single_stride_graph()
    atm8.make_homogenous()
    minimize_automata(atm8, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=True)
    print atm8.get_summary()

    strided_automatas.append(atm)


hd_gen.generate_full_lut(strided_automatas, single_out=False, before_match_reg=False, after_match_reg=False,
                         ste_type=1, folder_name='Snort0-199', use_bram=False, bram_criteria=lambda n: len(n.symbols) > 8)

#hd_gen.generate_full_lut(strided_automatas, single_out=False, before_match_reg=False, after_match_reg=False,
#                         ste_type=1, folder_name='Snort20', use_bram=True, bram_criteria=lambda n: len(n.symbols) > 8)

