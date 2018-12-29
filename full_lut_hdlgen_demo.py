import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream, draw_symbols_len_histogram
import automata.HDL.hdl_generator as hd_gen
import csv
import logging

#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering
under_process_atm = AnmalZoo.Snort
target_stride_val = 1
#automatas = atma.parse_anml_file(anml_path[under_process_atm])
automatas = pickle.load(open('snort1-20.pkl', 'rb'))
#automatas.remove_ors()
#automatas = automatas.get_connected_components_as_automatas()
exempt_ids = {1411}
processed_atms = []
number_of_stages = 10

for atm_idx, atm in enumerate(automatas):
    if atm_idx in exempt_ids:
        continue
    print 'processing automata', atm_idx, 'from ', len(automatas)
    atm.remove_all_start_nodes()
    atm.remove_ors()

    for stride_val in range(target_stride_val):
        atm=atm.get_single_stride_graph()

    if atm.is_homogeneous is not True:
        atm.make_homogenous()

    minimize_automata(atm, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=True)

    processed_atms.append(atm)

atms_per_stage = len(processed_atms) / number_of_stages

hd_gen.generate_full_lut([processed_atms[i:i+atms_per_stage] for i in range(0,len(processed_atms), atms_per_stage)],
                         single_out=False, before_match_reg=False, after_match_reg=False,
                         ste_type=1, folder_name=str(under_process_atm), use_bram=False, bram_criteria=lambda n: len(n.symbols) > 8)

# hd_gen.generate_full_lut(processed_atms, single_out=False, before_match_reg=False, after_match_reg=False,
#                          ste_type=2, folder_name=str(under_process_atm), use_bram=False, bram_criteria=lambda n: len(n.symbols) > 8)
#
# hd_gen.generate_full_lut(processed_atms, single_out=False, before_match_reg=False, after_match_reg=True,
#                          ste_type=1, folder_name=str(under_process_atm), use_bram=False, bram_criteria=lambda n: len(n.symbols) > 8)
#
# hd_gen.generate_full_lut(processed_atms, single_out=False, before_match_reg=False, after_match_reg=True,
#                          ste_type=2, folder_name=str(under_process_atm), use_bram=False, bram_criteria=lambda n: len(n.symbols) > 8)
