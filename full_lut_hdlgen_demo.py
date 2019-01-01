import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream, draw_symbols_len_histogram
import automata.HDL.hdl_generator as hd_gen
import csv
import logging
import math
import random

#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering
under_process_atms = [AnmalZoo.Bro217]
exempts = {(AnmalZoo.EntityResolution, 1411)}
number_of_stages = 10

number_of_autoamtas = 200

for uat in under_process_atms:
    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()
    if len(automatas) > number_of_autoamtas:
        automatas = random.sample(automatas, number_of_autoamtas)

    number_of_stages = math.ceil(len(automatas) / 50.0)
    for stride_val in range(3, 4):
        strided_automatas = []
        for atm_idx, atm in enumerate(automatas):
            if (uat, atm_idx) in exempts:
                continue
            print 'processing {0} stride{3} automata {1} from {2}'.format(uat, atm_idx, len(automatas), stride_val)

            for _ in range(stride_val):
                atm = atm.get_single_stride_graph()

            if not atm.is_homogeneous:
                atm.make_homogenous()

            minimize_automata(atm, merge_reports=True, same_residuals_only=True, same_report_code=True,
                          combine_symbols=True)

            strided_automatas.append(atm)

        atms_per_stage = int(math.ceil(len(strided_automatas) / float(number_of_stages)))

        hd_gen.generate_full_lut(
                    [strided_automatas[i:i + atms_per_stage] for i in range(0, len(strided_automatas), atms_per_stage)],
                     single_out=False, before_match_reg=False, after_match_reg=False,
                     ste_type=1, folder_name=str(uat), use_bram=False,
                     bram_criteria=lambda n: len(n.symbols) > 8)











