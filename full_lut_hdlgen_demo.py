import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream, draw_symbols_len_histogram, get_equivalent_symbols, replace_equivalent_symbols
import utility
import automata.HDL.hdl_generator as hd_gen
import csv
import logging
import math
import random
import os.path
from automata.elemnts.ste import PackedInput


def reza_test(inp_dic, atm):
    eq_set = set()

    cands = [16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 30]


    for key, val in inp_dic.iteritems():
        if val in cands:
            eq_set.add(key)


    kh= None
    for e in eq_set:
        packed_pt  = PackedInput(e)
        result = [node.symbols.can_accept(packed_pt) for node in atm.nodes if node.id!=0]
        if kh == None:
            kh = result
        else:
            assert kh == result



random.seed=3

#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering
under_process_atms = [AnmalZoo.Hamming]
exempts = {(AnmalZoo.Snort, 1411)}
hom_between = True
number_of_autoamtas = 6
automata_per_stage = 3.0
use_compression = True
single_out=False
before_match_reg=False
after_match_reg=False
ste_type=1
use_bram=False

for uat in under_process_atms:
    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    if len(automatas) > number_of_autoamtas:
        #automatas = random.sample(automatas, number_of_autoamtas)
        automatas = automatas[:number_of_autoamtas]



    number_of_stages = math.ceil(len(automatas) / automata_per_stage)
    for stride_val in range(4):

        hdl_apth = hd_gen.get_hdl_folder_path(prefix=str(uat), number_of_atms=len(automatas), stride_value=stride_val,
                                              before_match_reg=before_match_reg, after_match_reg=after_match_reg,
                                              ste_type=ste_type, use_bram=use_bram)

        hd_gen.clean_and_make_path(hdl_apth)

        strided_automatas, bit_size,  = [], []
        for atm_idx, atm in enumerate(automatas):
            if (uat, atm_idx) in exempts:
                continue

            print 'processing {0} stride{3} automata {1} from {2}'.format(uat, atm_idx, len(automatas), stride_val)

            if use_compression:
                symbol_dict, symbol_dictionary_list = get_equivalent_symbols([atm])
                print 'number of first pipeline symbols', len(set(symbol_dict.values()))

                initial_dic = symbol_dict
                initial_bits = int(math.ceil(math.log(max(initial_dic.values()), 2)))
                width_list = [initial_bits]
                replace_equivalent_symbols(symbol_dictionary_list, [atm])
                bit_size.append(initial_bits)

                hd_gen.generate_compressors(original_width=8, byte_trans_map=symbol_dict, byte_map_width=initial_bits,
                                            translation_list=[], idx=atm_idx, width_list=[], initial_width=initial_bits,
                                            output_width=initial_bits, file_path=os.path.join(hdl_apth, 'compressor'+ str(atm_idx)+ '.v'))

            for _ in range(stride_val):
                atm = atm.get_single_stride_graph()
                if hom_between is True:
                    atm.make_homogenous()

            if not atm.is_homogeneous:
                atm.make_homogenous()
            minimize_automata(atm, merge_reports=True, same_residuals_only=True, same_report_code=True,
                            combine_symbols=True if hom_between is not True else False)

            strided_automatas.append(atm)

        atms_per_stage = int(math.ceil(len(strided_automatas) / float(number_of_stages)))

        hd_gen.generate_full_lut(
                                [strided_automatas[i:i + atms_per_stage]
                                 for i in range(0, len(strided_automatas), atms_per_stage)],
                                single_out=False, before_match_reg=False, after_match_reg=False,
                                ste_type=1, folder_name=hdl_apth, use_bram=False,
                                bram_criteria=lambda n: len(n.symbols) > 8 ,bit_feed_size=pow(2, stride_val)*8,
                                id_to_comp_dict=[{i:bs * pow(2, stride_val) for i, bs in zip(range(atms_per_stage), bit_size[j:j+atms_per_stage])}
                                                 for j in range(0, len(strided_automatas), atms_per_stage)] if use_compression else None,
                                comp_dict=[{atm.id:i for atm, i in zip(strided_automatas[j: j+atms_per_stage],
                                                             range(atms_per_stage))} for j in range(0, len(strided_automatas), atms_per_stage)] if use_compression else None,
                                use_compression=use_compression)
















