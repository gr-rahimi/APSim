import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType, get_bit_automaton, get_strided_automata
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream, get_equivalent_symbols, replace_equivalent_symbols
import math
from automata.HDL.hdl_generator import test_compressor
import logging
logging.basicConfig(level=logging.DEBUG)

#automatas = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
#automatas.remove_ors()
#automatas = automatas.get_connected_components_as_automatas()

automatas=pickle.load(open('Snort1-10.pkl','rb'))

org_atm = automatas[0]
print org_atm.get_summary(logo='original')
org_atm.draw_graph('original.svg')

bit_atm=get_bit_automaton(atm=org_atm, original_bit_width=8)
bit_atm.draw_graph('bitwise.svg')
print bit_atm.get_summary(logo='bitwise')

strided_b_atm=get_strided_automata(atm=bit_atm, stride_value=16, is_scalar=True, base_value=2)
print strided_b_atm.get_summary(logo='strided bitwise')
#strided_b_atm.draw_graph('strided.svg')

strided_b_atm.make_homogenous()
print strided_b_atm.get_summary(logo='homogeneous')
#strided_b_atm.draw_graph('homogeneous.svg', draw_edge_label=False)


minimize_automata(strided_b_atm)
print strided_b_atm.get_summary()
strided_b_atm.draw_graph('minimized.svg')

compare_input(True, True, False, None, org_atm, strided_b_atm)
exit(0)

for s in snort50:

    stride_dict_list = []
    for i in range(4):

        symbol_dict, symbol_dictionary_list = get_equivalent_symbols([s])

        if i == 0:
            initial_dic = symbol_dict
            initial_bits = int(math.ceil(math.log(max(initial_dic.values()), 2)))
            width_list=[initial_bits]
            test_compressor(original_width=8,
                            byte_trans_map=initial_dic,
                            byte_map_width=initial_bits,
                            translation_list=[],
                            idx=0,
                            width_list=[],
                            initial_width=initial_bits,
                            output_width=initial_bits)
            replace_equivalent_symbols(symbol_dictionary_list, [s])
        else:
            stride_dict_list.append(symbol_dict)
            width_list.append(int(math.ceil(math.log(max(symbol_dict.values()), 2))))
            print len(set(symbol_dict.values()))
            replace_equivalent_symbols(symbol_dictionary_list, [s])
            test_compressor(original_width=pow(2, i) * 8,
                            byte_trans_map=initial_dic,
                            byte_map_width=initial_bits,
                            translation_list=stride_dict_list,
                            idx=0,
                            width_list=width_list,
                            initial_width=pow(2,i) * initial_bits,
                            output_width=width_list[-1])

        s = s.get_single_stride_graph()
        s.make_homogenous()
        minimize_automata(s)



for i in range(3):
    new_total =[]
    for s in total_atms:
        s = s.get_single_stride_graph()
        s.make_homogenous()
        minimize_automata(s)
        new_total.append(s)

    total_atms = new_total

    symbol_dict, symbol_dictionary_list = get_equivalent_symbols(total_atms)
    print len(set(symbol_dict.values()))
    replace_equivalent_symbols(symbol_dictionary_list, total_atms)













