import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream, draw_symbols_len_histogram, replace_with_unified_symbol
import automata.HDL.hdl_generator as hd_gen
import csv
import logging

#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering

uat = AnmalZoo.TCP

automatas = atma.parse_anml_file(anml_path[uat])
automata_name = str(uat)

exempts = {(AnmalZoo.Snort, 1411)}

automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()

print("Number of automata: ", len(automatas))
print(automata_name)

filed_names = ['#States', '#Edges', 'max_fan_in', 'max_fan_out', 'total_sym']

for stride_val in range(4):
    n_states = 0
    n_edges = 0
    n_maxFanin = 0
    n_maxFanout = 0
    n_symbols = 0

    with open(automata_name + '_' + str(stride_val) + '.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(filed_names)
        for atm_idx, atm in enumerate(automatas):

            for _ in range(stride_val):
                atm = atm.get_single_stride_graph()

            if atm.is_homogeneous is False:
                atm.make_homogenous()

            minimize_automata(atm)
            atm.fix_split_all()

            all_nodes = filter(lambda n: n.is_fake is False, atm.nodes)  # filter fake root
            all_nodes_symbols_len_count = [len(n.symbols) for n in all_nodes]

            n_states += atm.nodes_count
            n_edges += atm.get_number_of_edges()
            n_maxFanin += atm.max_STE_in_degree()
            n_maxFanout += atm.max_STE_out_degree()
            n_symbols += sum(all_nodes_symbols_len_count)

            csv_writer.writerow([atm.nodes_count, atm.get_number_of_edges(), atm.max_STE_in_degree(), atm.max_STE_out_degree(), sum(all_nodes_symbols_len_count)])
        csv_writer.writerow([])
        csv_writer.writerow([n_states/len(automatas),n_edges/len(automatas),n_maxFanin/len(automatas),n_maxFanout/len(automatas),n_symbols/len(automatas)])
        print stride_val, ":\t", n_states/len(automatas), "\t", n_edges/len(automatas), "\t", n_maxFanin/len(automatas), "\t", n_maxFanout/len(automatas), "\t", n_symbols/len(automatas)