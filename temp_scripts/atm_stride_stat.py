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
under_process_atms = [AnmalZoo.RandomForest, AnmalZoo.Snort]
exempts = {(AnmalZoo.Snort, 1411)}

for uat in under_process_atms:

    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    filed_names = ['number_of_states', 'number_of_edges', 'max_fan_in', 'max_fan_out',
                   'max_symbol_len', 'min_symbol_len', 'total_sym']
    for stride_val in range(4):
        with open(str(uat)+'_'+str(stride_val)+'.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(filed_names)
            for atm_idx, atm in enumerate(automatas):
                if (uat, atm_idx) in exempts:
                    continue
                print 'processing {0} stride{3} automata {1} from {2}'.format(uat, atm_idx, len(automatas), stride_val)
                atm.remove_ors()
                for _ in range(stride_val):
                    atm = atm.get_single_stride_graph()
                if not atm.is_homogeneous:
                    atm.make_homogenous()
                minimize_automata(atm, merge_reports=True, same_residuals_only=True, same_report_code=True,
                                  combine_symbols=True)

                all_nodes = filter(lambda n: n.id != 0 , atm.nodes)  # filter fake root
                all_nodes_symbols_len_count = [len(n.symbols) for n in all_nodes]

                csv_writer.writerow([atm.nodes_count, atm.get_number_of_edges(), atm.max_STE_in_degree(),
                                     atm.max_STE_out_degree(), max(all_nodes_symbols_len_count),
                                     min(all_nodes_symbols_len_count), sum(all_nodes_symbols_len_count)])

