import shutil
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging

logging.basicConfig(level=logging.DEBUG)


uat = AnmalZoo.Snort
out_dir = '../Results/BV_Stat/'
result_dir = out_dir + str(uat)

shutil.rmtree(result_dir, ignore_errors=True)
os.mkdir(result_dir)
exempts = {(AnmalZoo.Snort, 1411)}

max_target_stride = 3
uat_count = 200

automatas = atma.parse_anml_file(anml_path[uat])
automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()


#uat_count = len(automatas)  # comment this to test a subset of automatons defined in uat_count

automatas = automatas[:uat_count]
uat_count = len(automatas)

filed_names = ['number_of_states', 'number_of_edges', 'max_fan_in', 'max_fan_out',
               'max_symbol_len', 'min_symbol_len', 'total_sym']
for hom_between, is_Bram in [(False, True), (False, False)]:

    for stride_val in range(max_target_stride + 1):

        with open(result_dir + '/S' + str(stride_val) + '_' + str(uat_count) +
                  'is_HNH' + str(hom_between) + 'is_Bram' + str(is_Bram) + 'len' + str(len(automatas)) +'.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(filed_names)

            for atm_idx, atm in enumerate(automatas):
                if (uat, atm_idx) in exempts:
                    continue
                print 'processing {0} stride {3} number {1} from {2}'.format(uat, atm_idx, uat_count, stride_val)
                b_atm = atma.automata_network.get_bit_automaton(atm, original_bit_width=atm.max_val_dim_bits_len)
                atm = atma.automata_network.get_strided_automata2(atm=b_atm,
                                                                      stride_value=4,
                                                                      is_scalar=True,
                                                                      base_value=2,
                                                                      add_residual=True)
                for _ in range(stride_val):
                    atm = atm.get_single_stride_graph()

                if atm.is_homogeneous is False:
                    atm.make_homogenous()

                minimize_automata(atm, combine_equal_syms_only=is_Bram)

                if is_Bram is True and hom_between is False:
                    atm.fix_split_all()

                all_nodes = filter(lambda n:n.id != 0, atm.nodes)  # filter fake root
                all_nodes_symbols_len_count = [len(n.symbols) for n in all_nodes]

                csv_writer.writerow([atm.nodes_count, atm.edges_count, atm.max_STE_in_degree(),
                                     atm.max_STE_out_degree(), max(all_nodes_symbols_len_count),
                                     min(all_nodes_symbols_len_count), sum(all_nodes_symbols_len_count)])

