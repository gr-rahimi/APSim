import shutil
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging

logging.basicConfig(level=logging.DEBUG)


uat = AnmalZoo.Hamming

shutil.rmtree('../Results/Stat/' + str(uat), ignore_errors=True)
os.mkdir('../Results/Stat/' + str(uat))
exempts = {(AnmalZoo.Snort, 1411)}

max_target_stride = 3
uat_count = 1

automatas = atma.parse_anml_file(anml_path[uat])
automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()


#uat_count = len(automatas)  # comment this to test a subset of automatons defined in uat_count

automatas = automatas[:uat_count]
uat_count = len(automatas)



###########
# tmp_atm = automatas[0]
#
# src_ste = tmp_atm.get_STE_by_id(5)
# dst_ste = tmp_atm.get_STE_by_id(1)
# tmp_atm.delete_edge(src_ste, dst_ste)
#
# src_ste = tmp_atm.get_STE_by_id(6)
# dst_ste = tmp_atm.get_STE_by_id(2)
# tmp_atm.delete_edge(src_ste, dst_ste)
#
# src_ste = tmp_atm.get_STE_by_id(7)
# dst_ste = tmp_atm.get_STE_by_id(3)
# tmp_atm.delete_edge(src_ste, dst_ste)
#
# src_ste = tmp_atm.get_STE_by_id(8)
# dst_ste = tmp_atm.get_STE_by_id(4)
# tmp_atm.delete_edge(src_ste, dst_ste)
#
# tmp_atm.draw_graph('spm0trimmed.svg')


###########


filed_names = ['number_of_states', 'number_of_edges', 'max_fan_in', 'max_fan_out',
               'max_symbol_len', 'min_symbol_len', 'total_sym']
for hom_between, is_Bram in [(False, False),  (False, True), (True, True)]:

    for stride_val in range(max_target_stride + 1):

        with open('../Results/Stat/' + str(uat) + '/S' + str(stride_val) + '_' + str(uat_count) +
                  'is_HNH' + str(hom_between) + 'is_Bram' + str(is_Bram) + '.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(filed_names)

            for atm_idx, atm in enumerate(automatas):
                if (uat, atm_idx) in exempts:
                    continue
                print 'processing {0} stride {3} number {1} from {2}'.format(uat, atm_idx, uat_count, stride_val)

                for _ in range(stride_val):
                    if is_Bram is True and hom_between is True and atm.is_homogeneous is False:
                        atm.make_homogenous()
                        atm.make_parentbased_homogeneous()

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

