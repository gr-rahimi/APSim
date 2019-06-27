import shutil
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging
from multiprocessing.dummy import Pool as ThreadPool
import fcntl

#logging.basicConfig(level=logging.DEBUG)


out_dir = '../../Results/Stat/'

def process_single_ds(uat):

    #uat = AnmalZoo.Ranges05

    return_result = {}
    result_dir = out_dir + str(uat)

    shutil.rmtree(result_dir, ignore_errors=True)
    os.mkdir(result_dir)
    exempts = {(AnmalZoo.Snort, 1411)}

    max_target_stride = 2
    uat_count = 200

    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()


    #uat_count = len(automatas)  # comment this to test a subset of automatons defined in uat_count

    automatas = automatas[:uat_count]
    uat_count = len(automatas)



    filed_names = ['number_of_states', 'number_of_edges', 'max_fan_in', 'max_fan_out',
                   'max_symbol_len', 'min_symbol_len', 'total_sym']
    for hom_between, is_Bram in [(False, False),  (False, True)]:

        for stride_val in range(max_target_stride + 1):
            n_states = 0.0
            n_edges = 0.0
            max_fan_in = 0.0
            max_fan_out = 0.0
            max_sym_len = 0.0
            min_sym_len = 0.0
            total_sym = 0.0

            with open(result_dir + '/S' + str(stride_val) + '_' + str(uat_count) +
                      'is_HNH' + str(hom_between) + 'is_Bram' + str(is_Bram) + 'len' + str(len(automatas)) +'.csv', 'w') as csv_file:
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

                    minimize_automata(atm)

                    if is_Bram is True and hom_between is False:
                        atm.fix_split_all()

                    all_nodes = filter(lambda n:n.id != 0, atm.nodes)  # filter fake root
                    all_nodes_symbols_len_count = [len(n.symbols) for n in all_nodes]

                    n_s = atm.nodes_count
                    n_states += n_s

                    n_e = atm.edges_count
                    n_edges += n_e

                    m_f_i = atm.max_STE_in_degree()
                    max_fan_in += m_f_i

                    m_f_o = atm.max_STE_out_degree()
                    max_fan_out += m_f_o

                    mx_s_l = max(all_nodes_symbols_len_count)
                    max_sym_len += mx_s_l

                    mn_s_l = min(all_nodes_symbols_len_count)
                    min_sym_len += mn_s_l

                    t_s = sum(all_nodes_symbols_len_count)
                    total_sym += t_s

                    csv_writer.writerow([n_s, n_e, m_f_i, m_f_o, mx_s_l, mn_s_l, t_s])

            n_states /= uat_count
            n_edges /= uat_count
            max_fan_in /= uat_count
            max_fan_out /= uat_count
            max_sym_len /= uat_count
            min_sym_len /= uat_count
            total_sym /= uat_count

            return_result[(is_Bram, stride_val)] = (n_states,
                                                    n_edges,
                                                    max_fan_in,
                                                    max_fan_out,
                                                    max_sym_len,
                                                    min_sym_len,
                                                    total_sym)

            with open(out_dir + 'summary.txt', 'a+') as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                to_w_lns =[]
                to_w_lns.append(str(uat) + "L" + str(uat_count) + "S" + str(stride_val) + "BRam" + str(is_Bram) + "\n")
                to_w_lns.append("    average number of states = " + str(n_states) + "\n")
                to_w_lns.append("    average number of edges = " + str(n_edges) + "\n")
                to_w_lns.append("    average max fan-in = " + str(max_fan_in) + "\n")
                to_w_lns.append("    average max fan-out = " + str(max_fan_out) + "\n")
                to_w_lns.append("    average max sym-len = " + str(max_sym_len) + "\n")
                to_w_lns.append("    average min sym-len = " + str(min_sym_len) + "\n")
                to_w_lns.append("    average total_sym-len = " + str(total_sym) + "\n")
                f.writelines(to_w_lns)
                fcntl.flock(f, fcntl.LOCK_UN)

    return uat, return_result



if __name__ == '__main__':

    #os.remove(out_dir + 'summary.txt')

    old_ds = [AnmalZoo.Snort,
          AnmalZoo.RandomForest,
          AnmalZoo.Ranges1,
          AnmalZoo.PowerEN,
          AnmalZoo.Protomata,
          AnmalZoo.Dotstar03,
          AnmalZoo.ExactMath,
          AnmalZoo.SPM,
          AnmalZoo.Custom]

    ds = [a for a in AnmalZoo if a not in old_ds]

    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()



