import shutil
import traceback
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging
from multiprocessing.dummy import Pool as ThreadPool
import fcntl

#logging.basicConfig(level=logging.DEBUG)


out_dir = '../../Results/BV_Stat32/'

def process_single_ds(uat):
    try:
        #uat = AnmalZoo.Ranges05

        return_result = {}
        result_dir = out_dir + str(uat)

        shutil.rmtree(result_dir, ignore_errors=True)
        os.mkdir(result_dir)
        exempts = {(AnmalZoo.Snort, 1411)}

        min_target_stride, max_target_stride = 3, 3
        uat_count = 200

        automatas = atma.parse_anml_file(anml_path[uat])
        automatas.remove_ors()
        automatas = automatas.get_connected_components_as_automatas()


        #uat_count = len(automatas)  # comment this to test a subset of automatons defined in uat_count

        automatas = automatas[:uat_count]
        uat_count = len(automatas)

        filed_names = ['number_of_states', 'number_of_edges', 'max_fan_in', 'max_fan_out',
                       'max_symbol_len', 'min_symbol_len', 'total_sym']
        for hom_between, is_Bram in [(False, True)]:
            n_states = [0.0 for _ in range(max_target_stride + 1)]
            n_edges = [0.0 for _ in range(max_target_stride + 1)]
            max_fan_in = [0.0 for _ in range(max_target_stride + 1)]
            max_fan_out = [0.0 for _ in range(max_target_stride + 1)]
            max_sym_len = [0.0 for _ in range(max_target_stride + 1)]
            min_sym_len = [0.0 for _ in range(max_target_stride + 1)]
            total_sym = [0.0 for _ in range(max_target_stride + 1)]

            csv_writers = []
            for i in range(max_target_stride + 1):
                f = open(result_dir + '/S' + str(i) + '_' + str(uat_count) + 'is_HNH' +
                         str(hom_between) + 'is_Bram' + str(is_Bram) + 'len' + str(uat_count) + '.csv', 'w')
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(filed_names)
                csv_writers.append(csv_writer)


            for atm_idx, atm in enumerate(automatas):
                b_atm = atma.automata_network.get_bit_automaton(atm, original_bit_width=atm.max_val_dim_bits_len)
                atm = atma.automata_network.get_strided_automata2(atm=b_atm,
                                                                  stride_value=4,
                                                                  is_scalar=True,
                                                                  base_value=2,
                                                                  add_residual=True)

                for stride_val in reversed(range(min_target_stride, max_target_stride + 1)):
                    if (uat, atm_idx) in exempts:
                        continue
                    print 'processing {0} stride {3} number {1} from {2}'.format(uat, atm_idx, uat_count, stride_val)
                    s_atm = atm

                    for _ in range(stride_val):
                        if s_atm is atm:
                            s_atm = atm.get_single_stride_graph()
                        else:
                            s_atm = s_atm.get_single_stride_graph()

                    if s_atm.is_homogeneous is False:
                        s_atm.make_homogenous()

                    minimize_automata(s_atm)

                    if is_Bram is True and hom_between is False:
                        s_atm.fix_split_all()


                    all_nodes = filter(lambda n:n.id != 0, s_atm.nodes)  # filter fake root
                    all_nodes_symbols_len_count = [len(n.symbols) for n in all_nodes]

                    n_s = s_atm.nodes_count
                    n_states[stride_val] += n_s

                    n_e = s_atm.edges_count
                    n_edges[stride_val] += n_e

                    m_f_i = s_atm.max_STE_in_degree()
                    max_fan_in[stride_val] += m_f_i

                    m_f_o = s_atm.max_STE_out_degree()
                    max_fan_out[stride_val] += m_f_o

                    mx_s_l = max(all_nodes_symbols_len_count)
                    max_sym_len[stride_val] += mx_s_l

                    mn_s_l = min(all_nodes_symbols_len_count)
                    min_sym_len[stride_val] += mn_s_l

                    t_s = sum(all_nodes_symbols_len_count)
                    total_sym[stride_val] += t_s

                    csv_writers[stride_val].writerow([n_s, n_e, m_f_i, m_f_o, mx_s_l, mn_s_l, t_s])

            del csv_writers
            for i in range(max_target_stride + 1):
                n_states[i] /= uat_count
                n_edges[i] /= uat_count
                max_fan_in[i] /= uat_count
                max_fan_out[i] /= uat_count
                max_sym_len[i] /= uat_count
                min_sym_len[i] /= uat_count
                total_sym[i] /= uat_count

                return_result[(is_Bram, i)] = (n_states,
                                                    n_edges,
                                                    max_fan_in,
                                                    max_fan_out,
                                                    max_sym_len,
                                                    min_sym_len,
                                                    total_sym)

                with open(out_dir + 'summary.txt', 'a+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    to_w_lns =[]
                    to_w_lns.append(str(uat) + "L" + str(uat_count) + "S" + str(i) + "BRam" + str(is_Bram) + "\n")
                    to_w_lns.append("    average number of states = " + str(n_states[i]) + "\n")
                    to_w_lns.append("    average number of edges = " + str(n_edges[i]) + "\n")
                    to_w_lns.append("    average max fan-in = " + str(max_fan_in[i]) + "\n")
                    to_w_lns.append("    average max fan-out = " + str(max_fan_out[i]) + "\n")
                    to_w_lns.append("    average max sym-len = " + str(max_sym_len[i]) + "\n")
                    to_w_lns.append("    average min sym-len = " + str(min_sym_len[i]) + "\n")
                    to_w_lns.append("    average total_sym-len = " + str(total_sym[i]) + "\n")
                    f.writelines(to_w_lns)
                    fcntl.flock(f, fcntl.LOCK_UN)

        return uat, return_result
    except Exception, e:
        tracebackString = traceback.format_exc(e)
        print tracebackString
        raise StandardError, "\n\nError occurred. Original traceback is\n%s\n" %(tracebackString)

if __name__ == '__main__':
    open(out_dir + 'summary.txt', 'w')  # creat a new file

    old_ds = []

    ds = [a for a in AnmalZoo if a not in old_ds]

    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()



