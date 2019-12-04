import shutil
import traceback
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
from multiprocessing.dummy import Pool as ThreadPool
import threading

global_lock = threading.Lock()

def process_single_ds(uat):
    try:

        uat_count = 200
        # number of total automata to be processed. Make this number smaller to get results faster for a subset

        automatas = atma.parse_anml_file(anml_path[uat])
        automatas.remove_ors()
        automatas = automatas.get_connected_components_as_automatas()

        automatas = automatas[:uat_count]
        # pick the number of automataon to be processed. comment this line if you want to process the whole benchmark

        target_bit_widths = [1, 2, 4, 16]
        # bitwidth to be calculated

        stats = [[0, 0] for _ in range(len(target_bit_widths))]

        for atm in automatas:
            b_atm = atma.automata_network.get_bit_automaton(atm, original_bit_width=atm.max_val_dim_bits_len)
            # generate bit automaton

            for tb_idx, tb in enumerate(target_bit_widths):
                if tb == 1:
                    atm = b_atm.clone()
                else:
                    atm = atma.automata_network.get_strided_automata2(atm=b_atm,
                                                                  stride_value=tb,
                                                                  is_scalar=True,
                                                                  base_value=2,
                                                                  add_residual=True)

                if atm.is_homogeneous is False:
                    atm.make_homogenous()

                minimize_automata(atm)

                n_s = atm.nodes_count
                n_e = atm.edges_count

                stats[tb_idx][0] += n_s
                stats[tb_idx][1] += n_e

        global_lock.acquire()
        print uat
        for tb_idx, tb in enumerate(target_bit_widths):
            print "bitwidth = ", tb, "number of states = ", stats[tb_idx][0], "number of edges = ", stats[tb_idx][1]

        global_lock.release()
        return uat, stats
    except Exception, e:
        tracebackString = traceback.format_exc(e)
        print tracebackString
        raise StandardError, "\n\nError occurred. Original traceback is\n%s\n" %(tracebackString)

if __name__ == '__main__':

    ds = [AnmalZoo.Brill, AnmalZoo.Bro217, AnmalZoo.Dotstar03,
          AnmalZoo.Dotstar06, AnmalZoo.Dotstar09, AnmalZoo.ExactMath,
          AnmalZoo.PowerEN, AnmalZoo.Protomata, AnmalZoo.Ranges05,
          AnmalZoo.Ranges1, AnmalZoo.Snort, AnmalZoo.TCP, AnmalZoo.Hamming,
          AnmalZoo.Levenshtein, AnmalZoo.EntityResolution, AnmalZoo.Fermi,
          AnmalZoo.RandomForest, AnmalZoo.SPM, AnmalZoo.Synthetic_BlockRings, AnmalZoo.Synthetic_CoreRings]

    ds = [AnmalZoo.ExactMath]
    # this will only calculate for ExactMatch benchmark. add benchmarks to this list for more


    thread_count = 8
    #  process benchmarks in parallel

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()