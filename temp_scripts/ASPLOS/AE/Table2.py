import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
from multiprocessing.dummy import Pool as ThreadPool
import threading

global_lock = threading.Lock()

def process_single_ds(uat):

    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()


    uat_count = len(automatas)


    n_states = 0.0
    n_edges = 0.0
    total_sym = 0.0

    for atm_idx, atm in enumerate(automatas):

        minimize_automata(atm)

        all_nodes = filter(lambda n:n.id != 0, atm.nodes)  # filter fake root
        all_nodes_symbols_len_count = [len(list(n.symbols.points)) for n in all_nodes]

        n_s = atm.nodes_count
        n_states += n_s

        n_e = atm.edges_count
        n_edges += n_e

        t_s = sum(all_nodes_symbols_len_count)
        total_sym += t_s

    global_lock.acquire()
    print str(uat), "\t nodes count = ", n_states, "\tedges count = ", n_edges, "\tAvg node degree = ",\
        2 * n_edges / n_states, "\tAvg symbol size = ", total_sym/n_states

    global_lock.release()

    return True



if __name__ == '__main__':


    ds = [AnmalZoo.Brill, AnmalZoo.Bro217, AnmalZoo.Dotstar03,
          AnmalZoo.Dotstar06, AnmalZoo.Dotstar09, AnmalZoo.ExactMath,
          AnmalZoo.PowerEN, AnmalZoo.Protomata, AnmalZoo.Ranges05,
          AnmalZoo.Ranges1, AnmalZoo.Snort, AnmalZoo.TCP, AnmalZoo.Hamming,
          AnmalZoo.Levenshtein, AnmalZoo.EntityResolution, AnmalZoo.Fermi,
          AnmalZoo.RandomForest, AnmalZoo.SPM, AnmalZoo.Synthetic_BlockRings, AnmalZoo.Synthetic_CoreRings]



    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()