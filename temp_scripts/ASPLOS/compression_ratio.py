import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata, get_equivalent_symbols
import automata.HDL.hdl_generator as hd_gen
import math
import random
from automata.elemnts.ste import PackedInput
from multiprocessing.dummy import Pool as ThreadPool
import fcntl
import numpy as np
#logging.basicConfig(level=logging.DEBUG)



def process_single_ds(uat):


    with open(str(uat) + '.txt', "w+") as f:
        pass
    all_automata = atma.parse_anml_file(anml_path[uat])
    all_automata.remove_ors()
    automatas = all_automata.get_connected_components_as_automatas()
    number_of_autoamtas = len(automatas)

    if len(automatas) > number_of_autoamtas:
        #automatas = random.sample(automatas, number_of_autoamtas)
        automatas = automatas[:number_of_autoamtas]


    number_of_autoamtas = len(automatas)
    total_bit_len = []
    total_bit_hist = {}

    for atm_idx, atm in enumerate(automatas):

        print 'processing {0} automata {1} from {2}'.format(uat, atm_idx + 1, number_of_autoamtas)

        bc_sym_dict = get_equivalent_symbols([atm], replace=True)
        bc_bits_len = int(math.ceil(math.log(max(bc_sym_dict.values()), 2)))

        total_bit_len.append(bc_bits_len)
        total_bit_hist[bc_bits_len] = total_bit_hist.get(bc_bits_len, 0) + 1

        #with open(str(uat) + '.txt', "a") as f:
        #   print >> f, "compressed bit length: " + str(bc_bits_len)

    with open('compression_ratio_summary.txt', "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print >> f, "----------------------------------------------------------------"
        print >> f, str(uat)
        print >> f, "AVG:" + str(sum(total_bit_len) / float(number_of_autoamtas))
        print >> f, "STD:" + str(np.std(total_bit_len))
        print >> f, "---------Histogram---------"
        for k, v in total_bit_hist.iteritems():
            print >> f, "bit len {0} = {1}".format(k, v)
        print >> f, "----------------------------------------------------------------"
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == '__main__':

    ds = [a for a in AnmalZoo if a not in [AnmalZoo.Levenshtein, AnmalZoo.PowerEN, AnmalZoo.Hamming, AnmalZoo.Brill,
                                           AnmalZoo.Fermi, AnmalZoo.Protomata, AnmalZoo.Dotstar, AnmalZoo.Dotstar03,
                                           AnmalZoo.Synthetic_BlockRings, AnmalZoo.Dotstar06, AnmalZoo.Dotstar09,
                                           AnmalZoo.Ranges05, AnmalZoo.Bro217, AnmalZoo.Ranges1, AnmalZoo.ExactMath,
                                           AnmalZoo.Custom]]

    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()