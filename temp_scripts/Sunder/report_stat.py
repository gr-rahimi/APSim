import os
import fcntl

import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo, input_path10M
from automata.utility.utility import minimize_automata, get_approximate_automata, automata_run_stat
from automata.utility import total_reports, reports_per_cycle, total_active_states, actives_per_cycle, reports_in_cycle
from multiprocessing.dummy import Pool as ThreadPool

import cProfile

per_cc = False  # report results per connected components
iterations = 1000000

def process_single_ds(uat):
    automatas = atma.parse_anml_file(anml_path[uat])
    real_final = []  # these lists keep number of reports for each CC

    automatas.remove_ors()
    if per_cc:
        automatas = automatas.get_connected_components_as_automatas()
    else:
        automatas = [automatas]

    with open(str(uat) + '.txt', "w+") as f:
        pass

    for atm in automatas:
        atm.set_all_symbols_mutation(False)

        run_result = automata_run_stat(atm=atm, file_path=input_path10M[uat], cycle_detail=False, report_detail=True,
                                       bytes_per_dim=1, iterations=iterations)
        real_final.append(run_result[total_reports])


        with open(str(uat) + '.txt', "a") as f:
            print >> f, "total reports: " + str(real_final[-1])
            print >> f, "real nodes count: " + str(atm.nodes_count)
            print >> f, "reports per cycle: " + str(run_result[reports_per_cycle])
            print >> f, "reports in cycle: " + str(run_result[reports_in_cycle])
            print >>f, "----------------------------------------------------------------"


    with open('summary.txt', "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print >> f, "----------------------------------------------------------------"
        print >>f, str(uat)
        print >>f, "total cycles: " + str(iterations)
        print >> f, "total number of reports: " + str(sum(real_final))
        print >> f, "total number of report cycles: " + str(sum(run_result[reports_in_cycle]))
        print >> f, "average number of reports per cycle: " + str(sum(real_final) / float(iterations))
        print >> f, "average number of report  cycle: " + str(sum(run_result[reports_in_cycle]) / float(iterations))
        print >> f, "----------------------------------------------------------------"
        fcntl.flock(f, fcntl.LOCK_UN)


def cp_wrapper():
    process_single_ds(AnmalZoo.Snort)

if __name__ == '__main__':

    #cProfile.run('cp_wrapper()')
    #exit(0)

    #process_single_ds(AnmalZoo.Snort)
    #exit(0)

    with open('summary.txt', "w+") as f:
        pass

    ds = [a for a in AnmalZoo]
    ds.remove(AnmalZoo.Hamming)
    ds.remove(AnmalZoo.Levenshtein)
    ds.remove(AnmalZoo.EntityResolution)
    ds.remove(AnmalZoo.Dotstar)
    ds.remove(AnmalZoo.PowerEN)
    ds.remove(AnmalZoo.Brill)
    ds.remove(AnmalZoo.RandomForest)
    ds.remove(AnmalZoo.Dotstar03)
    ds.remove(AnmalZoo.Dotstar06)
    ds.remove(AnmalZoo.Dotstar09)
    ds.remove(AnmalZoo.Protomata)
    ds.remove(AnmalZoo.Ranges05)
    ds.remove(AnmalZoo.Bro217)
    ds.remove(AnmalZoo.Ranges1)
    ds.remove(AnmalZoo.ExactMath)
    ds.remove(AnmalZoo.Custom)
    ds.remove(AnmalZoo.TCP)
    ds.remove(AnmalZoo.Synthetic_BlockRings)

    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()
