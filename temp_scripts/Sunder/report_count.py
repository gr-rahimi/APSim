import os
import fcntl

import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo, input_path10M
from automata.utility.utility import minimize_automata, get_approximate_automata, automata_run_stat
from automata.utility import total_reports, reports_per_cycle, total_active_states, actives_per_cycle, reports_in_cycle
from multiprocessing.dummy import Pool as ThreadPool

import cProfile

per_cc = True  # report results per connected components
iterations = 1000000

def process_single_ds(uat):
    try:
        automatas = atma.parse_anml_file(anml_path[uat])

        automatas.remove_ors()
        if per_cc:
            automatas = automatas.get_connected_components_as_automatas()
        else:
            automatas = [automatas]

        report_count, total_states = 0, 0
        for atm in automatas:

            report_count += atm.number_of_report_nodes
            total_states += atm.nodes_count

        with open('report_count_summary.txt', "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print >> f, "----------------------------------------------------------------"
            print >>f, str(uat)
            print >>f, "total nodes: " + str(total_states)
            print >> f, "total report nodes: " + str(report_count)
            print >> f, "----------------------------------------------------------------"
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print e


def cp_wrapper():
    process_single_ds(AnmalZoo.Snort)

if __name__ == '__main__':

    #cProfile.run('cp_wrapper()')
    #exit(0)

    #process_single_ds(AnmalZoo.Snort)
    #exit(0)

    with open('report_count_summary.txt', "w+") as f:
        pass

    ds = [a for a in AnmalZoo]
    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()
