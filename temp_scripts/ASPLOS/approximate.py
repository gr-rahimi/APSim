import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo, input_path
from automata.utility.utility import minimize_automata, get_approximate_automata, automata_run_stat
from automata.utility import total_reports, reports_per_cycle, total_active_states, actives_per_cycle
from multiprocessing.dummy import Pool as ThreadPool

uat_count = 10
approximate_ratio = 16

def process_single_ds(uat):
    automatas = atma.parse_anml_file(anml_path[uat])
    approximate_final, real_final = [], []  # these lists keep number of reports for each CC
    real_states, appr_states = 0, 0  # these integers count number of states
    translation_dic = {x: x % approximate_ratio for x in range(automatas.max_val_dim + 1)}

    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()
    with open(str(uat) + '.txt', "w+") as f:
        pass
    for atm in automatas[:uat_count]:
        real_states+= atm.nodes_count
        atm.set_all_symbols_mutation(False)

        appr_automata = get_approximate_automata(atm=atm, translation_dic=translation_dic,
                                                max_val_dim=atm.max_val_dim / approximate_ratio)
        minimize_automata(automata=appr_automata)
        appr_states += appr_automata.nodes_count

        run_result = automata_run_stat(atm=atm, file_path=input_path[uat], cycle_detail=True, report_detail=False, bytes_per_dim=1)
        real_final.append(run_result[total_reports])
        appr_run_result = automata_run_stat(atm=appr_automata, file_path=input_path[uat], cycle_detail=True, report_detail=False, bytes_per_dim=1,
                                       translation_dic=translation_dic)
        approximate_final.append(appr_run_result[total_reports])

        with open(str(uat) + '.txt', "a") as f:
            print >> f, "real reports: " + str(real_final[-1])
            print >> f, "approximate reports: " + str(approximate_final[-1])
            print >> f, "real nodes count: " + str(atm.nodes_count)
            print >> f, "approximate nodes count:" + str(appr_automata.nodes_count)
            print >>f, "----------------------------------------------------------------"


    with open(str(uat) + '.ttxt', "a") as f:
        print >>f, "***************sum*******************"
        print >>f, "real reports: " + str(sum(real_final))
        print >>f, "approximate reports: " + str(sum(approximate_final))
        print >>f, "real nodes count: " + str(real_states)
        print >>f, "approximate nodes count:" + str(appr_states)

if __name__ == '__main__':

    ds = [a for a in AnmalZoo]
    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()
