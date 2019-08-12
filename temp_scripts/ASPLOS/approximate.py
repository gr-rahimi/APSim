import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo, input_path
from automata.utility.utility import minimize_automata, get_approximate_automata, automata_run_stat
import csv

ds = [a for a in AnmalZoo]
ds = [AnmalZoo.Snort]
approximate_ratio = 16



for uat in ds:
    automatas = atma.parse_anml_file(anml_path[uat])
    translation_dic = {x: x / approximate_ratio for x in range(automatas.max_val_dim + 1)}

    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    for atm in automatas:
        print atm.get_summary(logo="before approximation")
        run_result = automata_run_stat(atm=atm, file_path=input_path[uat], cycle_detail=True, bytes_per_dim=1)
        print "\n results for real automata:", run_result

        atm.set_all_symbols_mutation(False)
        appr_automata = get_approximate_automata(atm=atm, translation_dic=translation_dic,
                                                max_val_dim=atm.max_val_dim / approximate_ratio)
        minimize_automata(automata=appr_automata)
        print appr_automata.get_summary(logo="after approximation")
        appr_run_result = automata_run_stat(atm=appr_automata, file_path=input_path[uat], cycle_detail=True, bytes_per_dim=1,
                                       translation_dic=translation_dic)
        print "\n results for approximate automata:", appr_run_result

        exit(0)




