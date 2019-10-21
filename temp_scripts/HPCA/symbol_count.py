import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv

ds = [a for a in AnmalZoo]


results = {}


for uat in ds:
    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    print uat
    for atm in automatas:
        #atm.remove_all_start_nodes()
        for node in atm.nodes:
            if node.is_fake:
                continue

            sym_len = len(list(node.symbols.points))
            results[sym_len] = results.get(sym_len, 0) + 1


for k, v in results.items():
    print " %d symbols count = %d" %(k, v)


