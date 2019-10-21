import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv

ds = [a for a in AnmalZoo]
ds = [AnmalZoo.Synthetic_BlockRings]


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

            if node.symbols.is_star(max_val=atm.max_val_dim):
                continue

            for sym in node.symbols.points:
                results[sym] = results.get(sym, 0) + 1


for k in sorted(results.keys()):
    print " %s symbols count = %d" % (k, results[k])


