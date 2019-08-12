import math
from sortedcontainers.sortedlist import SortedList
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo

fcb_size = 256  # size of the local switches. we assume GS has also the same size
fcb_to_gs = 16  # number of wires from local switches to GS

bigest_component_size = fcb_size / fcb_to_gs * fcb_size

ds = [a for a in AnmalZoo]

for uat in ds:
    r = SortedList()
    big128 , big256 = 0, 0


    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    for atm in automatas:
        nc = atm.nodes_count
        if nc >= 128:
            big128 += 1
        if nc >= 256:
            big256 += 1

        if nc > bigest_component_size:
            print "this NFA can not be fit:", uat
            break

        if r and nc <= r[-1]:  # can be packed
            cand_residual = r.pop(-1)
            r.add(cand_residual - nc)
        else:  # new fcb
            r.add(bigest_component_size - nc)

    print "uat %s needs %d connected local switches each with (%d,%d) size. There are %d nodes not being assigned." \
          " It has %d CCs bigger than 128 and %d CCs bigger than 256"\
          %(uat, len(r), fcb_size, fcb_size, sum(r[:-1]), big128, big256)

















