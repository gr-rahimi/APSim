import logging
from automata.automata_network import StartType
import automata
from automata.utility.utility import minimize_automata
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo

logging.getLogger().setLevel(logging.WARNING)

#csv_file = open('stride_result.csv', mode='w')
#csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)


automatas = automata.parse_anml_file(anml_path[AnmalZoo.Dotstar09])
automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()

old_nodes ,new_nodes = 0, 0
for atm_idx, atm in enumerate(automatas[:50]):
    atm.remove_ors()
    print atm.get_summary()
    old_nodes += atm.nodes_count

    atm2 = atm.get_single_stride_graph()
    atm2.make_homogenous(plus_src=False)
    #atm2.make_parentbased_homogeneous()
    atm4 = atm2.get_single_stride_graph()
    atm4.make_homogenous(plus_src=False)
    minimize_automata(atm4)
    atm4.fix_split_all()
    new_nodes += atm4.nodes_count
    #atm4.fix_split_all()

    print atm4.get_summary()
    for n in atm4.nodes:
        if n.is_fake:
            continue
        s = n.is_symbolset_splitable()
        if not s:
            print atm_idx
            exit(1)

print old_nodes, new_nodes

