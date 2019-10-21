import shutil
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging

logging.basicConfig(level=logging.DEBUG)


uat = AnmalZoo.Snort

automatas = atma.parse_anml_file(anml_path[uat])
automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()

for atm in automatas[7:8]:
    atm.draw_graph('orig.svg')
    #trad_s1 = atm.get_single_stride_graph()
    trad_s2 = atm.get_single_stride_graph()
    trad_s2.draw_graph('trad_before_homo.svg')
    trad_s2.make_homogenous(use_espresso=False)
    #minimize_automata(trad_s2)
    trad_s2.fix_split_all()
    trad_s2.draw_graph('trad_before.svg')
    minimize_automata(trad_s2, combine_equal_syms_only=True)
    trad_s2.draw_graph('trad.svg')

    #new_s1 = atm.get_single_stride_graph()
    new_s2 = atm.get_single_stride_graph()
    new_s2.draw_graph('new_before_homo.svg')
    new_s2.make_homogenous(use_espresso=True)
    new_s2.draw_graph('new_before.svg')
    minimize_automata(new_s2, combine_equal_syms_only=True, remove_dead_states=False)
    new_s2.draw_graph('new.svg')

    print "trad has {} nodes new has {} nodes".format(trad_s2.nodes_count, new_s2.nodes_count)
    print "trad has {} edges new has {} edges".format(trad_s2.edges_count, new_s2.edges_count)

    #exit(0)
