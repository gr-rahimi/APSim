import shutil
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging

logging.basicConfig(level=logging.DEBUG)


uat = AnmalZoo.Snort

uat_index = 14

max_target_stride = 3
uat_count = 200

automatas = atma.parse_anml_file(anml_path[uat])
automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()

atm = automatas[uat_index]

atm.draw_graph("original.svg")

#eight_stride_1 = atm.get_single_stride_graph()
#eight_stride_1.make_homogenous()
#minimize_automata(eight_stride_1)
#eight_stride_1.draw_graph("eight_bit_strie1.svg")

b_atm = atma.automata_network.get_bit_automaton(atm, original_bit_width=atm.max_val_dim_bits_len)
fourbit_atm = atma.automata_network.get_strided_automata2(atm=b_atm,
                                                          stride_value=4, is_scalar=True, base_value=2,
                                                          add_residual=True)
four_bit_s1 = fourbit_atm.get_single_stride_graph()
four_bit_s2 = four_bit_s1.get_single_stride_graph()

four_bit_s2.make_homogenous()
minimize_automata(four_bit_s2)

print "four bit=", four_bit_s2.nodes_count
four_bit_s2.draw_graph("fourbit_s2.svg")
