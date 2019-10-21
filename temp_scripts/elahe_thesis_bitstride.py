import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata, get_equivalent_symbols
import automata.HDL.hdl_generator as hd_gen
import math
import random
from automata.elemnts.ste import PackedInput

two_bit = False
three_bit = False
four_bit = False
seven_bit = False
eight_bit = True
osix_bit = False

uat = AnmalZoo.EntityResolution
automatas = atma.parse_anml_file(anml_path[uat])
automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()
print "start"
atm = automatas[2]
print atm.get_summary(logo="original")
#atm.remove_all_start_nodes()
#atm.draw_graph("atmnonminimized.svg")
#minimize_automata(atm)
#atm.draw_graph("atmnonmin.svg")
one_bit_atm = atma.automata_network.get_bit_automaton(atm=atm, original_bit_width=8)
#one_bit_atm.make_homogenous()
#minimize_automata(one_bit_atm)

print "finished bit level"
#one_bit_atm.draw_graph("one_bit.svg")

if two_bit:
    two_bit_atm = atma.automata_network.get_strided_automata2(atm=one_bit_atm, stride_value=2, is_scalar=True,
                                                              base_value=2, add_residual=False)
    two_bit_atm.make_homogenous()
    minimize_automata(two_bit_atm)
    #two_bit_atm.draw_graph("two_bit.svg")

if three_bit:
    three_bit_atm = atma.automata_network.get_strided_automata2(atm=one_bit_atm, stride_value=3, is_scalar=True,
                                                              base_value=2, add_residual=False)
    three_bit_atm.make_homogenous()
    minimize_automata(three_bit_atm)
    #three_bit_atm.draw_graph("three_bit.svg")

if four_bit:
    four_bit_atm = atma.automata_network.get_strided_automata2(atm=one_bit_atm, stride_value=4, is_scalar=True,
                                                              base_value=2, add_residual=False)
    four_bit_atm.make_homogenous()
    minimize_automata(four_bit_atm)
    #four_bit_atm.draw_graph("four_bit.svg")

if seven_bit:
    seven_bit_atm = atma.automata_network.get_strided_automata2(atm=one_bit_atm, stride_value=7, is_scalar=True,
                                                              base_value=2, add_residual=False)
    seven_bit_atm.make_homogenous()
    minimize_automata(seven_bit_atm)

    #seven_bit_atm.draw_graph("seven_bit.svg")

if eight_bit:
    eight_bit_atm = atma.automata_network.get_strided_automata2(atm=one_bit_atm, stride_value=8, is_scalar=True,
                                                              base_value=2, add_residual=False)
    eight_bit_atm.make_homogenous()
    minimize_automata(eight_bit_atm)
    print atm.get_summary(logo="by2bit")

    #eight_bit_atm.draw_graph("eight_bit.svg")

if osix_bit:
    osix_bit_atm = atma.automata_network.get_strided_automata2(atm=one_bit_atm, stride_value=16, is_scalar=True,
                                                              base_value=2, add_residual=False)
    osix_bit_atm.make_homogenous()
    minimize_automata(osix_bit_atm)

    #osix_bit_atm.draw_graph("osix_bit.svg")
