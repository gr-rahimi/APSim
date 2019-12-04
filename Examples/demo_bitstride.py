import csv
import logging
import os
import shutil
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata

uat = AnmalZoo.Fermi
# Selected benchmark

uat_index = 1
# the index of automaton in the benchmark that will be processed
# change this value to process another automnta in the uat benchmark

dbw = 4
# the destetnation bitwidth

automatas = atma.parse_anml_file(anml_path[uat])
# parse the anml file

automatas.remove_ors()
# remove any OR gate if it exist in the automaton

automatas = automatas.get_connected_components_as_automatas()
# break the automtaon to its components

atm = automatas[uat_index]
# select the under test automaton

print atm.get_summary(logo=" of the orignal automata")
# print information about the automata

atm.draw_graph("original.svg")
# draw the autoamton to an svg file


b_atm = atma.automata_network.get_bit_automaton(atm, original_bit_width=atm.max_val_dim_bits_len)
#binary automata

print b_atm.get_summary(logo=" of bitwise automata")
# print information about the binary automata

new_bw_atm = atma.automata_network.get_strided_automata2(atm=b_atm,
                                                          stride_value=dbw, is_scalar=True, base_value=2,
                                                          add_residual=True)
# new automata that has dbw bits per symbol

new_bw_atm.make_homogenous()
# make the autoamta homogeneous

minimize_automata(new_bw_atm)
# minimizing the new autoamata

print new_bw_atm.get_summary(logo=" of %d-bit automata"%(dbw,))
# print summary of the new autoamta

new_bw_atm.draw_graph("new_bw_atm.svg")

print "SVG files were generated."



