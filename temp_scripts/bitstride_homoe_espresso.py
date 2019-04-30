import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata, get_equivalent_symbols
import automata.HDL.hdl_generator as hd_gen
import math
import random
from automata.elemnts.ste import PackedInput
import logging

random.seed=3
#logging.basicConfig(level=logging.DEBUG)
#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering
under_process_atms = [AnmalZoo.ExactMath]
exempts = {(AnmalZoo.Snort, 1411)}
number_of_autoamtas = 10

for uat in under_process_atms:
    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()
    #automatas=pickle.load(open('Snort1-10.pkl', 'rb'))

    if len(automatas) > number_of_autoamtas:
        #automatas = random.sample(automatas, number_of_autoamtas)
        automatas = automatas[:number_of_autoamtas]


    for bit_stride_val in [8]:

        strided_automatas, bit_size,  = [], []
        for atm_idx, atm in enumerate(automatas):
            if (uat, atm_idx) in exempts:
                continue
            old_count = atm.nodes_count

            atm = atma.automata_network.get_bit_automaton(atm=atm, original_bit_width=atm.max_val_dim.bit_length())
            #print "finished bitwise"
            bit_stride_atm = atma.automata_network.get_strided_automata2(atm=atm, stride_value=bit_stride_val, is_scalar=True,
                                                              base_value=2)

            #print 'finished {0} stride{3} automata {1} from {2}'.format(uat, atm_idx, len(automatas), bit_stride_val)



            if bit_stride_atm.is_homogeneous is False:
                bit_stride_atm.make_homogenous(use_espresso=True)

            minimize_automata(bit_stride_atm)

            print "old bit counts {} new bit count {}".format(old_count, bit_stride_atm.nodes_count)
