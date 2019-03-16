import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv

#logging.basicConfig(level=logging.DEBUG)

#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering

uat = AnmalZoo.Hamming

automatas = atma.parse_anml_file(anml_path[uat])
automata_name = str(uat)

exempts = {(AnmalZoo.Snort, 1411)}

automatas.remove_ors()
automatas = automatas.get_connected_components_as_automatas()

print("Number of automata: ", len(automatas))
print(automata_name)

filed_names = ['#States', '#Edges', 'max_fan_in', 'max_fan_out', 'total_sym']


for atm_idx, atm in enumerate(automatas):
    print "Processing:", uat, " ", atm_idx


    atm1E = atm.get_single_stride_graph()
    atm2E = atm1E.get_single_stride_graph()
    atm2E.make_homogenous(plus_src=False)

    minimize_automata(atm2E)
    atm2E.fix_split_all()

    print "HI"
