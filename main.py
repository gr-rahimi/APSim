import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle

automata = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
print "Finished processing from anml file. Here is the summary"

automata.remove_ors()

orig_automatas = automata.get_connected_components_as_automatas()


#print atm.max_STE_in_degree()
#print atm.max_STE_out_degree()

#atm.set_max_fan_in(3)
#atm.set_max_fan_out(3)

for atm_idx, atm in enumerate(orig_automatas):
    atm.draw_switch_box("snort/" + "atm_"+str(atm_idx))
    if atm_idx == 100:
        break





















