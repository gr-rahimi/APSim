import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata

#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering
under_process_atms = [AnmalZoo.Hamming]
exempts = {(AnmalZoo.Snort, 1411)}
plus_src = False

for uat in under_process_atms:

    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    for stride_val in range(3,4):
        for atm_idx, atm in enumerate(automatas[10:]):

            if (uat, atm_idx) in exempts:
                continue
            atm.remove_ors()

            print atm.get_summary()

            for _ in range(stride_val):
                atm = atm.get_single_stride_graph()

            if atm.is_homogeneous is False:
                atm.make_homogenous()
            minimize_automata(atm)

            atm.fix_split_all()

            print atm.get_summary()






