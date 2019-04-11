import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata, get_equivalent_symbols
import automata.HDL.hdl_generator as hd_gen
import math
import random
from automata.elemnts.ste import PackedInput

#logging.basicConfig(level=logging.DEBUG)

def reza_test(inp_dic, atm):
    eq_set = set()

    cands = [16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 30]


    for key, val in inp_dic.iteritems():
        if val in cands:
            eq_set.add(key)


    kh= None
    for e in eq_set:
        packed_pt  = PackedInput(e)
        result = [node.symbols.can_accept(packed_pt) for node in atm.nodes if node.id!=0]
        if kh == None:
            kh = result
        else:
            assert kh == result

random.seed=3

#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering
# AnmalZoo.Dotstar, AnmalZoo.EntityResolution, AnmalZoo.Fermi,
#                       AnmalZoo.Hamming, AnmalZoo.Levenshtein, AnmalZoo.PowerEN, AnmalZoo.Protomata,
#                       AnmalZoo.RandomForest, AnmalZoo.Snort, AnmalZoo.SPM, AnmalZoo.Synthetic_BlockRings,
#                       AnmalZoo.Dotstar03, AnmalZoo.Dotstar06, AnmalZoo.Dotstar09, AnmalZoo.Ranges05, AnmalZoo.Ranges1,
#                       AnmalZoo.ExactMath, AnmalZoo.Bro217, AnmalZoo.TCP,

under_process_atms = [AnmalZoo.ClamAV, AnmalZoo.Synthetic_CoreRings, AnmalZoo.Brill]
exempts = {(AnmalZoo.Snort, 1411)}
number_of_autoamtas = 200
automata_per_stage = 50
use_compression = False
single_out=False
before_match_reg=False
after_match_reg=False
ste_type=1
use_bram=False
compression_depth = 0


for uat in under_process_atms:
    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    if len(automatas) > number_of_autoamtas:
        #automatas = random.sample(automatas, number_of_autoamtas)
        automatas = automatas[:number_of_autoamtas]

    number_of_stages = math.ceil(len(automatas) / float(automata_per_stage))
    atms_per_stage = int(math.ceil(len(automatas) / float(number_of_stages)))

    for stride_val in range(3, 4):

        hdl_apth = hd_gen.get_hdl_folder_path(prefix=str(uat), number_of_atms=len(automatas), stride_value=stride_val,
                                              before_match_reg=before_match_reg, after_match_reg=after_match_reg,
                                              ste_type=ste_type, use_bram=use_bram, use_compression=use_compression,
                                              compression_depth=compression_depth)

        generator_ins = hd_gen.HDL_Gen(path=hdl_apth, before_match_reg=before_match_reg,
                                       after_match_reg=after_match_reg, ste_type=ste_type,
                                       total_input_len=hd_gen.HDL_Gen.get_bit_len(automatas[0].max_val_dim) *
                                                       pow(2, stride_val))

        for atm_idx, atm in enumerate(automatas):
            if (uat, atm_idx) in exempts:
                continue

            print 'processing {0} stride{3} automata {1} from {2}'.format(uat, atm_idx, len(automatas), stride_val)

            bc_bits_len = 8
            if use_compression:
                bc_sym_dict = get_equivalent_symbols([atm], replace=True)
                bc_bits_len = int(math.ceil(math.log(max(bc_sym_dict.values()), 2)))

            translation_list = []

            for s in range(stride_val):
                atm = atm.get_single_stride_graph()
                if use_compression and s < compression_depth:
                    new_translation = get_equivalent_symbols([atm], replace=True)
                    translation_list.append(new_translation)

            if atm.is_homogeneous is False:
                atm.make_homogenous()

            minimize_automata(atm)

            #lut_bram_dic = {n: (1, 2) for n in atm.nodes}
            generator_ins.register_automata(atm=atm, use_compression=use_compression)

            if use_compression:
                generator_ins.register_compressor([atm.id], byte_trans_map=bc_sym_dict,
                                                  translation_list=translation_list)

            if (atm_idx + 1) % atms_per_stage == 0:
                generator_ins.register_stage_pending(single_out=False)

        generator_ins.finilize()