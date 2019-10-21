import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata, get_equivalent_symbols
import automata.HDL.hdl_generator as hd_gen
import math
import random
from automata.elemnts.ste import PackedInput





random.seed=3

#Snort, EntityResolution, ClamAV, Hamming, Dotstart, Custom, Bro217, Levenstein, Bril,
# Randomfor, Dotstar03, ExactMath,Dotstar06, Fermi, PowerEN, Protomata, Dotstart09, Ranges1, SPM, Ranges 05
#SynthBring, Synthcorering
under_process_atms = [AnmalZoo.Hamming, AnmalZoo.Protomata, AnmalZoo.EntityResolution, AnmalZoo.Brill, AnmalZoo.SPM]
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
    #automatas=pickle.load(open('Snort1-10.pkl', 'rb'))

    if len(automatas) > number_of_autoamtas:
        #automatas = random.sample(automatas, number_of_autoamtas)
        automatas = automatas[:number_of_autoamtas]


    number_of_stages = math.ceil(len(automatas) / float(automata_per_stage))
    atms_per_stage = int(math.ceil(len(automatas) / float(number_of_stages)))

    for bit_stride_val in [1, 2, 4, 8, 12, 16]:

        hdl_apth = hd_gen.get_hdl_folder_path(prefix=str(uat), number_of_atms=len(automatas), stride_value=bit_stride_val,
                                              before_match_reg=before_match_reg, after_match_reg=after_match_reg,
                                              ste_type=ste_type, use_bram=use_bram, use_compression=use_compression,
                                              compression_depth=compression_depth)

        generator_ins = hd_gen.HDL_Gen(path=hdl_apth, before_match_reg=before_match_reg,
                                       after_match_reg=after_match_reg, ste_type=ste_type,
                                       total_input_len=bit_stride_val)

        strided_automatas, bit_size,  = [], []
        for atm_idx, atm in enumerate(automatas):
            if (uat, atm_idx) in exempts:
                continue

            atm = atma.automata_network.get_bit_automaton(atm=atm, original_bit_width=hd_gen.HDL_Gen.get_bit_len(atm.max_val_dim))
            atm = atma.automata_network.get_strided_automata2(atm=atm, stride_value=bit_stride_val, is_scalar=True,
                                                              base_value=2)

            print 'processing {0} stride{3} automata {1} from {2}'.format(uat, atm_idx, len(automatas), bit_stride_val)

            if use_compression:
                bc_sym_dict = get_equivalent_symbols([atm], replace=True)

            translation_list = []

            if atm.is_homogeneous is False:
                atm.make_homogenous()

            minimize_automata(atm)

            strided_automatas.append(atm.id)

            generator_ins.register_automata(atm=atm, use_compression=use_compression, byte_trans_map=bc_sym_dict if use_compression else None,
                                            translation_list=translation_list)
            if use_compression:
                generator_ins.register_compressor([atm.id], byte_trans_map=bc_sym_dict,
                                                  translation_list=translation_list)

            if (atm_idx + 1) % atms_per_stage == 0:
                generator_ins.register_stage_pending(single_out=single_out)

        generator_ins.register_stage_pending(single_out=single_out)
        generator_ins.finilize()
