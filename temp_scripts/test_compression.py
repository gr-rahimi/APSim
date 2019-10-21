import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata, get_equivalent_symbols
import automata.HDL.hdl_generator as hd_gen
import math
import random
from automata.elemnts.ste import PackedInput

#logging.basicConfig(level=logging.DEBUG)

under_process_atms = [AnmalZoo.Hamming]
exempts = {(AnmalZoo.Snort, 1411)}
number_of_autoamtas = 5
automata_per_stage = 5
use_compression = True
single_out = False
before_match_reg = False
after_match_reg = False
ste_type = 1
use_bram = False
compression_depth = 0

for uat in under_process_atms:
    all_automata = atma.parse_anml_file(anml_path[uat])
    all_automata.remove_ors()
    automatas = all_automata.get_connected_components_as_automatas()

    if len(automatas) > number_of_autoamtas:
        #automatas = random.sample(automatas, number_of_autoamtas)
        automatas = automatas[:number_of_autoamtas]

    number_of_stages = math.ceil(len(automatas) / float(automata_per_stage))
    atms_per_stage = int(math.ceil(len(automatas) / float(number_of_stages)))

    for stride_val in range(1):

        hdl_apth = hd_gen.get_hdl_folder_path(prefix="comptestrandom" + str(uat), number_of_atms=len(automatas), stride_value=stride_val,
                                              before_match_reg=before_match_reg, after_match_reg=after_match_reg,
                                              ste_type=ste_type, use_bram=use_bram, use_compression=use_compression,
                                              compression_depth=compression_depth)

        generator_ins = hd_gen.HDL_Gen(path=hdl_apth, before_match_reg=before_match_reg,
                                       after_match_reg=after_match_reg, ste_type=ste_type,
                                       total_input_len=hd_gen.HDL_Gen.get_bit_len(all_automata.max_val_dim) *
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
            generator_ins.register_automata(atm=atm, use_compression=use_compression,
                                            byte_trans_map=bc_sym_dict if use_compression else None)

            if use_compression:
                generator_ins.register_compressor([atm.id], byte_trans_map=bc_sym_dict,
                                                  translation_list=translation_list)

            if (atm_idx + 1) % atms_per_stage == 0:
                generator_ins.register_stage_pending(single_out=False)

        generator_ins.finilize()