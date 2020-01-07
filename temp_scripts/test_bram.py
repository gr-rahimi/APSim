import logging
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata, get_equivalent_symbols
import automata.HDL.hdl_generator as hd_gen
import math
import random
from automata.elemnts.ste import PackedInput

logging.basicConfig(level=logging.DEBUG)


random.seed=3

under_process_atms = [
                      AnmalZoo.TCP, AnmalZoo.Dotstar, AnmalZoo.Dotstar03, AnmalZoo.Dotstar06,AnmalZoo.Dotstar09,
                      AnmalZoo.EntityResolution, AnmalZoo.Ranges05, AnmalZoo.Ranges1, AnmalZoo.Protomata, AnmalZoo.Brill,
                      AnmalZoo.Snort, AnmalZoo.SPM

                      ]
exempts = {}
number_of_autoamtas = 200
automata_per_stage = 50


single_out=False
before_match_reg=False
after_match_reg=False
ste_type = 1
use_bram = True
compression_depth = 0


for uat in under_process_atms:
    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    if len(automatas) > number_of_autoamtas:
        #automatas = random.sample(automatas, number_of_autoamtas)
        automatas = automatas[3:number_of_autoamtas]

    number_of_stages = math.ceil(len(automatas) / float(automata_per_stage))
    atms_per_stage = int(math.ceil(len(automatas) / float(number_of_stages)))

    for stride_val in range(3, 4):

        hdl_apth = hd_gen.get_hdl_folder_path(prefix="bramtest" + str(uat), number_of_atms=len(automatas),
                                              stride_value=stride_val, before_match_reg=before_match_reg,
                                              after_match_reg=after_match_reg, ste_type=ste_type, use_bram=use_bram,
                                              use_compression=False, compression_depth=-1)

        generator_ins = hd_gen.HDL_Gen(path=hdl_apth, before_match_reg=before_match_reg,
                                       after_match_reg=after_match_reg, ste_type=ste_type,
                                       total_input_len=automatas[0].max_val_dim_bits_len * pow(2, stride_val), bram_shape=(512, 36))

        for atm_idx, atm in enumerate(automatas):
            if (uat, atm_idx) in exempts:
                continue

            print 'processing {0} stride{3} automata {1} from {2}'.format(uat, atm_idx, len(automatas), stride_val)

            translation_list = []

            for s in range(stride_val):
                atm = atm.get_single_stride_graph()

            if atm.is_homogeneous is False:
                atm.make_homogenous()

            minimize_automata(atm)

            atm.fix_split_all()

            lut_bram_dic = {n: tuple((2 for _ in range(pow(2, stride_val)))) for n in atm.nodes if n.is_fake is False}
            generator_ins.register_automata(atm=atm, use_compression=False, lut_bram_dic=lut_bram_dic)


            if (atm_idx + 1) % atms_per_stage == 0:
                generator_ins.register_stage_pending()

        generator_ins.finilize()