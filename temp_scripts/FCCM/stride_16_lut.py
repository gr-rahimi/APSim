import shutil
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging
import math
from multiprocessing.dummy import Pool as ThreadPool
import fcntl
import automata.HDL.hdl_generator as hd_gen

#logging.basicConfig(level=logging.DEBUG)
stride = 2

def process_single_ds(uat):
    automata_per_stage = 50
    exempts = {(AnmalZoo.Snort, 1411)}

    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    uat_count = len(automatas)
    automatas = automatas[:uat_count]
    uat_count = len(automatas)

    number_of_stages = math.ceil(len(automatas) / float(automata_per_stage))
    atms_per_stage = int(math.ceil(len(automatas) / float(number_of_stages)))

    hdl_folder_name = hd_gen.get_hdl_folder_name(prefix=str(uat), number_of_atms=uat_count,
                                                  stride_value=stride, before_match_reg=False,
                                                  after_match_reg=False, ste_type=1, use_bram=False,
                                                  use_compression=False, compression_depth=-1,
                                                  use_mid_fifo=False, use_rst=True)

    generator_ins = hd_gen.HDL_Gen(path=os.path.join("/home/gr5yf/FCCM_2020/lut16", hdl_folder_name), before_match_reg=False,
                                           after_match_reg=False, ste_type=1,
                                           total_input_len=automatas[0].max_val_dim_bits_len * pow(2, stride),
                                           bram_shape=None)

    for atm_idx, atm in enumerate(automatas):
        if (uat, atm_idx) in exempts:
            continue
        print 'processing {0} stride {3} number {1} from {2}'.format(uat, atm_idx, uat_count, stride)

        atm = atm.get_single_stride_graph()
        atm = atm.get_single_stride_graph()
        atm.make_homogenous()

        minimize_automata(atm)

        generator_ins.register_automata(atm=atm, use_compression=False, lut_bram_dic={})

        if (atm_idx + 1) % atms_per_stage == 0:
            generator_ins.register_stage_pending(use_bram=False)

    generator_ins.register_stage_pending(use_bram=False)

    generator_ins.finilize(dataplane_intcon_max_degree=5, contplane_intcon_max_degree=10)


if __name__ == '__main__':

    process_single_ds(AnmalZoo.TCP)



