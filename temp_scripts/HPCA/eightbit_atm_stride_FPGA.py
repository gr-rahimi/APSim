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


out_dir = '../../Results/HDL_test/'

def process_single_ds(uat):

    return_result = {}
    result_dir = out_dir + str(uat)

    shutil.rmtree(result_dir, ignore_errors=True)
    os.mkdir(result_dir)
    exempts = {(AnmalZoo.Snort, 1411)}

    max_target_stride = 2
    uat_count = 200
    automata_per_stage = 50

    automatas = atma.parse_anml_file(anml_path[uat])
    automatas.remove_ors()
    automatas = automatas.get_connected_components_as_automatas()

    after_match_reg = False
    actual_bram = False  # if True, actual bram will be used. Otherwise, LUT emulates bram

    automatas = automatas[:uat_count]
    uat_count = len(automatas)

    number_of_stages = math.ceil(len(automatas) / float(automata_per_stage))
    atms_per_stage = int(math.ceil(len(automatas) / float(number_of_stages)))

    for hom_between, is_Bram in [(False, True)]:

        for stride_val in range(max_target_stride + 1):

            hdl_folder_name = hd_gen.get_hdl_folder_name(prefix=str(uat), number_of_atms=len(automatas),
                                                  stride_value=stride_val, before_match_reg=False,
                                                  after_match_reg=after_match_reg, ste_type=1, use_bram=is_Bram,
                                                  use_compression=False, compression_depth=-1)

            generator_ins = hd_gen.HDL_Gen(path=os.path.join(result_dir, hdl_folder_name), before_match_reg=False,
                                           after_match_reg=False, ste_type=1,
                                           total_input_len=automatas[0].max_val_dim_bits_len * pow(2, stride_val),
                                           bram_shape=(512, 36))

            for atm_idx, atm in enumerate(automatas):
                if (uat, atm_idx) in exempts:
                    continue
                print 'processing {0} stride {3} number {1} from {2}'.format(uat, atm_idx, uat_count, stride_val)

                for _ in range(stride_val):
                    if is_Bram is True and hom_between is True and atm.is_homogeneous is False:
                        atm.make_homogenous()
                        atm.make_parentbased_homogeneous()

                    atm = atm.get_single_stride_graph()

                if atm.is_homogeneous is False:
                    atm.make_homogenous()

                minimize_automata(atm)

                if is_Bram is True and hom_between is False:
                    atm.fix_split_all()

                if is_Bram:
                    lut_bram_dic = {n: tuple((2 for _ in range(atm.stride_value))) for n in atm.nodes if
                                    n.is_fake is False}
                else:
                    lut_bram_dic = {}

                generator_ins.register_automata(atm=atm, use_compression=False, lut_bram_dic=lut_bram_dic)

                if (atm_idx + 1) % atms_per_stage == 0:
                    generator_ins.register_stage_pending(use_bram=actual_bram)


            generator_ins.finilize()

    return uat, return_result



if __name__ == '__main__':

    #os.remove(out_dir + 'summary.txt')

    ds = [AnmalZoo.Snort,
          AnmalZoo.RandomForest,
          AnmalZoo.Ranges1,
          AnmalZoo.PowerEN,
          AnmalZoo.Protomata,
          AnmalZoo.Dotstar03,
          AnmalZoo.ExactMath,
          AnmalZoo.SPM,
          AnmalZoo.Custom]


    ds = []
    ds = [a for a in AnmalZoo if a not in ds]

    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()



