import shutil
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging
from multiprocessing.dummy import Pool as ThreadPool
import fcntl
import automata.HDL.hdl_generator as hd_gen
import math

#logging.basicConfig(level=logging.DEBUG)


out_dir = '../../Results/BV_HDL_test/'

def process_single_ds(uat):

    #uat = AnmalZoo.Ranges05

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


    #uat_count = len(automatas)  # comment this to test a subset of automatons defined in uat_count

    automatas = automatas[:uat_count]
    uat_count = len(automatas)

    number_of_stages = math.ceil(len(automatas) / float(automata_per_stage))
    atms_per_stage = int(math.ceil(len(automatas) / float(number_of_stages)))

    for hom_between, is_Bram in [(False, True)]:
        hdl_writers = []
        for i in range(max_target_stride + 1):
            hdl_folder_name = hd_gen.get_hdl_folder_name(prefix=str(uat), number_of_atms=len(automatas),
                                                         stride_value=i, before_match_reg=False,
                                                         after_match_reg=after_match_reg, ste_type=1, use_bram=is_Bram,
                                                         use_compression=False, compression_depth=-1)

            generator_ins = hd_gen.HDL_Gen(path=os.path.join(result_dir, hdl_folder_name), before_match_reg=False,
                                           after_match_reg=after_match_reg, ste_type=1,
                                           total_input_len=4 * pow(2, i),
                                           bram_shape=(512, 36))
            hdl_writers.append(generator_ins)


        for atm_idx, atm in enumerate(automatas):
            b_atm = atma.automata_network.get_bit_automaton(atm, original_bit_width=atm.max_val_dim_bits_len)
            atm = atma.automata_network.get_strided_automata2(atm=b_atm,
                                                              stride_value=4,
                                                              is_scalar=True,
                                                              base_value=2,
                                                              add_residual=True)

            for stride_val in reversed(range(max_target_stride + 1)):
                if (uat, atm_idx) in exempts:
                    continue
                print 'processing {0} stride {3} number {1} from {2}'.format(uat, atm_idx, uat_count, stride_val)
                s_atm = atm

                for _ in range(stride_val):
                    if s_atm is atm:
                        s_atm = atm.get_single_stride_graph()
                    else:
                        s_atm = s_atm.get_single_stride_graph()

                if s_atm.is_homogeneous is False:
                    s_atm.make_homogenous()

                minimize_automata(s_atm)

                if is_Bram is True and hom_between is False:
                    s_atm.fix_split_all()

                if is_Bram:
                    lut_bram_dic = {n: tuple((2 for _ in range(s_atm.stride_value))) for n in s_atm.nodes if
                                    n.is_fake is False}
                else:
                    lut_bram_dic = {}

                hdl_writers[stride_val].register_automata(atm=s_atm, use_compression=False, lut_bram_dic=lut_bram_dic)

                if (atm_idx + 1) % atms_per_stage == 0:
                    hdl_writers[stride_val].register_stage_pending(use_bram=actual_bram)

        for i in range(max_target_stride + 1):
            hdl_writers[i].finilize()

    return uat, return_result



if __name__ == '__main__':
    open(out_dir + 'summary.txt', 'w')  # creat a new file

    ds = [AnmalZoo.Snort,
          AnmalZoo.RandomForest,
          AnmalZoo.Ranges1,
          AnmalZoo.PowerEN,
          AnmalZoo.Protomata,
          AnmalZoo.Dotstar03,
          AnmalZoo.ExactMath,
          AnmalZoo.SPM,
          AnmalZoo.Custom]

    ds = [a for a in AnmalZoo if a not in ds]



    thread_count = 8

    t_pool = ThreadPool(thread_count)
    results = t_pool.map(process_single_ds, ds)
    t_pool.close()
    t_pool.join()



