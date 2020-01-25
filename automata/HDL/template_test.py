from jinja2 import Environment, FileSystemLoader
import math
import os
import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import automata.HDL.hdl_generator as hd_gen

def test_axi_contoler():
    rep_bit_cnt = 34
    stage_index = 6
    metadata_bw = 34
    use_out_reg = False
    reduction_tree = hd_gen.HDL_Gen.get_intcon_tree(rep_bit_cnt, 6)
    reduction_pipe_count = len(reduction_tree)
    env = Environment(loader=FileSystemLoader('Templates'), extensions=['jinja2.ext.do'])
    template = env.get_template('./axi_fifo_controller.template')
    rendered_content = template.render(stage_index=stage_index,
                                       rep_bit_cnt=rep_bit_cnt,
                                       metadata_bw=metadata_bw,
                                       use_out_reg=use_out_reg,
                                       reduction_pipe_count=reduction_pipe_count,
                                       reduction_tree=reduction_tree)
    with open('/home/gr5yf/axi_ctrl.v', 'w') as f:
        f.writelines(rendered_content)

def intconn_test():
    env = Environment(loader=FileSystemLoader('Templates'), extensions=['jinja2.ext.do'])
    template = env.get_template('./report_block_design_tcl.template')
    rendered_content = template.render(report_packet_width_list=[64, 64, 64, 64],
                                       report_buffer_length=[4096, 2048, 1024, 512],
                                       intconn_info_list=[[32, 2, 256, [[0, 1], [2, 3]]],
                                                          [32, 1, 256, [[0, 1]]]],
                                       lite_intconn_info=[[2, [[0, 1], [2, 3]]],
                                                          [1, [[0, 1]]]],
                                       autoamta_clock_freq=150)

    with open('/home/gr5yf/intconn_test.tcl', 'w') as f:
        f.writelines(rendered_content)
def test_hdl1():
    out_dir_prefix = './'

    dbw = 8
    use_mid_fifo = True
    use_rst = True
    def process_single_ds(uat):

        return_result = {}
        # result_dir = out_dir_prefix + str(uat)

        # shutil.rmtree(result_dir, ignore_errors=True)
        # os.mkdir(result_dir)
        # cleaning the result folder

        automata_per_stage = 50
        # this is a pipelineing parameter for staging as pipeline. We usually use 50 for this parameter

        automatas = atma.parse_anml_file(anml_path[uat])
        automatas.remove_ors()
        automatas = automatas.get_connected_components_as_automatas()

        # uat_count = 20  # number of automata to be processed
        uat_count = len(automatas)  # comment this to test a subset of automatons defined in uat_count
        automatas = automatas[:uat_count]
        uat_count = len(automatas)

        number_of_stages = int(math.ceil(len(automatas) / float(automata_per_stage)))
        # number of pipleine stages

        atms_per_stage = int(math.ceil(len(automatas) / float(number_of_stages)))

        hdl_folder_name = hd_gen.get_hdl_folder_name(prefix=str(uat), number_of_atms=len(automatas),
                                                     stride_value=0, before_match_reg=False,
                                                     after_match_reg=False, ste_type=1, use_bram=False,
                                                     use_compression=False, compression_depth=-1,
                                                     use_mid_fifo=use_mid_fifo, use_rst=use_rst)

        print "folder name to store the HDLs:", hdl_folder_name

        generator_ins = hd_gen.HDL_Gen(path=os.path.join("/home/gr5yf/FCCM_2020", hdl_folder_name),
                                       before_match_reg=False,
                                       after_match_reg=False, ste_type=1,
                                       total_input_len=dbw, use_mid_fifo=use_mid_fifo, use_rst=use_rst)

        for atm_idx, atm in enumerate(automatas):
            print 'processing {0} number {1} from {2}'.format(uat, atm_idx + 1, uat_count)
            minimize_automata(atm)

            generator_ins.register_automata(atm=atm, use_compression=False)

            if (atm_idx + 1) % atms_per_stage == 0:
                generator_ins.register_stage_pending(use_bram=False)

        generator_ins.register_stage_pending(use_bram=False)

        generator_ins.finilize(dataplane_intcon_max_degree=5, contplane_intcon_max_degree=10)

        return uat, return_result

    for use_fifo in [True, False]:
        use_mid_fifo = use_fifo
        process_single_ds(AnmalZoo.Protomata)

    print "Results are ready in zip files in the Home directory"
    print "Please extract the zip file and synthesize them on your machine using Vivado. To run the synthesize, run the following command."
    print "vivado -mode tcl -source my_script.tcl"
    print "After the synthesize is done, report files are generated in utilization.txt and timing_summary.txt"
    print "Please note that in my_script.tcl, the target FPGA devide is xcvu9p-flgb2104-1-i. Please change it accordingly if you are targeting a different device"


if __name__ == '__main__':
    test_axi_contoler()

