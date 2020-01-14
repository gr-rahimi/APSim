from jinja2 import Environment, FileSystemLoader
import automata.HDL.hdl_generator as hd_gen
intcon_tree = hd_gen.HDL_Gen._get_intcon_tree(101, 10)
print intcon_tree
exit(0)

env = Environment(loader=FileSystemLoader('Templates'), extensions=['jinja2.ext.do'])
template = env.get_template('./report_block_design_tcl.template')
rendered_content = template.render(report_packet_width_list=[64, 64, 64, 64],
                                   report_buffer_length=[4096, 2048, 1024, 512],
                                   intconn_info_list=[[32, 2, 256, [[0, 1], [2, 3]]],
                                                      [32, 1, 256, [[0, 1]]]],
                                   lite_intconn_info=[[2, [[0, 1], [2, 3]]],
                                                       [1, [[0, 1]]]],
                                   autoamta_clock_freq=150)

with open('/home/gr5yf/intconn_test.tcl','w') as f:
    f.writelines(rendered_content)