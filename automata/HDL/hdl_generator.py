from jinja2 import Environment, FileSystemLoader
import networkx
import shutil, os
from automata.automata_network import Automatanetwork
from automata.elemnts.element import FakeRoot
import numpy as np
from itertools import count



def _get_stage_summary(atms):
    total_nodes = 0
    total_reports = 0
    total_edges = 0
    total_sym_count = 0

    for atm in atms:
        total_nodes += atm.nodes_count
        total_reports += sum (1 for r in atm.get_filtered_nodes(lambda ste: ste.report))
        total_edges += atm.get_number_of_edges()
        for n in atm.nodes:
            if n.id != FakeRoot.fake_root_id:
                total_sym_count += len(n.symbols)


    str_list = ['******************** Summary {}********************']

    str_list.append("total nodes = {}".format(total_nodes))
    str_list.append("total reports = {}".format(total_reports))
    str_list.append("total edges = {}".format(total_edges))
    str_list.append("average symbols len = {}".format(float(total_sym_count) / total_nodes))
    str_list.append('#######################################################')

    return '\n'.join(str_list)


def _genrate_bram_hex_content_from_matrix(bram_matrix):
    '''
    this functionreceive a bram matrxi with dimension (stride_val, 512, 72) and generate the brm content string hex value
    :param bram_matrix: numpy matrix with 1,0
    :return: a list of contents with size(stride_val, 144, 64) hex values
    '''

    def int_to_hexstr(val):
        if d_idx % 4 == 3:
            if hex_val < 10:
                return str(hex_val)
            elif hex_val == 10:
                return 'A'
            elif hex_val == 11:
                return 'B'
            elif hex_val == 12:
                return 'C'
            elif hex_val == 13:
                return 'D'
            elif hex_val == 14:
                return 'E'
            elif hex_val == 15:
                return 'F'

    hex_content=[]

    stride_val, brm_rows, bram_cols = bram_matrix.shape

    for s in range(stride_val):
        hes_content_stride = []
        data_part = bram_matrix[s,:512,:64].flat
        hex_val = 0
        hex_str =""
        for d_idx, d in enumerate(data_part):
            hex_val = hex_val * 2 + d

            if d_idx % 4 == 3:
                hex_str += int_to_hexstr(hex_val)
                hex_val = 0

            if d_idx % 256 == 255:
                hes_content_stride.append(hex_str)
                hex_str = ""

        hex_content.append(hes_content_stride)

        data_part = bram_matrix[s, :512, 64:].flat
        hex_val = 0
        hex_str = ""
        for d_idx, d in enumerate(data_part):
            hex_val = hex_val * 2 + d

            if d_idx % 4 == 3:
                hex_str += int_to_hexstr(hex_val)
                hex_val = 0

            if d_idx % 256 == 255:
                hes_content_stride.append(hex_str)
                hex_str = ""

        hex_content.append(hes_content_stride)

    return hex_content


def _generate_bram_matrix(bram_nodes):
    '''
    this function return bram content as strings of contents 144 * 256
    :param bram_nodes: list of (automata_id, nodes) that are going to be fit in the bram
    :return: a numpy array with dimension (stride_val, 512, 72)
    '''
    stride_val = bram_nodes[0][1].symbols.dim
    bram_content = np.zeros((stride_val, 512, 72), dtype=np.int8)
    for node_idx, (_, node) in enumerate(bram_nodes):
        for sym in node.symbols:
            left = sym.left
            right = sym.right

            for d in range(stride_val):
                bram_content[d, left[d]:right[d], node_idx] = 1

    return bram_content


def _generte_bram_stes(atms, classifier_func, placement_policy):
    '''
    this function generated the bram modules for strided autoamatas.
    :param atms: list of autoamtas
    :param classifier_func: a function with True/False output which decides if an in input node should use bram if
    it returns True
    :param placement_policy: "FF: place nodes in a continues manner"
    :return: a list of lists. In the nested list, we keep a tuple(atm id, and )
    '''
    bram_list = []
    bram_match_id_list_all = []
    for atm in atms:
        bram_nodes = []
        bram_match_id_list = []
        for node in atm.nodes:
            if node.id == FakeRoot.fake_root_id:
                continue

            if classifier_func(node): # this node matching should be placed in bram
                bram_nodes.append(node)
                bram_match_id_list.append(node.id)
        bram_match_id_list_all.append(bram_match_id_list)
        bram_len = len(bram_nodes)

        if placement_policy == 'FF':
            last_residual = 72 - len(bram_list[-1]) if bram_list else 0

            for node in bram_nodes[:last_residual]:
                bram_list[-1].append((atm, node))

            for start_index in range(last_residual, bram_len, 72):
                new_bram = []
                for node in bram_nodes[start_index: start_index + 72]:
                    new_bram.append((atm, node))

                bram_list.append(new_bram)

    return bram_list, bram_match_id_list_all

def test_compressor(original_width, byte_trans_map, byte_map_width, translation_list, idx, width_list, initial_width, output_width):
    env = Environment(loader=FileSystemLoader('automata/HDL/Templates'), extensions=['jinja2.ext.do'])
    template = env.get_template('compressor_pipeline.template')
    rendered_content = template.render(original_width=original_width,
                                       byte_trans_map=byte_trans_map,
                                       byte_map_width=byte_map_width,
                                       translation_list=translation_list,
                                       idx=idx,
                                       width_list=width_list,
                                       initial_width=initial_width,
                                       output_width=output_width)
    with open('test_compressor.v', 'w') as f:
        f.writelines(rendered_content)

def generate_compressors(original_width, byte_trans_map, byte_map_width, translation_list, idx, width_list, initial_width,
             output_width, file_path):
    '''

    :param original_width: the original width of compressor
    :param byte_trans_map: a dictionary for the byte level compressor
    :param byte_map_width: bit length of the byte compressor output
    :param translation_list: a list of dictionaries to convert input to output for compressor (not byte level)
    :param idx: the id that will be asigned to the compressor
    :param width_list: list of width of compressors
    :param initial_width: total bit size of input of compressor not (not byte level)
    :param output_width: total bit width of the output
    :param file_path: path to write the results to
    :return: None
    '''
    env = Environment(loader=FileSystemLoader('automata/HDL/Templates'), extensions=['jinja2.ext.do'])
    template = env.get_template('compressor_pipeline.template')
    rendered_content = template.render(original_width=original_width,
                                       byte_trans_map=byte_trans_map,
                                       byte_map_width=byte_map_width,
                                       translation_list=translation_list,
                                       idx=idx,
                                       width_list=width_list,
                                       initial_width=initial_width,
                                       output_width=output_width)

    with open(file_path, 'w') as f:
        f.writelines(rendered_content)



def generate_full_lut(atms_list, single_out ,before_match_reg, after_match_reg, ste_type,
                      use_bram, bram_criteria = None, folder_name = None):



    folder_name += 'stage_' + str(len(atms_list)) + '_stride' + str(atms_list[0][0].stride_value) + ('_before' if before_match_reg else '') + ('_after' if after_match_reg else '') +\
                   ('_ste' + str(ste_type)) + ('_withbram' if use_bram else '_nobram')

    env = Environment(loader=FileSystemLoader('automata/HDL/Templates'), extensions=['jinja2.ext.do'])

    total_path = os.path.join("../", folder_name)
    shutil.rmtree(total_path, ignore_errors=True)
    os.mkdir(total_path)

    for atms_idx, atms in enumerate(atms_list):
        if use_bram:
            template = env.get_template('bram_module.template')
            bram_list, bram_match_id_list_all = _generte_bram_stes(atms, bram_criteria, 'FF')
            for bram_idx, bram in enumerate(bram_list):
                bram_mat = _generate_bram_matrix(bram)
                bram_hex_contents = _genrate_bram_hex_content_from_matrix(bram_mat)
                rendered_content = template.render(mod_name="bram_module_"+str(bram_idx) ,stride_val=atms[0].stride_value,
                                before_match_reg = before_match_reg,after_match_reg= after_match_reg, contents = bram_hex_contents)
                with open(os.path.join(total_path, 'bram_module_'+ str(bram_idx)+'_ste.v'), 'w') as f:
                    f.writelines(rendered_content)
        else:
            bram_list, bram_match_id_list_all = [], [[]] * len(atms)

        template = env.get_template('Single_STE.template')
        rendered_content = template.render(ste_type=ste_type)
        with open(os.path.join(total_path, 'ste.v'), 'w') as f:
            f.writelines(rendered_content)

        template = env.get_template('Single_Automata.template')
        template.globals['predecessors'] = networkx.MultiDiGraph.predecessors
        template.globals['get_summary'] = Automatanetwork.get_summary # maybe better to move to utility module
        for automata, bram_match_id_list in zip(atms, bram_match_id_list_all):
            rendered_content = template.render(automata=automata,
                                               before_match_reg=before_match_reg, after_match_reg=after_match_reg,
                                               bram_match_id_list=bram_match_id_list)
            with open(os.path.join(total_path, automata.id+'.v',), 'w') as f:
                f.writelines(rendered_content)


        template = env.get_template('Automata_Stage.template')
        rendered_content = template.render(automatas=atms,
                                           summary_str=_get_stage_summary(atms), single_out=single_out,
                                           bram_list=bram_list, bram_match_id_list=bram_match_id_list_all,
                                           stage_index=atms_idx)
        with open(os.path.join(total_path, 'stage{}.v'.format(atms_idx)), 'w') as f:
            f.writelines(rendered_content)

    template = env.get_template('Top_Module.template')
    rendered_content = template.render(automatas = atms_list)
    with open(os.path.join(total_path, 'top_module.v'), 'w') as f:
        f.writelines(rendered_content)

    # TCL script
    template = env.get_template('tcl.template')
    rendered_content = template.render()
    with open(os.path.join(total_path, 'my_script.tcl'), 'w') as f:
        f.writelines(rendered_content.encode('utf-8'))

    # Timing constrains
    template = env.get_template('clk_constrain.template')
    rendered_content = template.render()
    with open(os.path.join(total_path, 'clk_constrain.xdc'), 'w') as f:
        f.writelines(rendered_content)

