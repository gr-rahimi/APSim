import shutil, os
from itertools import count, chain
import math
from os.path import expanduser
from collections import namedtuple
import logging
from jinja2 import Environment, FileSystemLoader
import numpy as np
import networkx
from automata.automata_network import Automatanetwork
from automata.elemnts.element import FakeRoot


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
        else:
            raise NotImplementedError()

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


def get_hdl_folder_name(prefix, number_of_atms, stride_value, before_match_reg, after_match_reg, ste_type, use_bram,
                        use_compression, compression_depth):
    folder_name = prefix + 'stage_' + str(number_of_atms) + '_stride' + str(stride_value) + (
        '_before' if before_match_reg else '') + ('_after' if after_match_reg else '') + \
                   ('_ste' + str(ste_type)) + ('_withbram' if use_bram else '_nobram') + \
                  ('with_compD' if use_compression else 'no_comp') + (str(compression_depth) if use_compression else '')

    return folder_name


def generate_full_lut(atms_list, single_out ,before_match_reg, after_match_reg, ste_type,
                      use_bram, bram_criteria = None, folder_name = None, bit_feed_size=None, id_to_comp_dict=None,
                      comp_dict=None, use_compression=False):

    assert use_bram == False
    '''

    :param atms_list:
    :param single_out:
    :param before_match_reg:
    :param after_match_reg:
    :param ste_type:
    :param use_bram:
    :param bram_criteria:
    :param folder_name:
    :param bit_feed_size: total amount of bits comming in before compression(depands on the stride value)
    :param id_to_comp_dict: a list of dictionaries from compressor ids to their output len. This is the total bit counts
    :param comp_dict:
    :param use_compression:
    :return:
    '''

    env = Environment(loader=FileSystemLoader('automata/HDL/Templates'), extensions=['jinja2.ext.do'])

    ################ Generate STE


    for stage_idx, stage in enumerate(atms_list):
        if use_bram:
            raise RuntimeError('this code needs to be modified')
            template = env.get_template('bram_module.template')
            bram_list, bram_match_id_list_all = _generte_bram_stes(stage, bram_criteria, 'FF')
            for bram_idx, bram in enumerate(bram_list):
                bram_mat = _generate_bram_matrix(bram)
                bram_hex_contents = _genrate_bram_hex_content_from_matrix(bram_mat)
                rendered_content = template.render(mod_name="bram_module_"+str(bram_idx) ,stride_val=stage[0].stride_value,
                                before_match_reg=before_match_reg, after_match_reg=after_match_reg, contents=bram_hex_contents)
                with open(os.path.join(folder_name, 'bram_module_'+ str(bram_idx)+'_ste.v'), 'w') as f:
                    f.writelines(rendered_content)
        else:
            bram_list, bram_match_id_list_all = [], [[]] * len(stage)


class HDL_Gen(object):
    def __init__(self, path, before_match_reg, after_match_reg, ste_type, total_input_len, bram_shape=None):
        '''
        :param path: path to generate verilog files
        :param before_match_reg
        :param after_match_reg
        :param bram_shape: (width, col) representing the size of bram to be used
        :param
        '''

        self._path = path
        shutil.rmtree(path, ignore_errors=True)
        os.mkdir(path)

        self._clean_and_make_path()
        self._before_match_reg = before_match_reg
        self._after_match_reg = after_match_reg
        self._ste_type = ste_type
        module_abs_dir = os.path.dirname(os.path.abspath(__file__))
        self._env = Environment(loader=FileSystemLoader(os.path.join(module_abs_dir, 'Templates')),
                                extensions=['jinja2.ext.do'])
        self._total_input_len = total_input_len
        self._comp_id, self._stage_id = -1, -1 # tracking assign id for compressors
        self._atm_to_comp_id = {}  # key= atm_id, value=compressor_id
        self._comp_info = {}  # key=comp_id, value=compressor out_len
        self._atm_info = {}  # this dictionary keeps the required information in tempalates for individual automatas
        self._stage_info = {} # keeps information about automatas in the same stage. key= stage_idx, value= list of atm_ids
         # this is a 2D list for keeping autoamatas that reide in the same stage
        self._Atm_Interface = namedtuple('Atm_Interface', ['id', 'nodes', 'nodes_count', 'reports_count', 'edges_count',
                                                          'stride_value', 'use_compression', 'max_val_dim'])
        self._Node_Interface = namedtuple('Node_Interface', ['id', 'report', 'sym_count', 'symbols'])
        self._pending_automatas = [] # this list keeps all the automata that has not been assigned to a stage [(atm_interface, lut_bram_dic),....]
        self._bram_shape = bram_shape


    def _generate_single_automata(self, automata, inp_bit_len, lut_bram_dic):
        '''
        this function generates a single automata
        :param automata: input autoamata
        :param inp_bit_len: bit size fo the input. if this autamata uses compressed input, this value represents the
        total len (with stride) of tat value
        :return: None
        '''

        template = self._env.get_template('Single_Automata.template')
        template.globals['predecessors'] = networkx.MultiDiGraph.predecessors
        template.globals['get_summary'] = Automatanetwork.get_summary  # maybe better to move to utility module

        rendered_content = template.render(automata=automata,
                                           before_match_reg=self._before_match_reg,
                                           after_match_reg=self._after_match_reg,
                                           bit_feed_size=inp_bit_len,
                                           lut_bram_dic=lut_bram_dic) # TODO change the name of bit_feed_size in template. it is confusing with other bit_feed_size which is non compressed len

        with open(os.path.join(self._path, automata.id + '.v', ), 'w') as f:
            f.writelines(rendered_content)

    def register_automata(self, atm, use_compression, byte_trans_map=None, translation_list=None, lut_bram_dic={}):
        '''
        :param atm:
        :param use_compression:
        :param byte_trans_map:
        :param translation_list:
        :param lut_bram_dic: a dictionary to specify put matching in LUT or BRAM. key: node, value: a tuple of 1,2...
        1 means LUT, 2 means BRAM. if a node does not exist in this dictionary, LUT will be used for matching for all
        dimensions by default
        :return:
        '''

        assert atm.fake_root not in lut_bram_dic

        def __check_dic_valid():
            for k, v in lut_bram_dic.iteritems():
                assert len(v) == atm.stride_value
                assert k.is_fake is True or (2 not in v or k.is_symbolset_splitable())
                if 2 in v:
                    assert use_compression is False

        __check_dic_valid() # check for validation of lutbram dic

        if use_compression is False:
            self._generate_single_automata(automata=atm, inp_bit_len=self._total_input_len, lut_bram_dic=lut_bram_dic)
        else:
            single_map_len = HDL_Gen._get_sym_map_bit_len(byte_trans_map if not translation_list else translation_list[-1])
            assert single_map_len == int(math.ceil(math.log(atm.max_val_dim + 1, 2)))
            inp_bit_len = single_map_len * atm.stride_value
            self._generate_single_automata(automata=atm, inp_bit_len=inp_bit_len, lut_bram_dic=lut_bram_dic)

        assert atm.id not in self._atm_info
        atm_interface = self._Atm_Interface(id=atm.id, nodes=[], nodes_count=atm.nodes_count,
                                            reports_count=sum(1 for _ in atm.get_filtered_nodes(lambda ste: ste.report)),
                                            edges_count=atm.edges_count, stride_value=atm.stride_value,
                                            use_compression=use_compression, max_val_dim=atm.max_val_dim)
        self._atm_info[atm.id] = atm_interface
        for node in atm.nodes:
            atm_interface.nodes.append(self._Node_Interface(id=node.id, report=node.report,
                                                            sym_count=0 if node.id == FakeRoot.fake_root_id else len(node.symbols),
                                                            symbols=node.symbols if 2 in lut_bram_dic.get(node, []) else None))

        self._pending_automatas.append((atm_interface, lut_bram_dic))



    def register_compressor(self, atm_ids,byte_trans_map, translation_list):
        '''

        :param atms: list of automata ids that will have same compression unit
        :param stride_value: stride value of compressor
        :param byte_trans_map:
        :param translation_list:
        :param compression_depth:
        :return: the id of the comressor
        '''
        assert atm_ids

        byte_map_width = HDL_Gen._get_sym_map_bit_len(byte_trans_map)
        self._comp_id += 1
        for atm_id in atm_ids:
            self._atm_to_comp_id[atm_id] = self._comp_id
        width_list = [] if not translation_list else [HDL_Gen._get_sym_map_bit_len(d) for d in chain([byte_trans_map],
                                                                                                     translation_list)]

        bc_counts = self._total_input_len / 8
        initial_width = bc_counts * byte_map_width
        if not translation_list:
            output_width = initial_width
        else:
            last_reduced_comp = bc_counts
            for _ in range(len(translation_list)):
                last_reduced_comp /= 2
            output_width = last_reduced_comp * width_list[-1]

        self._generate_compressors(original_width=self._total_input_len,
                                   byte_trans_map=byte_trans_map,
                                   byte_map_width=byte_map_width,
                                   translation_list=translation_list,
                                   idx=self._comp_id,
                                   width_list=width_list,
                                   initial_width=initial_width,
                                   output_width=output_width)
        self._comp_info[self._comp_id] = output_width
        return self._comp_id

    def _clean_and_make_path(self):
        shutil.rmtree(self._path, ignore_errors=True)
        os.mkdir(self._path)

    @classmethod
    def get_bit_len(cls, codes_max):
        """
        this fucntion returns the max value that needs
        :param codes_max: the maximum value that will be feed in
        :return: the number of bits needed
        """
        return int(math.ceil(math.log(codes_max + 1, 2)))

    @classmethod
    def _get_sym_map_bit_len(cls, sym_dict):
        codes_max = max(sym_dict.values())
        return cls.get_bit_len(codes_max)

    def _generate_compressors(self, original_width, byte_trans_map, byte_map_width, translation_list, idx, width_list,
                             initial_width, output_width):
        '''

        :param original_width: the original width of compressor
        :param byte_trans_map: a dictionary for the byte level compressor
        :param byte_map_width: bit length of the byte compressor output
        :param translation_list: a list of dictionaries to convert input to output for compressor (not contains byte level)
        :param idx: the id that will be asigned to the compressor
        :param width_list: list of width of compressors
        :param initial_width: total bit size of input of compressor not (not byte level)
        :param output_width: total bit width of the output
        :param file_path: path to write the results to
        :return: None
        '''
        template = self._env.get_template('compressor_pipeline.template')
        rendered_content = template.render(original_width=original_width,
                                           byte_trans_map=byte_trans_map,
                                           byte_map_width=byte_map_width,
                                           translation_list=translation_list,
                                           idx=idx,
                                           width_list=width_list,
                                           initial_width=initial_width,
                                           output_width=output_width)

        file_path = os.path.join(self._path, 'compressor' + str(idx) + '.v')

        with open(file_path, 'w') as f:
            f.writelines(rendered_content)

    def _get_stage_summary(self, atms):
        total_nodes = 0
        total_reports = 0
        total_edges = 0
        total_sym_count = 0

        for atm in atms:
            total_nodes += atm.nodes_count
            total_reports += atm.reports_count
            total_edges += atm.edges_count
            for n in atm.nodes:
                total_sym_count += n.sym_count

        str_list = ['******************** Summary {}********************']

        str_list.append("total nodes = {}".format(total_nodes))
        str_list.append("total reports = {}".format(total_reports))
        str_list.append("total edges = {}".format(total_edges))
        str_list.append("average symbols len = {}".format(float(total_sym_count) / total_nodes))
        str_list.append('#######################################################')

        return '\n'.join(str_list)

    def _generate_bram(self, atms_list):
        '''
        this function willl take care of generating brams.
        :param atms_list: this is a list of automatas [(atm id, lut_bram_dic)...]
        :return: a tuple

        first : list of chunked brams [[(bram1content, [[(atm1,ste1.id),(atm1,ste5.id)],[(atm1, ste3.id)],....]),()], [(bram2content, [[(atm1,ste1.id),(atm1,ste5.id)],[(atm1, ste3.id)],....]),()]]
                                                         -------column1---------
                                                                                         --column2---

        second : list of dictionaries. each dictionary belongs to one dimension in stride value. Key value in dictionary is a match vector and value is a list of  tuples (atmid, node.id)
        '''



        def numpy_bram_tostring(bram_content, base_value):
            '''
            this function receives a numpy array with 0,1 and convert it to the form that xilinx bram macro likes to receive
            :param bram_content: 
            :param base_value: 
            :return:
            '''

            assert base_value == 16 or base_value == 8 or base_value == 4 or base_value == 2

            def array_to_str(inp_array):
                sum = 0
                for i in inp_array:
                    sum = sum * 2 + i

                if sum < 10:
                    return str(sum)
                elif sum == 10:
                    return 'A'
                elif sum == 11:
                    return 'B'
                elif sum == 12:
                    return 'C'
                elif sum == 13:
                    return 'D'
                elif sum == 14:
                    return 'E'
                elif sum == 15:
                    return 'F'

            per_row_bits = 256
            bits_count = int(math.log(base_value, 2))
            result_list = []
            flatten_arr = bram_content.flatten()

            for row in np.split(flatten_arr, flatten_arr.size / per_row_bits):
                row_content = []
                for chunk in np.split(row, row.size / bits_count):
                    to_str = array_to_str(chunk)
                    row_content.append(to_str)

                result_list.append("".join(row_content))

            return result_list

        all_len = [len(bram_lut_dic) == 0 for _, bram_lut_dic in atms_list]
        if all(all_len):
            return []

        all_type = []
        for _, bram_lut_dic in atms_list:
            for t in bram_lut_dic.itervalues():
                all_type.append(2 not in t)

        if all(all_type):
            return []

        stride_vals = [len(v) for _, bram_lut_dic in atms_list for v in bram_lut_dic.itervalues()]

        assert len(set(stride_vals)) == 1 # all should have same stride value

        stride_val = atms_list[0][0].stride_value

        assert stride_val == stride_vals[0]
        assert self._bram_shape
        assert self._total_input_len / stride_val <= math.log(self._bram_shape[0], 2)

        brams_list = [{} for _ in range(stride_val)]

        for atm, bram_lut_dic in atms_list:
            for node, bram_lut_d in bram_lut_dic.iteritems():
                for d_idx, d in enumerate(bram_lut_d):
                    if d == 2:
                        match_vector = node.symbols.points_on_dim(d_idx, self._bram_shape[0])
                        brams_list[d_idx].setdefault(match_vector, []).append((atm, node.id))

        chunked_bram_list = [[] for _ in range(stride_val)]

        for d in range(stride_val):
            bram_list = brams_list[d]
            current_bram_content, current_bram_nodes = None, None
            for v_idx, (mv, atmid_nodeid_tuple) in enumerate(bram_list.iteritems()):
                if v_idx % self._bram_shape[1] == 0:
                    if current_bram_content is not None:
                        chunked_bram_list[d].append((numpy_bram_tostring(current_bram_content, base_value=16), current_bram_nodes))
                    current_bram_content = np.zeros(self._bram_shape, dtype=np.int8)
                    current_bram_nodes = []

                current_bram_content[:, v_idx % self._bram_shape[1]] = mv
                current_bram_nodes.append(atmid_nodeid_tuple)

            if current_bram_nodes:  # remaining nodes
                current_bram_nodes.reverse()
                chunked_bram_list[d].append((numpy_bram_tostring(current_bram_content[:, ::-1], base_value=16), current_bram_nodes))

        return chunked_bram_list, brams_list

    def register_stage_pending(self, single_out, use_bram):
        '''
        this function put all the pending autoamtas to one stage and register that stage
        :param single_out:
        :param use_bram: if True actual bram will be used otherwise LUT will be used to emulate bram
        :return:
        '''
        if not self._pending_automatas:
            return

        brams_contents, brams_list = self._generate_bram(self._pending_automatas)
        self._register_stage(atms_id=[atm.id for atm, _ in self._pending_automatas],
                             single_out=single_out,
                             brams_contents=brams_contents, brams_list=brams_list, use_bram=use_bram)

        # for atm, _ in self._pending_automatas:
        #     for node in atm.nodes:
        #         node.symbols = None
        self._pending_automatas = []


    def _register_stage(self, atms_id, single_out, brams_contents, brams_list, use_bram):
        '''

        :param atms_id: list of ids of automatas in the same stage
        :param single_out: if true, the stage will have one report out put which is the or of all of the outputs
        :param brams_content
        :param brams_list:
        :param pending_automatas
        :return: None
        '''
        self._stage_id += 1
        self._stage_info[self._stage_id] = list(atms_id)
        template = self._env.get_template('Automata_Stage.template')
        rendered_content = template.render(automatas=[self._atm_info[temp_atm_id] for temp_atm_id in atms_id],
                                           summary_str=self._get_stage_summary(self._atm_info.values()),
                                           single_out=single_out,
                                           stage_index=self._stage_id,
                                           bit_feed_size=self._total_input_len,
                                           id_to_comp_dict={self._atm_to_comp_id[atm_id]:self._comp_info[self._atm_to_comp_id[atm_id]] for atm_id in atms_id if atm_id in self._atm_to_comp_id},
                                           comp_dict={atm_id:self._atm_to_comp_id[atm_id] for atm_id in atms_id if atm_id in self._atm_to_comp_id},
                                           brams_contents=brams_contents, brams_list=brams_list, use_bram=use_bram,
                                           pending_atms=self._pending_automatas,
                                           after_match=self._after_match_reg)
        with open(os.path.join(self._path, 'stage{}.v'.format(self._stage_id)), 'w') as f:
            f.writelines(rendered_content)


    def finilize(self):

        atms_list = [[self._atm_info[atm_id] for atm_id in atms_id]for atms_id in self._stage_info.values()]

        template = self._env.get_template('Top_Module.template')
        rendered_content = template.render(automatas=atms_list, bit_feed_size=self._total_input_len)
        with open(os.path.join(self._path, 'top_module.v'), 'w') as f:
            f.writelines(rendered_content)

        template = self._env.get_template('Single_STE.template')
        rendered_content = template.render(ste_type=self._ste_type)
        with open(os.path.join(self._path, 'ste.v'), 'w') as f:
            f.writelines(rendered_content)

        # TCL script
        template = self._env.get_template('tcl.template')
        rendered_content = template.render()
        with open(os.path.join(self._path, 'my_script.tcl'), 'w') as f:
            f.writelines(rendered_content.encode('utf-8'))

        # Timing constrains
        template = self._env.get_template('clk_constrain.template')
        rendered_content = template.render()
        with open(os.path.join(self._path, 'clk_constrain.xdc'), 'w') as f:
            f.writelines(rendered_content)
