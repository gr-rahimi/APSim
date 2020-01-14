import subprocess
import re
import logging
from itertools import product
import os
import threading
from jinja2 import Environment, FileSystemLoader
from automata.elemnts.ste import PackedIntervalSet, PackedInterval, PackedInput



class Espresso(object):

    _module_abs_dir = os.path.dirname(os.path.abspath(__file__))
    _env = Environment(loader=FileSystemLoader(os.path.join(_module_abs_dir, 'Templates')), extensions=['jinja2.ext.do'])
    _split_sym_template = _env.get_template('espresso_symbolsplit.template')
    _hom_template = _env.get_template('espresso_homogeneous.template')
    _p_pattern = re.compile(r'\s\.p (\d+)')
    _to_one_transition_pattern = re.compile(r'01')
    _to_zero_transition_pattern = re.compile(r'10')

    @classmethod
    def get_splitted_sym_sets(cls, symset, max_val):
        '''
        :param symset: the symbol set that we want to split
        :param max_val: the maximum value that the automata accept
        :return:
        '''

        tid = threading.current_thread().ident
        template_rendered = cls._split_sym_template.render(symset=symset, max_val=max_val)

        with open('/tmp/rezasim_espresso{}split.txt'.format(tid), 'w') as f:
            f.writelines(template_rendered)

        logging.debug("Spresso split started...")
        try:
            espresso_out = subprocess.check_output('/home/gr5yf/Git/espresso-logic/bin/espresso /tmp/rezasim_espresso{}split.txt'
                                               .format(tid),
                                               shell=True)
        except subprocess.CalledProcessError as e:
            logging.error(e.output)
            exit(-1)

        logging.debug("Spresso Split Done!")

        p_matches = cls._p_pattern.findall(espresso_out)

        if len(p_matches) is not 1:
            logging.fatal('could not find P\n{}'.format(espresso_out))
            exit(1)

        new_symbols_count = int(p_matches[0])
        new_symbols_list = []

        symbols_pattern = re.compile('[01]{{{max_val}}} '.format(max_val=max_val+1) * symset.dim + '01')

        symset_matches = symbols_pattern.findall(espresso_out)

        for sym_text in symset_matches:
            new_symbol_set, _ = cls._espresso_to_symset(sym_text)
            new_symbols_list.append(new_symbol_set)

        return new_symbols_count, new_symbols_list


    @classmethod
    def make_ste_homogeneous(cls, stride_value, max_val_dim, inp_function):
        """
        this function calls the espresso template for making homogeneous and parse the results
        :param stride_value
        :param max_val_dim
        :param inp_function a dictionary with key = ("0101001", "1100001") , value = "0001011"
        :return: ([(PackedIntervalSet, String)...])a list of tuples. Each tuple is actually a
        """

        if not inp_function:
            return {}

        tid = threading.current_thread().ident

        output_functions_count = 0
        for v in inp_function.itervalues():
            output_functions_count = len(v)
            break
        template_rendered = cls._hom_template.render(stride_value=stride_value, bits_per_dim=max_val_dim.bit_length(),
                                                     output_functions_count=output_functions_count,
                                                     input_function=inp_function)

        with open('/tmp/rezasim_espresso{}homo.txt'.format(tid), 'w') as f:
            f.writelines(template_rendered)

        logging.debug("Spresso homo started...")
        try:
            espresso_out = subprocess.check_output('/zf15/gr5yf/Git/espresso-logic/bin/espresso /tmp/rezasim_espresso{}homo.txt'
                                               .format(tid),
                                               shell=True)
        except subprocess.CalledProcessError as e:
            logging.error(e.output)
            exit(-1)
        logging.debug("Spresso Homo Done!")

        p_matches = cls._p_pattern.findall(espresso_out)

        if len(p_matches) is not 1:
            logging.fatal('could not find P\n{}'.format(espresso_out))
            exit(1)

        symbols_pattern = re.compile('[01]{{{max_val}}} '.format(max_val=max_val_dim + 1) * stride_value +
                                     '[01]{{{out_func_cnt}}}'.format(out_func_cnt=output_functions_count))

        symset_matches = symbols_pattern.findall(espresso_out)
        new_symset_list = []
        for sym_text in symset_matches:
            new_symbol_set, out_func = cls._espresso_to_symset(sym_text)
            new_symset_list.append((new_symbol_set, out_func))

        return new_symset_list

    @classmethod
    def _espresso_to_symset(cls, sym_text):
        '''
        this funmction receives a single line of the espresso output and returns a single symbolset of that
        :param sym_text: (string) a single line of the espresso output .p
        :return: (PackedIntervalSet, String) string represents the function output
        '''
        assert sym_text, "fix this"

        dims_solutions = sym_text.split()
        dims_solutions, out_str = dims_solutions[:-1], dims_solutions[-1]
        max_val_per_dim = len(dims_solutions[0]) - 1
        all_dims_result = []

        for d in range(len(dims_solutions)):
            dim_solution = dims_solutions[d]

            start_indexs = [0] if dim_solution[0] == '1' else []
            for zo in cls._to_one_transition_pattern.finditer(dim_solution):
                start_indexs.append(zo.start() + 1)

            end_indexes = []
            for oz in cls._to_zero_transition_pattern.finditer(dim_solution):
                end_indexes.append(oz.start())

            if dim_solution[-1] == '1':
                end_indexes.append(max_val_per_dim)

            assert len(start_indexs) == len(end_indexes)

            all_dims_result.append([(s, e) for s, e in zip(start_indexs, end_indexes)])

        new_symbol_set = PackedIntervalSet([])

        for p in product(*all_dims_result):
            left = tuple(s for s, _ in p)
            right = tuple(e for _, e in p)

            left_pt = PackedInput(left)
            right_pt = PackedInput(right)
            pi = PackedInterval(left_pt, right_pt)
            new_symbol_set.add_interval(pi)
        return new_symbol_set, out_str

