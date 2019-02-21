import subprocess
import re
import logging
from itertools import product
import os
from jinja2 import Environment, FileSystemLoader
from automata.elemnts.ste import PackedIntervalSet, PackedInterval, PackedInput

def get_splitted_sym_sets(symset, max_val):
    '''

    :param symset: the symbol set that we want to split
    :param max_val: the maximum value that the automata accept
    :return:
    '''
    env = Environment(loader=FileSystemLoader('automata/Espresso/Templates'), extensions=['jinja2.ext.do'])
    template = env.get_template('espresso.template')
    template_rendered = template.render(symset=symset, max_val=max_val)

    with open('/tmp/rezasim_espresso{}.txt'.format(os.getpid()), 'w') as f:
        f.writelines(template_rendered)

    espresso_out = subprocess.check_output('/zf15/gr5yf/Git/espresso-logic/bin/espresso /tmp/rezasim_espresso{}.txt'.format(os.getpid()),
                                           shell=True)

    p_pattern = re.compile(r'\s\.p (\d+)')
    p_matches = p_pattern.findall(espresso_out)

    if len(p_matches) is not 1:
        logging.fatal('could not find P\n{}'.format(espresso_out))
        exit(1)

    new_symbols_count = int(p_matches[0])
    new_symbols_list = []

    symbols_pattern = re.compile('[01]{{{max_val}}} '.format(max_val=max_val+1) * symset.dim + '01')

    to_one_transition_pattern = re.compile(r'01')
    to_zero_transition_pattern = re.compile(r'10')

    symset_matches = symbols_pattern.findall(espresso_out)

    for symset_match in symset_matches:
        dims_soluions = symset_match.split()
        all_dims_result=[]
        for d in range(symset.dim):
            dim_solution = dims_soluions[d]

            start_indexs = [0] if dim_solution[0] == '1' else []
            for zo in to_one_transition_pattern.finditer(dim_solution):
                start_indexs.append(zo.start() + 1)

            end_indexes = []
            for oz in to_zero_transition_pattern.finditer(dim_solution):
                end_indexes.append(oz.start())

            if dim_solution[-1] == '1':
                end_indexes.append(max_val)

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

        new_symbols_list.append(new_symbol_set)

    return new_symbols_count, new_symbols_list























