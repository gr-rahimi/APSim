from jinja2 import Template, Environment, FileSystemLoader
from automata import Automatanetwork
from automata.elemnts.ste import S_T_E,StartType, PackedIntervalSet, PackedInterval, PackedInput
import networkx



env = Environment(loader=FileSystemLoader('Templates'), extensions=['jinja2.ext.do'])
# template = env.get_template('LUT_match.template')
#
# rendered_content = template.render(intervals=[[(1, 2), (3, 5)], [(44, 76), (78, 99)]])
#
# print rendered_content


atm = Automatanetwork(id='temp_automata', is_homogenous=True, stride=2, max_val=255)
symbol_set1 = PackedIntervalSet([PackedInterval(PackedInput((1,2)),PackedInput((3,5))),
                                 PackedInterval(PackedInput((44,76)),PackedInput((78,99)))])

ste1 = S_T_E(start_type=StartType.start_of_data, is_report=False, is_marked=False,
             id=atm.get_new_id(), symbol_set=symbol_set1, adjacent_S_T_E_s=None, report_residual=0,
             report_code=0
             )
atm.add_element(ste1)

symbol_set2 = PackedIntervalSet([PackedInterval(PackedInput((10,20)),PackedInput((30,50))),
                                 PackedInterval(PackedInput((4,76)),PackedInput((7,99)))])

ste2 = S_T_E(start_type=StartType.non_start, is_report=True, is_marked=False,
             id=atm.get_new_id(), symbol_set=symbol_set2, adjacent_S_T_E_s=None, report_residual=0,
             report_code=0
             )
atm.add_element(ste2)

symbol_set3 = PackedIntervalSet([PackedInterval(PackedInput((100,200)),PackedInput((3,50))),
                                 PackedInterval(PackedInput((44,76)),PackedInput((70,99)))])
ste3 = S_T_E(start_type=StartType.non_start, is_report=True, is_marked=False,
             id=atm.get_new_id(), symbol_set=symbol_set3, adjacent_S_T_E_s=None, report_residual=0,
             report_code=0
             )

atm.add_element(ste3)

atm.add_edge(ste1, ste2)
atm.add_edge(ste1, ste3)
atm.add_edge(ste2, ste3)

template = env.get_template('Automata.template')

template.globals['predecessors'] = atm._my_graph.predecessors

rendered_content = template.render(automata=atm)
print rendered_content

with open('/Users/gholamrezarahimi/Downloads/HDL/automata.v','w') as f:
    f.writelines(rendered_content)







