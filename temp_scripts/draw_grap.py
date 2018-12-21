from automata.automata_network import Automatanetwork
from automata.elemnts.ste import S_T_E, PackedIntervalSet, PackedInterval, PackedInput
from automata.elemnts.element import StartType
from utility import minimize_automata

my_Automata = Automatanetwork(id="test1", is_homogenous=True, stride=1)

ste0 = S_T_E(start_type=StartType.start_of_data, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=PackedIntervalSet([PackedInterval(PackedInput((0, )), PackedInput((0,))),
                         PackedInterval(PackedInput((2,)), PackedInput((2,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=-1)

ste1 = S_T_E(start_type=StartType.start_of_data, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=PackedIntervalSet([PackedInterval(PackedInput((2,)), PackedInput((2,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=-1)

ste2 = S_T_E(start_type=StartType.start_of_data, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=PackedIntervalSet([PackedInterval(PackedInput((1,)), PackedInput((1,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=-1)

ste3 = S_T_E(start_type=StartType.non_start, is_report=True, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=PackedIntervalSet([PackedInterval(PackedInput((3,)), PackedInput((3,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=0)

my_Automata.add_element(ste0)
my_Automata.add_element(ste1)
my_Automata.add_element(ste2)
my_Automata.add_element(ste3)

my_Automata.add_edge(ste0, ste0)
my_Automata.add_edge(ste0, ste1)
my_Automata.add_edge(ste0, ste2)
my_Automata.add_edge(ste1, ste3)
my_Automata.add_edge(ste2, ste3)
my_Automata.add_edge(ste3, ste3)

my_Automata.draw_graph('fccm.svg', draw_edge_label=False, use_dot=True, write_node_labels=False)

atm2 = my_Automata.get_single_stride_graph()
atm2.draw_graph('fccmS2.svg', draw_edge_label=False, use_dot=True, write_node_labels=False)
atm2.make_homogenous()
minimize_automata(atm2, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=True)
atm2.draw_graph('fccmS2H.svg', draw_edge_label=False, use_dot=True, write_node_labels=False)