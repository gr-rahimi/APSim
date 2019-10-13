from automata.automata_network import Automatanetwork, get_bit_automaton, get_strided_automata2
from automata.elemnts.ste import S_T_E, PackedIntervalSet, PackedInterval, PackedInput, get_Symbol_type
from automata.elemnts.element import StartType
from utility import minimize_automata, get_equivalent_symbols


my_Automata = Automatanetwork(id="test1", is_homogenous=True, stride=1, max_val=255)

ste0 = S_T_E(start_type=StartType.all_input, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=get_Symbol_type(True)([PackedInterval(PackedInput((0, )), PackedInput((112,))),
                         PackedInterval(PackedInput((114,)), PackedInput((255,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=-1)

ste1 = S_T_E(start_type=StartType.non_start, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=get_Symbol_type(True)([PackedInterval(PackedInput((1,)), PackedInput((1,))),
                                           PackedInterval(PackedInput((2,)), PackedInput((2,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=-1)

# ste2 = S_T_E(start_type=StartType.non_start, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
#              symbol_set=PackedIntervalSet([PackedInterval(PackedInput((1,)), PackedInput((1,)))]),
#              adjacent_S_T_E_s=None, report_residual=0, report_code=-1)

ste2 = S_T_E(start_type=StartType.non_start, is_report=True, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=get_Symbol_type(True)([PackedInterval(PackedInput((3,)), PackedInput((3,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=0)

my_Automata.add_element(ste0)
my_Automata.add_element(ste1)
my_Automata.add_element(ste2)
#my_Automata.add_element(ste3)

my_Automata.add_edge(ste0, ste0)
my_Automata.add_edge(ste0, ste1)
my_Automata.add_edge(ste1, ste2)
my_Automata.add_edge(ste2, ste2)
my_Automata.draw_graph('t.svg')
x = get_bit_automaton(my_Automata, 8)
x.draw_graph('x.svg')
y = get_strided_automata2(atm=x, stride_value=8, is_scalar=True, base_value=2)
y.draw_graph('y.svg')
y.make_homogenous()
y.draw_graph('n.svg')
minimize_automata(y)
y.draw_graph('z.svg')

# my_Automata.add_edge(ste2, ste3)
# my_Automata.add_edge(ste3, ste3)

#my_Automata.remove_all_start_nodes()

# ste0 = S_T_E(start_type=StartType.start_of_data, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
#              symbol_set=PackedIntervalSet([PackedInterval(PackedInput((0, )), PackedInput((0,)))]),
#              adjacent_S_T_E_s=None, report_residual=0, report_code=-1)
#
# ste1 = S_T_E(start_type=StartType.non_start, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
#              symbol_set=PackedIntervalSet([PackedInterval(PackedInput((1,)), PackedInput((1,))),
#                                            PackedInterval(PackedInput((2,)), PackedInput((2,)))]),
#              adjacent_S_T_E_s=None, report_residual=0, report_code=-1)
#
# ste2 = S_T_E(start_type=StartType.non_start, is_report=True, is_marked=False, id = my_Automata.get_new_id(),
#              symbol_set=PackedIntervalSet([PackedInterval(PackedInput((3,)), PackedInput((3,)))]),
#              adjacent_S_T_E_s=None, report_residual=0, report_code=5)
#
#
# my_Automata.add_element(ste0)
# my_Automata.add_element(ste1)
# my_Automata.add_element(ste2)
#
#
# my_Automata.add_edge(ste0, ste0)
# my_Automata.add_edge(ste0, ste1)
# my_Automata.add_edge(ste1, ste2)
# my_Automata.add_edge(ste2, ste2)

my_Automata.draw_graph('fccm.svg', draw_edge_label=True, use_dot=True, write_node_labels=False)

atm2 = my_Automata.get_single_stride_graph()
atm2.draw_graph('fccmS2.svg', draw_edge_label=True, use_dot=True, write_node_labels=False)
atm2.make_homogenous()
get_equivalent_symbols([atm2])
atm2.draw_graph('fccmNMS2H.svg', draw_edge_label=True, use_dot=True, write_node_labels=False)
minimize_automata(atm2, merge_reports=True, same_residuals_only=False, same_report_code=False,
                      combine_symbols=True)
atm2.draw_graph('fccmS2H.svg', draw_edge_label=True, use_dot=True, write_node_labels=False)

