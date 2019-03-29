from automata.automata_network import Automatanetwork, get_bit_automaton
from automata.elemnts.ste import S_T_E, PackedIntervalSet, PackedInterval, PackedInput, get_Symbol_type
from automata.elemnts.element import StartType
from automata.utility.utility import minimize_automata

my_Automata = Automatanetwork(id="test1", is_homogenous=True, stride=1, max_val=255)

ste1 = S_T_E(start_type=StartType.all_input, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=get_Symbol_type(True)([PackedInterval(PackedInput((1, )), PackedInput((1,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=-1)

ste2 = S_T_E(start_type=StartType.non_start, is_report=False, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=get_Symbol_type(True)([PackedInterval(PackedInput((0,)), PackedInput((255,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=-1)

ste3 = S_T_E(start_type=StartType.non_start, is_report=True, is_marked=False, id = my_Automata.get_new_id(),
             symbol_set=PackedIntervalSet([PackedInterval(PackedInput((3,)), PackedInput((3,)))]),
             adjacent_S_T_E_s=None, report_residual=0, report_code=1)


my_Automata.add_element(ste1)
my_Automata.add_element(ste2)
my_Automata.add_element(ste3)

my_Automata.add_edge(ste1, ste2)
my_Automata.add_edge(ste2, ste3)
my_Automata.draw_graph('orig.svg')

b=get_bit_automaton(my_Automata, original_bit_width=8)
b.make_homogenous()
minimize_automata(b)
b.draw_graph('t.svg')
print  b.get_summary()

