import shutil
import os
import automata as atma
import automata.elemnts.ste as ste
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from automata.utility.utility import minimize_automata
import csv
import logging

uat = atma.Automatanetwork(id="test", is_homogenous=False, stride=1, max_val=255)

node1 = ste.S_T_E(start_type=atma.StartType.unknown, is_report=False, is_marked=False,
                               id=uat.get_new_id(), symbol_set=None, adjacent_S_T_E_s=None,
                               report_residual=0, report_code=-1)
uat.add_element(node1)

p1 = ste.PackedInput((1,))
p2 = ste.PackedInput((1,))
i1 = ste.PackedInterval(p1, p2)
uat.add_edge(uat.fake_root, node1, start_type=atma.StartType.start_of_data,
             symbol_set=ste.PackedIntervalSet([i1]))


node2 = ste.S_T_E(start_type=atma.StartType.unknown, is_report=False, is_marked=False,
                               id=uat.get_new_id(), symbol_set=None, adjacent_S_T_E_s=None,
                               report_residual=0, report_code=-1)
uat.add_element(node2)
p1 = ste.PackedInput((2,))
p2 = ste.PackedInput((2,))
i1 = ste.PackedInterval(p1, p2)
uat.add_edge(uat.fake_root, node2, start_type=atma.StartType.start_of_data,
             symbol_set=ste.PackedIntervalSet([i1]))


node3 = ste.S_T_E(start_type=atma.StartType.unknown, is_report=False, is_marked=False,
                               id=uat.get_new_id(), symbol_set=None, adjacent_S_T_E_s=None,
                               report_residual=0, report_code=-1)
uat.add_element(node3)
p1 = ste.PackedInput((3,))
p2 = ste.PackedInput((3,))
i1 = ste.PackedInterval(p1, p2)
uat.add_edge(uat.fake_root, node3, start_type=atma.StartType.start_of_data,
             symbol_set=ste.PackedIntervalSet([i1]))

node4 = ste.S_T_E(start_type=atma.StartType.unknown, is_report=False, is_marked=False,
                               id=uat.get_new_id(), symbol_set=None, adjacent_S_T_E_s=None,
                               report_residual=0, report_code=-1)
uat.add_element(node4)
p1 = ste.PackedInput((4,))
p2 = ste.PackedInput((4,))
i1 = ste.PackedInterval(p1, p2)
uat.add_edge(uat.fake_root, node4, start_type=atma.StartType.start_of_data,
             symbol_set=ste.PackedIntervalSet([i1]))

node10 = ste.S_T_E(start_type=atma.StartType.unknown, is_report=True, is_marked=False,
                               id=uat.get_new_id(), symbol_set=None, adjacent_S_T_E_s=None,
                               report_residual=0, report_code=1)
uat.add_element(node10)

p1 = ste.PackedInput((1,))
p2 = ste.PackedInput((2,))
i1 = ste.PackedInterval(p1, p2)

p3 = ste.PackedInput((3,))
p4 = ste.PackedInput((4,))
i2 = ste.PackedInterval(p3, p4)

p5 = ste.PackedInput((1,))
p6 = ste.PackedInput((4,))
i3 = ste.PackedInterval(p5, p6)

p5 = ste.PackedInput((2,))
p6 = ste.PackedInput((3,))
i4 = ste.PackedInterval(p5, p6)

uat.add_edge(node1, node10, start_type=atma.StartType.non_start,
             symbol_set=ste.PackedIntervalSet([i1]))

uat.add_edge(node2, node10, start_type=atma.StartType.non_start,
             symbol_set=ste.PackedIntervalSet([i2]))

uat.add_edge(node3, node10, start_type=atma.StartType.non_start,
             symbol_set=ste.PackedIntervalSet([i3]))

uat.add_edge(node4, node10, start_type=atma.StartType.non_start,
             symbol_set=ste.PackedIntervalSet([i4]))
uat.draw_graph("uatNH.svg")


uat.make_homogenous(use_espresso=True)

uat.draw_graph("uat.svg")

