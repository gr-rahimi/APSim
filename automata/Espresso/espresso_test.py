import subprocess
from jinja2 import Environment, FileSystemLoader
from automata.elemnts.ste import PackedIntervalSet, PackedInterval, PackedInput

from automata.Espresso.espresso import get_splitted_sym_sets

import re

symset= PackedIntervalSet([])

symset.add_interval(PackedInterval(PackedInput((5, 5)), PackedInput((8, 9))))
symset.add_interval(PackedInterval(PackedInput((4, 5)), PackedInput((7, 9))))
symset.add_interval(PackedInterval(PackedInput((8, 1)), PackedInput((9, 6))))

a,b = get_splitted_sym_sets(symset=symset, max_val=15)
print a,b

