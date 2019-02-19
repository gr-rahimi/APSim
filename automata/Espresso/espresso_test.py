from jinja2 import Environment, FileSystemLoader
from automata.elemnts.ste import PackedIntervalSet, PackedInterval, PackedInput

symset= PackedIntervalSet([])

symset.add_interval(PackedInterval(PackedInput((10,20)), PackedInput((40, 60))))
symset.add_interval(PackedInterval(PackedInput((80,100)), PackedInput((85, 110))))

env = Environment(loader=FileSystemLoader('automata/Espresso/Templates'), extensions=['jinja2.ext.do'])
template = env.get_template('espresso.template')
out_str=template.render(symset=symset, max_val=255)
print out_str

