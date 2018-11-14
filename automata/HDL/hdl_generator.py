from jinja2 import Environment, FileSystemLoader
import sys


def generate_full_lut(atm):
    env = Environment(loader=FileSystemLoader('automata/HDL/Templates'), extensions=['jinja2.ext.do'])
    template = env.get_template('Automata.template')
    template.globals['predecessors'] = atm._my_graph.predecessors
    rendered_content = template.render(automata=atm)
    print rendered_content
    with open('/Users/gholamrezarahimi/Downloads/HDL/automata.v', 'w') as f:
        f.writelines(rendered_content)


