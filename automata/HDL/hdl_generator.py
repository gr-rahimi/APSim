from jinja2 import Environment, FileSystemLoader
import networkx
import shutil, os
from automata.automata_network import Automatanetwork
from automata.elemnts.element import FakeRoot



def _get_top_module_summary(atms):
    total_nodes = 0
    total_reports = 0
    total_edges = 0
    total_sym_count = 0

    for atm in atms:
        total_nodes += atm.nodes_count
        total_reports += sum (1 for r in atm.get_filtered_nodes(lambda ste: ste.report))
        total_edges += atm.get_number_of_edges()
        for n in atm.nodes:
            if n.id != FakeRoot.fake_root_id:
                total_sym_count += len(n.symbols)


    str_list = ['******************** Summary {}********************']

    str_list.append("total nodes = {}".format(total_nodes))
    str_list.append("total reports = {}".format(total_reports))
    str_list.append("total edges = {}".format(total_edges))
    str_list.append("average symbols len = {}".format(float(total_sym_count) / total_nodes))
    str_list.append('#######################################################')

    return '\n'.join(str_list)


def generate_full_lut(atms, capture_symbol = True , single_file = True, folder_name = None):
    env = Environment(loader=FileSystemLoader('automata/HDL/Templates'), extensions=['jinja2.ext.do'])

    if single_file:
        # Single file option is not maintained and will be removed
        template = env.get_template('Top_Module.template')
        template.globals['predecessors'] = networkx.MultiDiGraph.predecessors
        rendered_content = template.render(automatas=atms, single_file=single_file, single_out=True)
        with open('/Users/gholamrezarahimi/Downloads/HDL/automata.v', 'w') as f:
            f.writelines(rendered_content)

    else:

        total_path = os.path.join("/Users/gholamrezarahimi/Downloads/HDL", folder_name)
        shutil.rmtree(total_path, ignore_errors=True)
        os.mkdir(total_path)
        template = env.get_template('single_STE.template')
        rendered_content = template.render()
        with open(os.path.join(total_path, 'ste.v'), 'w') as f:
            f.writelines(rendered_content)

        template = env.get_template('Single_Automata.template')
        template.globals['predecessors'] = networkx.MultiDiGraph.predecessors
        template.globals['get_summary'] = Automatanetwork.get_summary # maybe better to move to utility module
        for automata in atms:
            rendered_content = template.render(automata=automata, capture_symbol=capture_symbol)
            with open(os.path.join(total_path, automata.id+'.v',), 'w') as f:
                f.writelines(rendered_content)

        template = env.get_template('Top_Module.template')
        rendered_content = template.render(automatas=atms, single_file=single_file, single_out=False,
                                           summary_str=_get_top_module_summary(atms))
        with open(os.path.join(total_path, 'top_module.v'), 'w') as f:
            f.writelines(rendered_content)

        # TCL script
        template = env.get_template('tcl.template')
        rendered_content = template.render()
        with open(os.path.join(total_path, 'my_script.tcl'), 'w') as f:
            f.writelines(rendered_content.encode('utf-8'))

        # Timing constrains
        template = env.get_template('clk_constrain.template')
        rendered_content = template.render()
        with open(os.path.join(total_path, 'clk_constrain.xdc'), 'w') as f:
            f.writelines(rendered_content)

