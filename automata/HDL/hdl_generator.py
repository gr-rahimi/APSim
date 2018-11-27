from jinja2 import Environment, FileSystemLoader
import networkx
import shutil, os


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
        for automata in atms:
            rendered_content = template.render(automata=automata, capture_symbol=capture_symbol)
            print rendered_content
            with open(os.path.join(total_path, automata.id+'.v',), 'w') as f:
                f.writelines(rendered_content)

        template = env.get_template('Top_Module.template')
        rendered_content = template.render(automatas=atms, single_file=single_file, single_out=False)
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

