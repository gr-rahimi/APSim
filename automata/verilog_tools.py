'''
    The purpose of this tool is to convert our truth-table
    into a set of Verilog modules for use within our FPGA code.
    There is one truth table module per bit and one for reporting.
'''

from enum import Enum
import copy
import sys

class TruthTableType(Enum):
    """
    For now we support two different types of truth tables
    reporting: this combinational truth table sets the reporting bit/bits
    transition: this sequential truth table transitions between states, symbolically
    """
    REPORTING = 1
    TRANSITION = 2

class TruthTable:
    """
        This class encapsulates one truth table
    """
    def __init__(self, type, header):
        """
            Initialize a TruthTable object with a type and header
        """
        self.type = type
        self.header = header
        self.transitions = []
    
    def add_transition(self, transition):
        """
            Adds a transition to our Truth Table
        """
        self.transitions.append(transition)
    
    def __str__(self):
        """
            ToString() for our TruthTable class
        """
        return_string = "Truth Table type="
        return_string += 'REPORTING' if self.type == TruthTableType.REPORTING else 'TRANSITION'
        return_string += '\n'
        
        for k,v in self.header.items():
            if k not in ['next_state', 'output']:
                return_string += '[' + k + '=' + ','.join(v) + ']'
            else:
                return_string += '[' + k + '=' + v + ']'
        return_string += '\n'
        return_string += '--------------------------------------\n'
        for transition_dict in self.transitions:
            for k,v in transition_dict.items():
                return_string += '[' + k + '=' + ','.join(v) + ']'
            return_string += '\n'
        return return_string

def parse_truth_tables(tables, start_states, accept_states):
    """
        This function breaks the data into multiple
        truth tables to be passed to verilog generator
    """

    results = []
    tt = None

    for state_index, transitions in tables['transition'].items():

        headers = transitions[0]
        inputs = headers[0]
        previous_state = headers[1]
        next_state = headers[2]

        # Create a new TRANISITION type Truth Table
        tt = TruthTable(TruthTableType.TRANSITION,
            {
                'inputs': inputs,
                'previous_state': previous_state,
                'next_state': next_state
            }
        )

        for transition in transitions[1:]:

            inputs = transition[0]

            if tt.type == TruthTableType.TRANSITION:
                previous_state = transition[1]
                next_state = transition[2]

                # Add a transition from previous_state with inputs to next_state
                tt.add_transition(
                    {
                        'inputs': inputs,
                        'previous_state': previous_state,
                        'next_state': next_state
                    }
                )

            else:

                assert tt.type == TruthTableType.REPORTING, "ERROR: Incorrect TT Type"

                output = transition[1].strip()

                # Add a transition from inputs to outputs
                tt.add_transition(
                    {
                        'inputs': inputs,
                        'outputs': output
                    }
                )

        results.append(tt)

    reports = tables['report']
    headers = reports[0]
    inputs = headers[1]
    output = headers[2]

    tt = TruthTable(TruthTableType.REPORTING,
        {
            'inputs': inputs,
            'output': output
        }
    )

    for _, inputs, output in reports[1:]:
        tt.add_transition(
            {
                'inputs': inputs,
                'output': output
            }
        )

    results.append(tt)
    
    return results

def build_truthtable(tables, start_states, accept_states, module_name, output_verilog_file):
    """
    This function parses a truth tables and generates
    custom primitive truthtable files.
    """

    print "Starting building truthtables"

    truth_tables = parse_truth_tables(tables, start_states, accept_states)

    verilog_code = ""

    def define_tts(truth_tables, start_states):
        verilog_code = ""
        for truth_table in truth_tables:
            if truth_table.type == TruthTableType.REPORTING:
                verilog_code += make_combinationatorial_udp(truth_table)

            elif truth_table.type == TruthTableType.TRANSITION:
                verilog_code += make_sequential_udp(truth_table, start_states)
            else:
                raise Exception('Unsupported truth table type: {}'.format(truth_table.type))
            
            #Add a newline between tables
            verilog_code += "\n"
        return verilog_code

    # Write out the module definition
    inputs, vc = make_module(truth_tables, start_states, module_name, define_tts)
    verilog_code += vc

    try:
        with open(output_verilog_file, 'w') as f:
            f.write(verilog_code)
    
    except Exception as e:
        print("Cannot write to file {}".format(output_verilog_file))
        print("\tException: ", e)
        exit(-1)
    
    return inputs


def make_combinationatorial_udp(truth_table):
    """
    This function generates a TruthTable Verilog module
    This is only used for the reporting truthtable, which is combinational
    """

    output_name = truth_table.header['output']
    inputs = truth_table.header['inputs']
    transitions = truth_table.transitions

    verilog_code = "module {}TruthTable \n".format(output_name)
    verilog_code += "(\n"
    verilog_code += "\toutput reg {},\n".format(output_name)
    verilog_code += "\tinput wire {}\n".format(','.join(inputs))
    verilog_code += ");\n"
    verilog_code += "\n"
    verilog_code += "\talways @*\n"
    verilog_code += "\tcasex ({" + "{}".format(','.join(inputs)) + "})\n"
    verilog_code += "\t\t// {} : {}\n".format(' '.join(inputs), output_name)

    for transition in transitions:
        transition_inputs = transition['inputs']
        transition_output = transition['output']

        verilog_code += "\t\t{}'b{} : {} = 1'b{};\n".format(
            len(transition_inputs),
            ''.join(transition_inputs),
            output_name,
            transition_output
        )

    # For now, let's default to an output of 0; this might need to be changed
    verilog_code += "\t\tdefault : {} = 1'b0;\n".format(output_name)
        
    verilog_code += '\n'
    verilog_code += "\tendcase\n"
    verilog_code += "endmodule\n"
    verilog_code += '\n'

    return verilog_code


def make_start_sequential_udp(start_state, truth_table):
    """
    This function generates a TruthTable Verilog module
    This is used for start states
    """

    inputs = truth_table.header['inputs']

    verilog_code = "module {}TruthTable (\n".format(start_state)
    verilog_code += "\toutput reg {},\n".format(start_state)
    verilog_code += "\tinput wire clk, run, rst, {}\n".format(','.join(inputs))
    verilog_code += ");\n"
    verilog_code += "\treg {};\n".format(previous_state_name)
    verilog_code += "\tinitial\n"
    verilog_code += "\t\t{} = 1'b0;\n".format(start_state)
    verilog_code += "\n"
    verilog_code += "\talways @(posedge clk)\n"
    verilog_code += "\tcasex ({" + "{}".format(', '.join(inputs)) + "})\n"
    verilog_code += "\t\t//{} : {}\n".format(', '.join(inputs), start_state)

    for transition in transitions:
        transition_inputs = transition['inputs']

        previous_states = transition['previous_state']
        next_state_value = transition['next_state']

        transition_inputs += ''.join(previous_states)

        verilog_code += "\t\t{}'b{} : {} = 1'b{};\n".format(
            len(inputs),
            ''.join(['X' for x in range(len(inputs))]),
            start_state,
            1
        )
    verilog_code += "\t\t{} : {} = 1'b0;\n".format(
        "default",
        next_state
    )

    verilog_code += "\tendcase\n"
    verilog_code += "\n"
    verilog_code += "\talways @(posedge clk, posedge rst)\n"
    verilog_code += "\tbegin\n"
    verilog_code += "\t\tif(rst == 1'b1)\n"
    verilog_code += "\t\t\t{} = 1'b0;\n".format(previous_state_name)
    verilog_code += "\t\telse if (run == 1'b1)\n"
    verilog_code += "\t\t\t{} = {};\n".format(previous_state_name, next_state)
    verilog_code += "\tend\n"
    verilog_code += "endmodule\n\n"
    verilog_code += '\n'

    return verilog_code



def make_sequential_udp(truth_table, start_states):
    """
    This function generates a TruthTable Verilog module
    This is used for state transition logic, which is sequential
    """

    inputs = truth_table.header['inputs']

    previous_states = truth_table.header['previous_state']

    next_state = truth_table.header['next_state']

    # print "Start States: ", start_states
    # if next_state in start_states:
    #     print "A START STATE!!"
    #     exit()

    # This is a little tricky
    # If we have more than one bit in the previous_state
    # We need to take all other bits and consider those inputs
    # from other truth tables
    assert 'new' in next_state, 'Next State not properly named: {}'.format(next_state)

    # This is one of the previous state signals
    previous_state_name = next_state.replace('new', 'old')

    # Make sure that the naming convention is followed
    assert previous_state_name in previous_states, 'State names are not properly named: {}'.format(previous_state)
    
    # We'll use the index of the previous state and remove it below
    previous_state_index = previous_states.index(previous_state_name)

    transition_state_names = []
    for i, state in enumerate(previous_states):
        if i != previous_state_index:
            transition_state_names.append(state.replace('old', 'new'))
        else:
            transition_state_names.append(state)
    
    # Make a copy for external signals
    external_state_names = list(transition_state_names)

    # Now remove our one previous state signal
    external_state_names.remove(previous_state_name)

    # Add all of the new state bits to our inputs list
    module_inputs = list(inputs)
    module_inputs.extend(external_state_names)

    transitions = truth_table.transitions

    verilog_code = "module {}TruthTable (\n".format(next_state)
    verilog_code += "\toutput reg {},\n".format(next_state)
    verilog_code += "\tinput wire clk, run, rst, {}\n".format(','.join(module_inputs))
    verilog_code += ");\n"
    verilog_code += "\treg {};\n".format(previous_state_name)
    verilog_code += "\tinitial\n"
    verilog_code += "\t\t{} = 1'b0;\n".format(previous_state_name)
    verilog_code += "\n"
    verilog_code += "\talways @(posedge clk)\n"
    verilog_code += "\tcasex ({" + "{},{}".format(
        ', '.join(inputs), ', '.join(transition_state_names)) + "})\n"
    verilog_code += "\t\t//{} {} : {}\n".format(', '.join(inputs), ', '.join(transition_state_names), next_state)

    for transition in transitions:
        transition_inputs = transition['inputs']

        previous_states = transition['previous_state']
        next_state_value = transition['next_state']

        transition_inputs += ''.join(previous_states)

        verilog_code += "\t\t{}'b{} : {} = 1'b{};\n".format(
            len(transition_inputs),
            transition_inputs,#''.join(inputs),
            next_state,
            next_state_value
        )
    verilog_code += "\t\t{} : {} = 1'b0;\n".format(
        "default",
        next_state
    )

    verilog_code += "\tendcase\n"
    verilog_code += "\n"
    verilog_code += "\talways @(posedge clk, posedge rst)\n"
    verilog_code += "\tbegin\n"
    verilog_code += "\t\tif(rst == 1'b1)\n"
    verilog_code += "\t\t\t{} = 1'b0;\n".format(previous_state_name)
    verilog_code += "\t\telse if (run == 1'b1)\n"
    verilog_code += "\t\t\t{} = {};\n".format(previous_state_name, next_state)
    verilog_code += "\tend\n"
    verilog_code += "endmodule\n\n"
    verilog_code += '\n'

    return verilog_code


def make_module(truth_tables, start_states, module_name, define_tts):
    """
    This function generates the module interface for the symbolic
    finite state automaton.
    """

    module_inputs = None
    output = None
    wires = []

    # print "Truth Tables:"
    # for tt in truth_tables:
    #     print tt
    # print "Start States: "
    # for ss in start_states:
    #     print ss
    # print "Module Name: "
    # print module_name
    # exit()
    
    for truth_table in truth_tables:
        if truth_table.type == TruthTableType.TRANSITION:
            if not module_inputs:
                module_inputs = truth_table.header['inputs']
            else:
                assert truth_table.header['inputs'] == module_inputs, "Input assertion fails!"
            
            if truth_table.header['next_state'] not in wires:
                wires.append(truth_table.header['next_state'])
        
        elif truth_table.type == TruthTableType.REPORTING:
            if not output:
                output = truth_table.header['output']
            else:
                assert truth_table.header['output'] == output, "Output assertion fails!"

    verilog_code = "module {}({}, clk, run, rst, report);\n".format(module_name, ', '.join(module_inputs))
    verilog_code += "\n"
    verilog_code += "\tinput {}, clk, run, rst;\n".format(', '.join(module_inputs))
    verilog_code += "\toutput wire {};\n".format(output)
    verilog_code += "\twire {};\n".format(', '.join(wires))
    # for ss in start_states:
    #     wire_name = 'new' + str(ss)
    #     verilog_code += "\twire {};\n".format(wire_name)
    #     verilog_code += "\tassign {} = 1'b1;\n".format(wire_name)
    verilog_code += "\n"

    verilog_code += define_tts(truth_tables, start_states)

    # Instantiate all truth tables here
    verilog_code += "\t// Instantiate truth tables\n"

    for truth_table in truth_tables:

        if truth_table.type == TruthTableType.REPORTING:
            output = truth_table.header['output']
            inputs = truth_table.header['inputs']
            previous_states = []

        elif truth_table.type == TruthTableType.TRANSITION:
            output = truth_table.header['next_state']
            inputs = truth_table.header['inputs']

            # Get the names of input state signals from other modules
            previous_states = [x.replace('old', 'new') for x in truth_table.header['previous_state']]
            if output in previous_states:
                previous_states.remove(output)

        else:
            raise Exception("Unsupported Truth Table Type!")

        verilog_code += "\t{}TruthTable {}tt(\n".format(output, output)
        verilog_code += "\t\t.{}({}),\n".format(output, output)

        for input in inputs[:-1]:
            verilog_code += "\t\t.{}({}),\n".format(input, input)
        
        # If we don't have any external previous statedds, we're done
        if (len(previous_states) == 0) and truth_table.type != TruthTableType.TRANSITION:
            verilog_code += "\t\t.{}({})\n".format(inputs[-1], inputs[-1])
        else:
            verilog_code += "\t\t.{}({}),\n".format(inputs[-1], inputs[-1])

        for previous_state in previous_states:
            verilog_code += "\t\t.{}({}),\n".format(previous_state, previous_state)

        if truth_table.type == TruthTableType.TRANSITION:
            verilog_code += "\t\t.clk(clk),\n"
            verilog_code += "\t\t.run(run),\n"
            verilog_code += "\t\t.rst(rst)\n"

        verilog_code += "\t);\n"
        verilog_code += "\n"
        

    verilog_code += "endmodule\n"

    return module_inputs, verilog_code