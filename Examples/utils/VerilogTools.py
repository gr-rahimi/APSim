'''
    The purpose of this tool is to convert our truth-table
    into a set of Verilog modules for use within our FPGA code.
    There is one truth table module per bit and one for reporting.
'''

from enum import Enum
import copy
import sys


def read_truth_table_file(truth_table_file):
    """
    This function reads a truth table file
    and returns a list of string for each line
    """

    file_content = None

    # Open the truth table file and read
    with open(truth_table_file, 'r') as f:
        file_content = f.readlines()
    
    return file_content


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


def parse_truth_tables(truth_table_data):
    """
        This function breaks the data into multiple
        truth tables to be passed to the verilog generator
    """

    results = []
    tt = None
    header = True
    transitions = []

    # Go through each line
    for line in truth_table_data:
        
        # We found an empty line, which means the next
        # non-empty line is a header (or EOF)
        if not line.strip():

            # If we have valid transitions from the previous
            # truth table, add them to our list
            if len(tt.transitions) > 0:
                results.append(tt)
            header = True
            continue

        # We're in header mode
        if header:

            # Take each colon-separated header and strip off newlines and spaces
            headers = list(map(lambda x: x.strip(), line.split(':')))

            inputs = list(map(lambda x: x.strip(), headers[0].split()))

            # If we only have two header sections, this is a reporting table
            #            section 0                   section 1
            # new_state_bit_0 new_state_bit_1 ... : report
            if len(headers) == 2:
                output = headers[1].strip()
                assert output == 'report', "Output signal is not called 'report'!"
                tt = TruthTable(TruthTableType.REPORTING, 
                                {
                                    'inputs': inputs,
                                    'output': output}
                                )
            
            # Else, it must be a state-transition table, and therefore have 3 header sections
            #             section 0                section 1                 section 2
            # input_bit_0 input_bit_1 ... : old_state_0 old_state_1 ... : new_state_bit
            else:
                assert len(headers) == 3, "Cannot parse this line: {}".format(line)
                previous_state = list(map(lambda x: x.strip(), headers[1].split()))
                next_state = headers[2].strip()
                tt = TruthTable(TruthTableType.TRANSITION,
                                {
                                    'inputs': inputs,
                                    'previous_state': previous_state,
                                    'next_state': next_state
                                }
                            )

            # We're done with the header; start a new transitions list
            header = False
            transitions = []
        
        else:
            transition = list(map(lambda x: x.strip(), line.split(':')))

            inputs = list(map(lambda x: x.strip(), transition[0].split()))

            if tt.type == TruthTableType.TRANSITION:
                previous_state = list(map(lambda x: x.strip(), transition[1].split()))
                next_state = transition[2].strip()
                
                tt.add_transition({
                    'inputs': inputs,
                    'previous_state': previous_state,
                    'next_state': next_state
                })
            else:
                output = transition[1].strip()

                tt.add_transition({
                    'inputs': inputs,
                    'output': output
                })

    if len(transitions) > 0:
        results.append(tt)
    
    return results


def build_truthtable(truth_table_file, module_name, output_verilog_file):
    """
    This function parses a truth table (.tt) file and generates
    custom truthtable files.
    It returns input signal names and output report names
    """

    # Read in truth table data generated by Lucas's script
    data = read_truth_table_file(truth_table_file)
    truth_tables = parse_truth_tables(data)
    
    verilog_code = ""

    inputs = set()
    wires = set()
    outputs = set()
    for truth_table in truth_tables:
        if truth_table.type == TruthTableType.TRANSITION:
            for input in truth_table.header['inputs']:
                inputs.add(input)
            wires.add(truth_table.header['next_state'])
        
        elif truth_table.type == TruthTableType.REPORTING:
            outputs.add(truth_table.header['output'])

    # Write out the module definition
    verilog_code += make_module(truth_tables, module_name, inputs, wires, outputs)

    try:
        with open(output_verilog_file, 'w') as f:
            f.write(verilog_code)
    
    except Exception as e:
        print("Cannot write to file {}".format(output_verilog_file))
        print("\tException: ", e)
        exit(-1)
    
    # Return inputs and outputs
    return inputs, outputs


def make_combinationatorial_udp(truth_table):
    """
    This function generates a TruthTable Verilog module
    This is only used for the reporting truthtable, which is combinational
    """

    output_name = truth_table.header['output']
    inputs = truth_table.header['inputs']
    transitions = truth_table.transitions

    verilog_code = "\tmodule {}TruthTable \n".format(output_name)
    verilog_code += "\t(\n"
    verilog_code += "\t\toutput reg {},\n".format(output_name)
    verilog_code += "\t\t{}\n".format(','.join(['input ' + str(input) for input in inputs]))
    verilog_code += "\t);\n"
    verilog_code += "\n"
    verilog_code += "\t\talways @*\n"
    verilog_code += "\t\tcasex ({" + "{}".format(','.join(inputs)) + "})\n"
    verilog_code += "\t\t\t// {} : {}\n".format(' '.join(inputs), output_name)

    for transition in transitions:
        inputs = transition['inputs']
        output = transition['output']

        verilog_code += "\t\t\t{}'b{} : {} = 1'b{};\n".format(
            len(inputs),
            ''.join(inputs),
            output_name,
            output
        )

    # For now, let's default to an output of 0; this might need to be changed
    verilog_code += "\t\t\tdefault : {} = 1'b0;\n".format(output_name)
        
    verilog_code += '\n'
    verilog_code += "\t\tendcase\n"
    verilog_code += "\tendmodule\n"
    verilog_code += '\n'

    return verilog_code


def make_sequential_udp(truth_table):
    """
    This function generates a TruthTable Verilog module
    This is used for state transition logic, which is sequential
    """

    inputs = copy.copy(truth_table.header['inputs'])
    previous_states = copy.copy(truth_table.header['previous_state'])
    next_state_name = copy.copy(truth_table.header['next_state'])

    # This is a little tricky
    # If we have more than one bit in the previous_state
    # We need to take all other bits and consider those inputs
    # from other truth tables
    assert 'new' in next_state_name, 'Next State not properly named: {}'.format(next_state_name)

    # This is one of the previous state signals
    previous_state_name = next_state_name.replace('new', 'old')

    # Make sure that the naming convention is followed
    assert previous_state_name in previous_states, 'State names are not properly named: {}'.format(previous_state)
    
    # We'll use the index of the previous state and remove it below
    previous_state_index = previous_states.index(previous_state_name)

    # Now remove our one previous state signal from the remaining
    previous_states.remove(previous_state_name)

    # Replace all the other signals with 'new'; they'll be inputs from other TTs
    input_state_bits = []
    for state in previous_states:
        input_state = state.replace('old', 'new')
        input_state_bits.append(input_state)

    # Add all of the new state bits to our inputs list
    inputs.extend(input_state_bits)

    transitions = truth_table.transitions

    verilog_code = "\tmodule {}TruthTable (\n".format(next_state_name)
    verilog_code += "\t\toutput reg {},\n".format(next_state_name)
    verilog_code += "\t\tinput clk, input run, input rst, {}\n".format(','.join(['input ' + str(input) for input in inputs]))
    verilog_code += "\t);\n"
    verilog_code += "\t\treg {};\n".format(previous_state_name)
    verilog_code += "\t\tinitial\n"
    verilog_code += "\t\t{} = 1'b0;\n".format(previous_state_name)
    verilog_code += "\n"
    #verilog_code += "wire {};\n".format(', '.join(inputs))
    #verilog_code += "\n"
    verilog_code += "\t\talways @(posedge clk)\n"
    verilog_code += "\t\tcasex ({" + "{}".format(', '.join(inputs)) + ', ' + previous_state_name + "})\n"
    verilog_code += "\t\t\t// {} {} : {}\n".format(' '.join(inputs), previous_state_name, next_state_name)

    for transition in transitions:
        inputs = transition['inputs']
        previous_states = transition['previous_state']
        next_state = transition['next_state']

        for i,state in enumerate(previous_states):
            
            inputs.append(state)

            if i == previous_state_index:
                previous_state = state

        verilog_code += "\t\t\t{}'b{} : {} = 1'b{};\n".format(
            len(inputs),
            ''.join(inputs),
            next_state_name,
            next_state
        )

    verilog_code += "\t\tendcase\n"
    verilog_code += "\n"
    verilog_code += "\t\talways @(posedge clk, posedge rst)\n"
    verilog_code += "\t\tbegin\n"
    verilog_code += "\t\t\tif(rst == 1'b1)\n"
    verilog_code += "\t\t\t\t{} = 1'b0;\n".format(previous_state_name)
    verilog_code += "\t\t\telse if (run == 1'b1)\n"
    verilog_code += "\t\t\t\t{} = {};\n".format(previous_state_name, next_state_name)
    verilog_code += "\t\tend\n"
    verilog_code += "\tendmodule\n\n"
    verilog_code += '\n'

    return verilog_code


def make_module(truth_tables, module_name, inputs, wires, outputs):
    """
    This function generates the module interface for the symbolic
    finite state automaton.
    """

    verilog_code = "module {} (\n".format(module_name)
    assert len(outputs) == 1, "Assumption about outputs is false"
    verilog_code += "\toutput wire {},\n".format(', '.join(outputs))
    verilog_code += "\tinput clk, input run, input rst, {}\n".format(','.join(['input ' + str(input) for input in inputs]))
    verilog_code += ");\n"
    verilog_code += "\n"
    verilog_code += "\twire {};\n".format(', '.join(wires))
    verilog_code += "\n"

    # Define truth tables here

    for truth_table in truth_tables:
        if truth_table.type == TruthTableType.REPORTING:
            verilog_code += make_combinationatorial_udp(truth_table)

        elif truth_table.type == TruthTableType.TRANSITION:
            verilog_code += make_sequential_udp(truth_table)
        else:
            raise Exception('Unsupported truth table type: {}'.format(truth_table.type))
        
        #Add a newline between tables
        verilog_code += "\n"

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
        
        # If we don't have any external previous states, we're done
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

    return verilog_code

# Get the usage string
def usage():
    usage = "----------------- Usage ----------------\n"
    usage += "./VerilogTools.py <Truth Table Input> <Verilog Output>"
    return usage

if __name__ == '__main__':

    # Check the correct number of command line arguments
    if len(sys.argv) == 3:
        reverse = False
    else:
        print(usage())
        exit(-1)

    # Grab the input and output filenames
    tt_input = sys.argv[1] # This is the truth table input
    verilog_output = sys.argv[2] # This is the verilog output

    build_truthtable(tt_input, verilog_output)
