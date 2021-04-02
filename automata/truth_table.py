def generate_transition_table(automata):
    """Generate a non-homogeneous transition table for a TT representation"""

    # Supported start types
    start_types = {0: None, 1: 'start-of-data', 2: 'all-input'}

    # A dictionary relating (source_state, destination_state) to transition_alphabet
    transition_dict = {}
    start_states = []
    accept_states = []

    # Converts from ANML-type character sets to lists of sorted accepted symbols
    def alphabet(character_set):

        symbol_list = []

        # For converting hex to ints, we need to use the '0xFF' format
        temp = character_set.replace('\\', '0')
        symbol_set = [x for x in temp.split('0x') if len(x) > 0]

        i  = 0
        while i < len(symbol_set):
            value = symbol_set[i]

            if '-' in value:
                start = int('0x' + value[:-1], 16)
                end = int('0x' + symbol_set[i+1], 16)
                symbol_list.extend([x for x in range(start, end + 1)])
                i += 2
            else:
                value = int('0x' + value, 16)
                symbol_list.append(value)
                i += 1

        symbol_list.sort()

        return symbol_list

    # Iterate through all of the states
    for node in automata.nodes:

        # Throw away the fake
        if not node.is_fake:

            start_type = start_types[node._start_type]

            # For now we can only support start_on_all
            if start_type:
                start_states.append(node.id)
            
            # Keep track of accepting (reporting) states
            if node._is_report:
                accept_states.append(node.id)

            # Make the current node a source node
            source_state = node.id

            # Iterate through all neighbors
            for n in automata.get_neighbors(node):

                destination_state = automata.get_STE_by_id(n).id

                # Generate symbol set
                character_set =  expand_symbol_set(automata.get_STE_by_id(n)._symbol_set)
                transition_alphabet = alphabet(character_set)

                assert n == destination_state, "Check for id consistancy failed; bummer"

                if (source_state, destination_state) not in transition_dict:
                    transition_dict[(source_state, destination_state)] = transition_alphabet
                else:
                    transition_dict[(source_state, destination_state)].extend(transition_alphabet)

    # Check to make sure that all nodes are reachable
    for node in automata.nodes:
        if not node.is_fake:
            found = False
            for (src, dst) in transition_dict.keys():
                if dst == node:
                    found = True
            if not found:
                assert node in start_states, "Found unreachable node {} that is not a start state!".format(node)

    # Make starting nodes reachable
    for node in start_states:

        character_set =  expand_symbol_set(automata.get_STE_by_id(node)._symbol_set)
        transition_alphabet = alphabet(character_set)
        transition_dict[(None, node)] = transition_alphabet

    print "Done building transition dict"

    return transition_dict, start_states, accept_states


def generate_tt(automata):
    """ generate a truthtable output """

    # Supported start types
    start_types = {0: None, 1: 'start-of-data', 2: 'all-input'}

    # We will have a truth table per state (per bit in the vector rep.)
    nodes = [n for n in automata.nodes if not n.is_fake]

    state_names = [str(node.id) for node in nodes]

    # Use this dictionary to index state names
    state_index_dict = {}
    for i in range(len(state_names)):
        state_index_dict[int(state_names[i])] = i 

    # We will have a bit per state
    bit_length = len(state_names)

    # The names of the input signals
    input_names = ['bit_'+str(i) for i in range(7,-1,-1)]

    # Names of old signals
    old_state_names = ['old' + s for s in state_names]

    # Grab a transition dict
    transition_dict, start_states, accept_states = generate_transition_table(automata)

    # We create a reverse dictionary from destination to labels to source
    # That is, reverse_dict[d][l] is the list of source states that
    # transition to destination state d on label l
    reverse_dict = {}

    def add_to_reverse_dict(dest, label, source):
        if dest not in reverse_dict:
            reverse_dict[dest] = {label: [source]}
        elif label not in reverse_dict[dest]:
            reverse_dict[dest][label] = [source]
        else:
            reverse_dict[dest][label].append(source)
    
    # Populate reverse dictionary mapping destination and label to a source
    for (source, dest), labels in transition_dict.items():
        for label in labels:
            add_to_reverse_dict(dest, label, source)

    del transition_dict

    # We're interested in transition and report tables
    tables = {'report':[], 'transition':{}}
    label_format_string = '0%db' % len(input_names)


    print "Done building reverse dict"

    # For each destination state...
    for dest, labels_to_sources in reverse_dict.items():

        header = (input_names, old_state_names, 'new' + str(dest))

        # Create a new state for each destination state
        # First line will be the header
        table = [header]

        # For each possible input (label), map the state to 0 or 1
        for label, sources in labels_to_sources.items():

            # Convert the label into a binary representation
            binary_label = format(label, label_format_string)

            # Iteratively construct the rows for a given transition label
            # Example: if label = 101 and sources = [0, 2, 3] then the rows are
            # 101 1XXX 1
            # 101 XX1X 1
            # 101 XXX1 1

            # For each source state
            for source in sources:

                # Create a full dont-care row
                next_row = ['X'] * bit_length

                # Create an entry in the table that sets our 
                # current state to 1 if the source is a 1

                if source:
                    next_row[state_index_dict[int(source)]] = '1'

                # Add to the table
                table.append((binary_label, list(next_row), '1'))

        tables['transition'][dest] = table
    
    print "Done with transitions"

    new_state_names = ['new' + s for s in state_names]
    report_table = [(None, new_state_names, 'report')]

    for state in accept_states:

        # Create a full dont-care row
        report_row = ['X'] * bit_length

        # Create a row in the the table that sets
        # the report to 1 if the source is 1
        report_row[state_index_dict[int(state)]] = '1'

        # Add to the report table
        report_table.append((None, list(report_row), '1'))

    tables['report'] = report_table

    print("Finished generating tables")

    return tables, start_states, accept_states


def expand_symbol_set(symbol_set):
    """
        This function generates an ANML-compatible symbol-set string
    """

    character_set = ""

    # Go through each interval and combine symbols
    for ss in symbol_set._interval_set._list:
        lower = ss._left_pt[0]
        upper = ss._right_pt[0]

        # If the range is one symbol wide, add that one symbol
        if lower == upper:
           character_set += r"\x%02X" % int(lower)
        else:
            # If it's more than one symbol wide, represent a range
            character_set += r"\x%02X-\x%02X" % (int(lower), int(upper))

    # Return the generated character-set string
    return character_set