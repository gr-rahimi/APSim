import automata as atma
from automata.automata_network import compare_input, compare_strided

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Brill/anml/brill.1chip.anml")
print "Finished processing from anml file. Here is the summary"
automata.print_summary()
automata.remove_ors()
print len(automata.get_connected_components_size())

cc = automata.get_connected_components_as_automatas()

for c in cc:
    print len(c.get_nodes())







