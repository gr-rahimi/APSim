import automata as atma
from automata.automata_network import compare_input, compare_strided

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Snort/anml/snort.1chip.anml")
print "Finished processing from anml file. Here is the summary"
automata.remove_ors()
automata.print_summary()


automata.right_merge()
automata.print_summary()

#compare_input(True, "/home/reza/Git/ANMLZoo/Brill/inputs/brill_1MB.input", cc[0], strided_automata)







