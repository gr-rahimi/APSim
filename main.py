import automata as atma
from automata.automata_network import compare_input

automata = atma.parse_anml_file("/home/reza/Downloads/POST/POST.anml")
print "Finished processing from anml file. Here is the summary"
automata.print_summary()
automata.remove_ors()
automata.remove_all_start_nodes()
strided_automata_NH = automata.get_single_stride_graph()
strided_automata_NH.make_homogenous()
strided_automata_NH.print_summary()
compare_input(True,"/home/reza/Git/ANMLZoo/Brill/inputs/brill_10MB.input", automata, strided_automata_NH)

