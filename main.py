import automata as atma
from automata.automata_network import compare_input, compare_strided

automata = atma.parse_anml_file("/home/reza/Downloads/POST/POST.anml")
print "Finished processing from anml file. Here is the summary"
automata.print_summary()
automata.remove_ors()
automata.remove_all_start_nodes()
strided_automata_NH = automata.get_single_stride_graph()
strided_automata_NH.make_homogenous()

for node in strided_automata_NH.get_nodes():
    for other_node in strided_automata_NH.get_nodes():
        node.is_symbolset_a_subsetof_self_symbolset(other_symbol_set=other_node.get_symbols())


strided_automata_NH2= strided_automata_NH.get_single_stride_graph()
strided_automata_NH.print_summary()
strided_automata_NH2.make_homogenous()
strided_automata_NH2.print_summary()
#compare_input(True,"/home/reza/Git/ANMLZoo/Brill/inputs/brill_10MB.input", automata, strided_automata_NH)
left_automata, right_automata = strided_automata_NH.split()
compare_strided(True, "/home/reza/Git/ANMLZoo/Brill/inputs/brill_1MB.input",[strided_automata_NH],[left_automata,right_automata],[strided_automata_NH2])

