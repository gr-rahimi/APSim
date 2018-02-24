import automata as atma
from automata.automata_network import compare_input

automata = atma.parse_anml_file("/home/reza/Downloads/POST/simple1.anml")
print "Finished processing from anml file. Here is the summary"
automata.print_summary()
automata.remove_ors()
automata.remove_all_start_nodes()
automata.draw_graph("h.png", True)

strided_automata_NH = automata.get_single_stride_graph()
strided_automata_NH.draw_graph("snh.png", True)

strided_automata_NH.make_homogenous()
strided_automata_NH.draw_graph("sh.png", True)
strided_automata_NH.draw_graph("shnolabel.png", False)
strided_automata_NH.does_have_self_loop()
compare_input(True,"/home/reza/Downloads/POST/simple1.txt", automata, strided_automata_NH)

