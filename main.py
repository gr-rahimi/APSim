import automata as atma
import matplotlib.pyplot as plt
import networkx as nx
from profilehooks import profile
from automata.automata_network import compare_input

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/PowerEN/anml/complx_01000_00123.1chip.anml")
print "Finished processing from anml file. Here is the summary"
automata.print_summary()

for active_states, is_report in automata.feed_file("/home/reza/Git/ANMLZoo/Dotstar/inputs/backdoor_1MB.input"):
    if is_report:
        print "report detected."


strided_automata_NH = automata.get_single_stride_graph()
strided_automata_NH.make_homogenous()


compare_input(True,"/home/reza/Downloads/POST/simple1.txt", automata, strided_automata_NH)

