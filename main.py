import automata as atma
import matplotlib.pyplot as plt
import networkx as nx

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Levenshtein/anml/24_20x3.1chip.anml")
print "Finished processing from anml file. Graph has %d nodes."%automata.get_number_of_nodes()
automata.get_single_stride_graph()
print "finished double stride"


