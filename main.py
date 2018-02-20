import automata as atma
import matplotlib.pyplot as plt
import networkx as nx

automata = atma.parse_anml_file("/home/reza/Downloads/POST/simple1.anml")
print "Finished processing from anml file. Here is the summary"
automata.print_summary()
automata.draw_graph("original.png")


strided_automata = automata.get_single_stride_graph()
print "finished double striding. Here is the summary"
strided_automata.print_summary()
strided_automata.draw_graph("strided.png", draw_edge_label= True)

left_automata, right_automata = strided_automata.split()

print "left automata:"
left_automata.print_summary()
left_automata.draw_graph("left_automata.png",draw_edge_label = True)

print "right automata:"
right_automata.print_summary()
right_automata.draw_graph("right_automata.png",draw_edge_label = True)

#second_strided = strided_automata.get_single_stride_graph()
#print "finished second double striding. Here is the summary"
#second_strided.print_summary()



