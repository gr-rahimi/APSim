from igraph import *

g1 = Graph(directed = True)
g1.add_vertices(3)

g1.add_edges([(0,1),(1,2),(2,1)])

g2 = Graph(directed = True)
g2.add_vertices(3)

g2.add_edges([(0,1),(1,2)])

print g1.subisomorphic_lad(g2)

