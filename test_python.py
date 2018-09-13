import networkx as nx

class A(object):
    def __init__(self, id, temp):
        self.temp = temp
        self._id = id

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        res = (self._id == other)
        return self._id == other

    def __str__(self):
        return str(self._id)



G = nx.DiGraph()

G.add_node(A(0,1))
G.add_node(A(1,1))
G.add_node(A(2,1))
G.add_node(A(3,1))

G.add_edge(0,1)
G.add_edge(1,0)


for e in G.edges:
    print e

















