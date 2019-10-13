# APSim

APSim is an automata compiler using python. It supports many essential automata compiling features such as minimization, fan in/out constraint, connected components extraction,... etc.
APSim uses NetworkX as its underlying data straucture to maintain the automataon as a graph and run automaton processing algorithms to reshape the underlying graph.

APSim works closely with [ANMLZoo](https://github.com/jackwadden/ANMLZoo).

There are two steps required to use APSim:

* In CPP folder, run comile.sh script.
* After cloning ANMLZoo, just update the path in  anml_zoo.py (_base_address) to directly use them in APSim.

