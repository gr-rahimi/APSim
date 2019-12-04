# APSim (Manual under construction ...)

APSim is an automata compiler using python. It supports many essential automata compiling features such as minimization, fan in/out constraint, connected components extraction,... etc.
APSim uses NetworkX as its underlying data straucture to maintain the automataon as a graph and run automaton processing algorithms to reshape the underlying graph.

Setup
-----

External dependencies: `g++, swig, python`
OS: Linux, mac OS

1. Clone a fresh copy of the git APSim repository (`git clone <path to APSim repo>`).

2. Download and Install Anaconda (pyhton 2.7)

3. Install the following python packages:

    `sortedcontainers, numpy, matplotlib, pathos, networkx, deap, tqdm, Jinja2, pygraphviz`

4. Go to the CPP folder and run the compile script with python include directoy path. For example:
    `./compile ~/anaconda2/include/python2.7/`
    
5. Add APSim to your PYTHONPATH

