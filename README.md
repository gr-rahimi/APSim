# APSim 

APSim is an automata compiler using python. It supports many essential automata compiling features such as minimization, fan in/out constraint, connected components extraction,... etc.
APSim uses NetworkX as its underlying data straucture to maintain the automataon as a graph and run automaton processing algorithms to reshape the underlying graph.

Setup
-----

External dependencies: `g++, swig, python`
OS: Linux, mac OS

1. Clone a fresh copy of the git APSim repository (`git clone -b ASPLOS_AE https://github.com/gr-rahimi/APSim.git`).

2. Download and Install Anaconda (python 2.7)

3. Install the following python packages using all available in Anaconda repositories:

    `sortedcontainers, numpy, matplotlib, pathos, networkx, deap, tqdm, Jinja2, pygraphviz`

4. Go to the CPP folder and run the compile script with python include directoy path. For example:

    `./compile ~/anaconda2/include/python2.7/`
    
5. Add APSim to your PYTHONPATH

    `export PYTHONPATH=$PYTHONPATH:/home/foo/APSim`

6. Clone a fresh copy of ANMLZoo

    `git clone https://github.com/gr-rahimi/ANMLZoo.git`

7. Update the variable ANMLZoo's address path in APSim's module `automata/AnmalZoo/anml_zoo.py` variable `_base_address`
