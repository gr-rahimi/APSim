#!/bin/bash
rm *.so *.o *.cxx *.pyc *.py
touch __init__.py
g++ -c -fPIC -fPIC -std=c++11  VASim.cpp
swig -c++ -python VASim.i
g++ -c -fPIC -std=c++11 VASim_wrap.cxx  -I /home/reza/anaconda2/include/python2.7  -I /home/reza/anaconda2/lib/python2.7/
g++ -shared -Wl,-soname,_VASim.so -o _VASim.so VASim.o VASim_wrap.o
