cython point_comperator.pyx
g++ -c -fPIC -std=c++11 point_comperator.c  -I ~/anaconda2/include/python2.7
g++ -undefined dynamic_lookup  point_comperator.o -o point_comperator.so