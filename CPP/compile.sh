#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Include the path to the python2.7 include directory"
    exit 1
fi

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

	
	
Python_Path=$1 

if [ ! -d $Python_Path ]
then
    echo "Python include path does not exist"
    exit 1 # die with error code 1
fi

rm *.so *.o *.cxx *.pyc *.py 2> /dev/null
touch __init__.py
g++ -c -fPIC -fPIC -std=c++11  VASim.cpp
swig -c++ -python VASim.i
g++ -c -fPIC -std=c++11 VASim_wrap.cxx  -I $Python_Path

if [ $machine = Linux ]
then
    g++ -shared -Wl,-soname,_VASim.so -o _VASim.so VASim.o VASim_wrap.o
elif [ $machine = Mac ]
then
    g++ -undefined dynamic_lookup  VASim.o VASim_wrap.o -o _VASim.so
else
    echo "this operating system is not supported"
    exit 2    
fi

me=`dirname "$0"`
so_path="$me/_VASim.so"

if [ -f "$so_path" ]
then
	echo "Done!"
else
	echo "Unsuccessful. Please follow instructions in www.swig.org to install swig and run this script again"
fi


