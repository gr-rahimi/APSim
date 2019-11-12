#!/bin/bash

concurrent_vivado=4

run_vivado(){
    
    cd "$(dirname "$1")";
    vivado -mode tcl -source my_script.tcl
    return 0
}    

export -f run_vivado

find . -name my_script.tcl | xargs -n 1 -P $concurrent_vivado -I {} bash -c 'run_vivado "$@"' _ {}
