#!/bin/bash
#set -x
echo "FPGA results" > results.txt

summary_files=(`find . -name utilization.txt`)

for f in "${summary_files[@]}"
do
    benchmark_path=$(dirname $f)
    benchmark_name=$(basename $benchmark_path)
    echo $benchmark_name >> results.txt
    echo $(grep "| Top_Module" $f) >> results.txt
    timing_slacks=( $(grep "slack" $benchmark_path/timing_summary.txt) )
    echo max delay=${timing_slacks[1]} >> results.txt
    echo $(grep "Total On-Chip Power" $benchmark_path/power_summary.txt) >> results.txt
    echo "-------------------------------------------------------------" >> results.txt
done    

