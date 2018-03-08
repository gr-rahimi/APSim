import automata as atma
#import matplotlib.pyplot as plt
import networkx as nx
import os
from anml_zoo import anml_path, AnmalZoo

#calculate how many hardware block (of size hardware_block_sz) is needed to fit connected components using greedy approach.
def cal_utilization (cc_size, hardware_block_sz):
    curr_utilization = 0
    tot_hardware_blocks = 0
    for i in range(0,len(cc_size)):
        if cc_size[i] > hardware_block_sz:
            print "Connected component size is larger than hardware block size"
            return 0
        if curr_utilization + cc_size[i] < hardware_block_sz:
            curr_utilization += cc_size[i]
            if i == len(cc_size)-1:
                tot_hardware_blocks += 1
        else:
            tot_hardware_blocks += 1
            curr_utilization =  cc_size[i]
    return tot_hardware_blocks



automata = atma.parse_anml_file(anml_path[AnmalZoo.PowerEN])
automata.remove_ors()
print "Finished processing from anml file. Here is the summary"
automata.print_summary()
#automata.draw_graph("original.png", draw_edge_label= True)
#automata.feed_file("/home/reza/Git/ANMLZoo/SPM/inputs/SPM_1MB.input")
#print "finished feeding input file"

connected_components_list = automata.get_connected_components_as_automatas()
connected_component_size_list = automata.get_connected_components_size()
#connected_component_size_list = connected_component_size_list[::-1]
#print "Maximum connected component size: ", connected_component_size_list[len(connected_component_size_list)-1]
#print "Number of connected components: ", len(connected_component_size_list)
#print connected_component_size_list

#harware_block_size = 256
#used_harware_blocks = cal_utilization (connected_component_size_list, harware_block_size)
#print "Total number of used hardware blocks: ", used_harware_blocks
#if used_harware_blocks != 0:
#    print "Harware utilization:", sum(connected_component_size_list)/float(harware_block_size * used_harware_blocks)
#print "\n\n"


#calcualte connectivity utilization
for i in range(0, len(connected_components_list)):
    print(connected_components_list[i].get_STEs_out_degree())