import automata as atma
#import matplotlib.pyplot as plt
import networkx as nx
import os

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


for file in os.listdir("/root/elaheh/RezaSim/ANMLZoo"):
    if file.endswith(".anml"):
        file_path = os.path.join("/root/elaheh/RezaSim/ANMLZoo", file)
        print file, "\n"

        automata = atma.parse_anml_file(file_path)
        automata.remove_ors()
        print "Finished processing from anml file. Here is the summary"
        automata.print_summary()
        #automata.draw_graph("original.png", draw_edge_label= True)
        #automata.feed_file("/home/reza/Git/ANMLZoo/SPM/inputs/SPM_1MB.input")
        #print "finished feeding input file"
        connected_component_size_list = automata.get_connected_components_size()
        connected_component_size_list = connected_component_size_list[::-1]
        print "Maximum connected component size: ", connected_component_size_list[len(connected_component_size_list)-1]
        print "Number of connected components: ", len(connected_component_size_list)
        #print connected_component_size_list

        harware_block_size = 256
        used_harware_blocks = cal_utilization (connected_component_size_list, harware_block_size)
        print "Total number of used hardware blocks: ", used_harware_blocks
        if used_harware_blocks != 0:
            print "Harware utilization:", sum(connected_component_size_list)/float(harware_block_size * used_harware_blocks)
        print "\n\n"



