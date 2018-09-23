import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream

snort_automata = atma.parse_anml_file(anml_path[AnmalZoo.Snort])
snort_automata.remove_ors()
snort_automatas = snort_automata.get_connected_components_as_automatas()

for idx, automata in tqdm(enumerate(snort_automatas[3:])):
    print "automata:", idx
    automata.remove_all_start_nodes()
    st2 = automata.get_single_stride_graph()
    st4 = st2.get_single_stride_graph()
    st4.print_summary()
    print "start striding"
    st8= st4.get_single_stride_graph()
    #st4.print_summary()
    #print "starting making homogeneous"
    st8.print_summary()
    print "start homo"
    st8.make_homogenous()
    #print "starting minimizing"
    minimize_automata(st8, merge_reports=True, same_residuals_only=True, same_report_code=True)
    st8.print_summary()
