import automata as atma
from automata.automata_network import compare_input, compare_strided, StartType
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
import pickle
from utility import minimize_automata, multi_byte_stream
import automata.HDL.hdl_generator as hd_gen



automatas = pickle.load(open('snort1-10.pkl', 'rb'))

atm = automatas[0]
atm.remove_all_start_nodes()
atm.remove_ors()
atm.print_summary()

atm2=atm.get_single_stride_graph()
atm2.make_homogenous()
minimize_automata(atm2, merge_reports=True, same_residuals_only=True, same_report_code=True,
                      combine_symbols=True)

hd_gen.generate_full_lut(atm2)



