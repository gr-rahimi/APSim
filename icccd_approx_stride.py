import automata as atma
from automata.automata_network import compare_input, compare_strided
from anml_zoo import anml_path,input_path,AnmalZoo
from tqdm import tqdm
tqdm.monitor_interval = 0
import pickle
from utility import minimize_automata
import csv
import os, shutil

samples_per_automata = 5

result_path = "APPX_CSV_ICCD"
if not os.path.exists(result_path):
    os.makedirs(result_path)  # make directory if it does not exist
else:
    #shutil.rmtree(result_path)  # remove content
    #os.makedirs(result_path)  # make directory if it does not exist
    pass

for automata_name, automata_path in anml_path.iteritems():
    if automata_name == AnmalZoo.Custom or automata_name == AnmalZoo.Synthetic_CoreRings or\
                    automata_name == AnmalZoo.TCP or automata_name != AnmalZoo.Levenshtein :
        continue
    if os.path.exists(os.path.join(result_path, str(automata_name)+".csv")):
        continue

    print "start processing ", str(automata_name)

    automata = atma.parse_anml_file(automata_path)
    automata.remove_ors()

    orig_automatas = automata.get_connected_components_as_automatas()
    orig_automatas = orig_automatas[0:samples_per_automata]
    total_result = []
    for orig_cc_idx, orig_cc in tqdm(enumerate(orig_automatas), unit="automata"):
        orig_cc.remove_all_start_nodes()
        strides_list = []
        strides_list.append(orig_cc.get_single_stride_graph()) #S2
        strides_list.append(strides_list[0].get_single_stride_graph()) #S4
        strides_list.append(strides_list[1].get_single_stride_graph()) #S8

        for scc in strides_list:
            result_wof = atma.compare_real_approximate(input_path[automata_name], scc)
            result_wof_nn = scc.get_number_of_nodes(without_fake_root = True)

            final_nodes = scc.get_filtered_nodes(lambda node: node.is_report())
            for fn in final_nodes:
                scc._make_homogeneous_STE(fn, delete_original_ste=True)


            result_wf = atma.compare_real_approximate(input_path[automata_name], scc)

            result_wf_nn = scc.get_number_of_nodes(without_fake_root=True)

            total_result.append((result_wof, result_wf, result_wof_nn, result_wf_nn ))

    del orig_automatas, strides_list, automata

    with open(os.path.join(result_path, str(automata_name)+".csv"), 'w') as csvfile:
        field_names = ['false_actives1','false_reports_exact1','false_reports_each_cycle1','true_total_reports1',
                       'false_actives2', 'false_reports_exact2', 'false_reports_each_cycle2', 'true_total_reports2',
                       'nn1','nn2',]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for result in total_result:
            row = {}
            row[field_names[0]] = result[0][0]
            row[field_names[1]] = result[0][1]
            row[field_names[2]] = result[0][2]
            row[field_names[3]] = result[0][3]
            row[field_names[4]] = result[1][0]
            row[field_names[5]] = result[1][1]
            row[field_names[6]] = result[1][2]
            row[field_names[7]] = result[1][3]
            row[field_names[8]] = result[2]
            row[field_names[9]] = result[3]


            writer.writerow(row)
