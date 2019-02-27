import automata as atma
from automata.AnmalZoo.anml_zoo import anml_path, AnmalZoo
from tqdm import tqdm
tqdm.monitor_interval = 0
from automata.utility.utility import minimize_automata
import csv
import os, shutil

samples_per_automata = 10

result_path = "CSV_ICCD"
if not os.path.exists(result_path):
    os.makedirs(result_path)  # make directory if it does not exist
else:
    shutil.rmtree(result_path)  # remove content
    os.makedirs(result_path)  # make directory if it does not exist

for automata_name, automata_path in anml_path.iteritems():
    if automata_name == AnmalZoo.Custom or automata_name == AnmalZoo.Synthetic_CoreRings or\
                    automata_name == AnmalZoo.TCP:
        continue


    print "start processing ", str(automata_name)

    automata = atma.parse_anml_file(automata_path)
    automata.remove_ors()

    orig_automatas = automata.get_connected_components_as_automatas()
    orig_automatas = orig_automatas[0:samples_per_automata]
    total_result = []
    for orig_cc_idx, orig_cc in tqdm(enumerate(orig_automatas), unit="automata"):
        orig_cc.remove_all_start_nodes()
        strides_list = [orig_cc] # original
        strides_list.append(strides_list[0].get_single_stride_graph()) #S2
        strides_list.append(strides_list[1].get_single_stride_graph()) #S4
        strides_list.append(strides_list[2].get_single_stride_graph()) #S8

        for scc in strides_list:
            result=[]
            result.append((scc.get_number_of_nodes(without_fake_root=True), scc.get_number_of_edges()))
            if(not scc.is_homogeneous()):
                scc.make_homogenous()
            result.append((scc.get_number_of_nodes(without_fake_root=True), scc.get_number_of_edges()))
            minimize_automata(scc, merge_reports = False, same_residuals_only = False, same_report_code = False)
            result.append((scc.get_number_of_nodes(without_fake_root=True), scc.get_number_of_edges()))
            minimize_automata(scc, merge_reports=True, same_residuals_only=True, same_report_code=True)
            result.append((scc.get_number_of_nodes(without_fake_root=True), scc.get_number_of_edges()))
            minimize_automata(scc, merge_reports=True, same_residuals_only=True, same_report_code=False)
            result.append((scc.get_number_of_nodes(without_fake_root=True), scc.get_number_of_edges()))
            minimize_automata(scc, merge_reports=True, same_residuals_only=False, same_report_code=False)
            result.append((scc.get_number_of_nodes(without_fake_root=True), scc.get_number_of_edges()))
            total_result.append(result)

    del orig_automatas, strides_list, automata

    with open(os.path.join(result_path, str(automata_name)+".csv"), 'w') as csvfile:
        field_names = ['Snn','Sne','SHnn','SHne',
                       'SHM1nn','SHM1ne','SHM2nn','SHM2ne',
                       'SHM3nn','SHM4ne','SHM4nn','SHM4ne']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        for result in total_result:
            row = {}
            row[field_names[0]] = result[0][0]
            row[field_names[1]] = result[0][1]
            row[field_names[2]] = result[1][0]
            row[field_names[3]] = result[1][1]
            row[field_names[4]] = result[2][0]
            row[field_names[5]] = result[2][1]
            row[field_names[6]] = result[3][0]
            row[field_names[7]] = result[3][1]
            row[field_names[8]] = result[4][0]
            row[field_names[9]] = result[4][1]
            row[field_names[10]] = result[5][0]
            row[field_names[11]] = result[5][1]

            writer.writerow(row)
