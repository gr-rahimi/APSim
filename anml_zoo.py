from enum import Enum
import os.path

_base_address = "/home/reza/Git/ANMLZoo"

class AnmalZoo(Enum):
    Brill = 0
    ClamAV = 1
    Dotstar = 2
    EntityResolution = 3 # for non homogeneous graphs, yes but will be determined
    Fermi = 4
    Hamming = 5
    Levenshtein = 6
    PowerEN = 7
    Protomata = 8
    RandomForest = 9
    SPM = 10
    Snort = 11
    Synthetic = 12

anml_path = {}
anml_path[AnmalZoo.Brill] = os.path.join(_base_address, "Bril/anml/brill.1chip.anml")
anml_path[AnmalZoo.ClamAV] = os.path.join(_base_address, "ClamAV/anml/515_nocounter.1chip.anml")
anml_path[AnmalZoo.Dotstar] = os.path.join(_base_address, "Dotstar/anml/backdoor_dotstar.1chip.anml")
anml_path[AnmalZoo.EntityResolution] = os.path.join(_base_address, "EntityResolution/anml/1000.1chip.anml")
anml_path[AnmalZoo.Fermi] = os.path.join(_base_address, "Fermi/anml/fermi_2400.1chip.anml")
anml_path[AnmalZoo.Hamming] = os.path.join(_base_address, "Hamming/anml/93_20X3.1chip.anml")
anml_path[AnmalZoo.Levenshtein] = os.path.join(_base_address, "Levenshtein/anml/24_20x3.1chip.anml")
anml_path[AnmalZoo.PowerEN] = os.path.join(_base_address, "PowerEN/anml/complx_01000_00123.1chip.anml")
anml_path[AnmalZoo.Protomata] = os.path.join(_base_address, "Protomata/anml/2340sigs.1chip.anml")
anml_path[AnmalZoo.RandomForest] = os.path.join(_base_address, "RandomForest/anml/rf.1chip.anml")
anml_path[AnmalZoo.SPM] = os.path.join(_base_address, "SPM/anml/bible_size4.1chip.anml")
anml_path[AnmalZoo.Snort] = os.path.join(_base_address, "Snort/anml/snort.1chip.anml")
anml_path[AnmalZoo.Synthetic] = os.path.join(_base_address, "Synthetic/anml/BlockRings.anml")

input_path = {}
input_path[AnmalZoo.Brill] = os.path.join(_base_address, "Brill/inputs/brill_1MB.input")
input_path[AnmalZoo.ClamAV] = os.path.join(_base_address, "ClamAV/inputs/vasim_1MB.input")
input_path[AnmalZoo.Dotstar] = os.path.join(_base_address, "Dotstar/inputs/backdoor_1MB.input")
input_path[AnmalZoo.EntityResolution] = os.path.join(_base_address, "EntityResolution/inputs/1000_1MB.input")
input_path[AnmalZoo.Fermi] = os.path.join(_base_address, "Fermi/inputs/rp_input_1MB.input")
input_path[AnmalZoo.Hamming] = os.path.join(_base_address, "Hamming/inputs/hamming_1MB.input")
input_path[AnmalZoo.Levenshtein] = os.path.join(_base_address, "Levenshtein/inputs/DNA_1MB.input")
input_path[AnmalZoo.PowerEN] = os.path.join(_base_address, "PowerEN/inputs/poweren_1MB.input")
input_path[AnmalZoo.Protomata] = os.path.join(_base_address, "Protomata/inputs/uniprot_fasta_1MB.input")
input_path[AnmalZoo.RandomForest] = os.path.join(_base_address, "RandomForest/inputs/mnist_1MB.input")
input_path[AnmalZoo.SPM] = os.path.join(_base_address, "SPM/inputs/SPM_1MB.input")
input_path[AnmalZoo.Snort] = os.path.join(_base_address, "Snort/inputs/snort_1MB.input")
input_path[AnmalZoo.Synthetic] = os.path.join(_base_address, "Synthetic/inputs/1MB.input")



