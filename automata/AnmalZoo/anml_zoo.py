from enum import Enum
import os.path

_base_address = "/home/gr5yf/Git/ANMLZoo"

class AnmalZoo(Enum):
    Brill = 0
    Dotstar = 1
    EntityResolution = 2  # for non homogeneous graphs, yes but will be determined
    Fermi = 3
    Hamming = 4
    Levenshtein = 5
    PowerEN = 6
    Protomata = 7
    RandomForest = 8
    Snort = 9
    SPM = 10
    Synthetic_BlockRings = 11
    Dotstar03 = 12
    Dotstar06 = 13
    Dotstar09 = 14
    Ranges05 = 15
    Ranges1 = 16
    ExactMath = 17
    Bro217 = 18
    TCP = 19
    ClamAV = 20
    Synthetic_CoreRings = 21
    Custom = 22

anml_path = {}
anml_path[AnmalZoo.Brill] = os.path.join(_base_address, "Brill/anml/brill.1chip.anml")
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
anml_path[AnmalZoo.Synthetic_BlockRings] = os.path.join(_base_address, "Synthetic/anml/BlockRings.anml")
anml_path[AnmalZoo.Synthetic_CoreRings] = os.path.join(_base_address, "Synthetic/anml/CoreRings.anml")
anml_path[AnmalZoo.Bro217] = os.path.join(_base_address, "Bro217/anml/bro217.anml")
anml_path[AnmalZoo.Dotstar03] = os.path.join(_base_address, "Dotstar03/anml/dotstar03.anml")
anml_path[AnmalZoo.Dotstar06] = os.path.join(_base_address, "Dotstar06/anml/dotstar06.anml")
anml_path[AnmalZoo.Dotstar09] = os.path.join(_base_address, "Dotstar09/anml/dotstar09.anml")
anml_path[AnmalZoo.ExactMath] = os.path.join(_base_address, "ExactMath/anml/exactmath.anml")
anml_path[AnmalZoo.Ranges1] = os.path.join(_base_address, "Ranges1/anml/ranges1.anml")
anml_path[AnmalZoo.Ranges05] = os.path.join(_base_address, "Ranges05/anml/ranges05.anml")
anml_path[AnmalZoo.TCP] = os.path.join(_base_address, "TCP/anml/tcp.anml")
anml_path[AnmalZoo.Custom] = os.path.join(_base_address, "Custom/anml/POST.anml")

input_path10M = {}
input_path10M[AnmalZoo.Brill] = os.path.join(_base_address, "Brill/inputs/brill_10MB.input")
input_path10M[AnmalZoo.ClamAV] = os.path.join(_base_address, "ClamAV/inputs/vasim_10MB.input")
input_path10M[AnmalZoo.Dotstar] = os.path.join(_base_address, "Dotstar/inputs/backdoor_10MB.input")
input_path10M[AnmalZoo.EntityResolution] = os.path.join(_base_address, "EntityResolution/inputs/1000_10MB.input")
input_path10M[AnmalZoo.Fermi] = os.path.join(_base_address, "Fermi/inputs/rp_input_10MB.input")
input_path10M[AnmalZoo.Hamming] = os.path.join(_base_address, "Hamming/inputs/hamming_10MB.input")
input_path10M[AnmalZoo.Levenshtein] = os.path.join(_base_address, "Levenshtein/inputs/DNA_10MB.input")
input_path10M[AnmalZoo.PowerEN] = os.path.join(_base_address, "PowerEN/inputs/poweren_10MB.input")
input_path10M[AnmalZoo.Protomata] = os.path.join(_base_address, "Protomata/inputs/uniprot_fasta_10MB.input")
input_path10M[AnmalZoo.RandomForest] = os.path.join(_base_address, "RandomForest/inputs/mnist_10MB.input")
input_path10M[AnmalZoo.SPM] = os.path.join(_base_address, "SPM/inputs/SPM_10MB.input")
input_path10M[AnmalZoo.Snort] = os.path.join(_base_address, "Snort/inputs/snort_10MB.input")
#input_path[AnmalZoo.Snort] = os.path.join(_base_address, "Snort/inputs/syn_input2.bin")
input_path10M[AnmalZoo.Synthetic_BlockRings] = os.path.join(_base_address, "Synthetic/inputs/10MB.input")
input_path10M[AnmalZoo.Synthetic_CoreRings] = os.path.join(_base_address, "Synthetic/inputs/10MB.input")
input_path10M[AnmalZoo.Bro217] = os.path.join(_base_address, "Bro217/inputs/Bro217_10MB_depth_s0_p0.75.trace.input")
input_path10M[AnmalZoo.Dotstar03] = os.path.join(_base_address, "Dotstar03/inputs/Dotstar03_10MB_depth_s0_p0.75.trace.input")
input_path10M[AnmalZoo.Dotstar06] = os.path.join(_base_address, "Dotstar06/inputs/Dotstar06_10MB_depth_s0_p0.75.trace.input")
input_path10M[AnmalZoo.Dotstar09] = os.path.join(_base_address, "Dotstar09/inputs/Dotstar09_10MB_depth_s0_p0.75.trace.input")
input_path10M[AnmalZoo.ExactMath] = os.path.join(_base_address, "ExactMath/inputs/ExactMath_10MB_depth_s0_p0.75.trace.input")
input_path10M[AnmalZoo.Ranges1] = os.path.join(_base_address, "Ranges1/inputs/Ranges1_10MB_depth_s0_p0.75.trace.input")
input_path10M[AnmalZoo.Ranges05] = os.path.join(_base_address, "Ranges05/inputs/Ranges05_10MB_depth_s0_p0.75.trace.input")
input_path10M[AnmalZoo.TCP] = os.path.join(_base_address, "TCP/inputs/tcp_depth_s0_p0.75.trace.input")
input_path10M[AnmalZoo.Custom] = os.path.join(_base_address, "Custom/inputs/input.inp")

input_path1M = {}
input_path1M[AnmalZoo.Brill] = os.path.join(_base_address, "Brill/inputs/brill_1MB.input")
input_path1M[AnmalZoo.ClamAV] = os.path.join(_base_address, "ClamAV/inputs/vasim_1MB.input")
input_path1M[AnmalZoo.Dotstar] = os.path.join(_base_address, "Dotstar/inputs/backdoor_1MB.input")
input_path1M[AnmalZoo.EntityResolution] = os.path.join(_base_address, "EntityResolution/inputs/1000_1MB.input")
input_path1M[AnmalZoo.Fermi] = os.path.join(_base_address, "Fermi/inputs/rp_input_1MB.input")
input_path1M[AnmalZoo.Hamming] = os.path.join(_base_address, "Hamming/inputs/hamming_1MB.input")
input_path1M[AnmalZoo.Levenshtein] = os.path.join(_base_address, "Levenshtein/inputs/DNA_1MB.input")
input_path1M[AnmalZoo.PowerEN] = os.path.join(_base_address, "PowerEN/inputs/poweren_1MB.input")
input_path1M[AnmalZoo.Protomata] = os.path.join(_base_address, "Protomata/inputs/uniprot_fasta_1MB.input")
input_path1M[AnmalZoo.RandomForest] = os.path.join(_base_address, "RandomForest/inputs/mnist_1MB.input")
input_path1M[AnmalZoo.SPM] = os.path.join(_base_address, "SPM/inputs/SPM_1MB.input")
input_path1M[AnmalZoo.Snort] = os.path.join(_base_address, "Snort/inputs/snort_1MB.input")
#input_path[AnmalZoo.Snort] = os.path.join(_base_address, "Snort/inputs/syn_input2.bin")
input_path1M[AnmalZoo.Synthetic_BlockRings] = os.path.join(_base_address, "Synthetic/inputs/1MB.input")
input_path1M[AnmalZoo.Synthetic_CoreRings] = os.path.join(_base_address, "Synthetic/inputs/1MB.input")
input_path1M[AnmalZoo.Bro217] = os.path.join(_base_address, "Bro217/inputs/bro217_depth_s0_p0.75.trace.input")
input_path1M[AnmalZoo.Dotstar03] = os.path.join(_base_address, "Dotstar03/inputs/dotstar0.3_depth_s0_p0.75.trace.input")
input_path1M[AnmalZoo.Dotstar06] = os.path.join(_base_address, "Dotstar06/inputs/dotstar0.6_depth_s0_p0.75.trace.input")
input_path1M[AnmalZoo.Dotstar09] = os.path.join(_base_address, "Dotstar09/inputs/dotstar0.9_depth_s0_p0.75.trace.input")
input_path1M[AnmalZoo.ExactMath] = os.path.join(_base_address, "ExactMath/inputs/exact-math_depth_s0_p0.75.trace.input")
input_path1M[AnmalZoo.Ranges1] = os.path.join(_base_address, "Ranges1/inputs/ranges1_depth_s0_p0.75.trace.input")
input_path1M[AnmalZoo.Ranges05] = os.path.join(_base_address, "Ranges05/inputs/ranges05_depth_s0_p0.75.trace.input")
input_path1M[AnmalZoo.TCP] = os.path.join(_base_address, "TCP/inputs/tcp_depth_s0_p0.75.trace.input")
input_path1M[AnmalZoo.Custom] = os.path.join(_base_address, "Custom/inputs/input.inp")





