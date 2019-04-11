import automata as atma


automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/ClamAV/anml/515_nocounter.1chip.anml")
print "Finished processing ClamAV"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()



automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Dotstar/anml/backdoor_dotstar.1chip.anml")
print "Finished processing Dotstar"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()


automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/EntityResolution/anml/1000.1chip.anml")
print "Finished processing Entity Resolution"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Fermi/anml/fermi_2400.1chip.anml")
print "Finished processing fermi"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Hamming/anml/93_20X3.1chip.anml")
print "Finished processing Hamming"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Levenshtein/anml/24_20x3.1chip.anml")
print "Finished processing Levenstein"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/PowerEN/anml/complx_01000_00123.1chip.anml")
print "Finished processing PowerEN"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Protomata/anml/2340sigs.1chip.anml")
print "Finished processing Protomata"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/RandomForest/anml/300f_15t_tree_from_model_MNIST.anml")
print "Finished processing Random forest"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Snort/anml/snort.1chip.anml")
print "Finished processing Snort"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/SPM/anml/bible_size4.1chip.anml")
print "Finished processing SPM"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Synthetic/anml/BlockRings.anml")
print "Finished processing Synthetic"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()

automata = atma.parse_anml_file("/home/reza/Git/ANMLZoo/Brill/anml/brill.1chip.anml")
print "Finished processing Brill"
automata.remove_ors()
automata.print_summary()
print "connecting component size=", automata.get_connected_components_size()