import pickle

nine_bit_snort = pickle.load(open('nine_bit_snort.pkl', 'rb'))

#nine_bit_snort.draw_graph('esi.svg', False)
atm = nine_bit_snort.get_single_stride_graph()