import os
import pickle

initial_freq_tab = {}

if not os.path.exists('freq_tab.pkl'):
    with open('freq_tab.pkl', 'wb') as f:
        pickle.dump(initial_freq_tab, f)

for tup in tuple_list_2_2:  # 1 image tuple list

    with open('freq_tab.pkl', 'rb') as f:
        freq_tab = pickle.load(f)

    if tup in freq_tab:
        freq_tab[tup] += 1

        with open('freq_tab.pkl', 'wb') as f:
            pickle.dump(freq_tab, f)

    else:
        freq_tab[tup] = 1

        with open('freq_tab.pkl', 'wb') as f:
            pickle.dump(freq_tab, f)
