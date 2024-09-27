import os
import pickle

initial_freq_tab_2_2 = {}

if not os.path.exists('freq_tab_2_2.pkl'):
    with open('freq_tab_2_2.pkl', 'wb') as f:
        pickle.dump(initial_freq_tab_2_2, f)

for image_list in tuple_list_2_2:

    for tup in image_list:

        with open('freq_tab_2_2.pkl', 'rb') as f:
            freq_tab_2_2 = pickle.load(f)

        if tup in freq_tab_2_2:
            freq_tab_2_2[tup] += 1

            with open('freq_tab_2_2.pkl', 'wb') as f:
                pickle.dump(freq_tab_2_2, f)

        else:
            freq_tab_2_2[tup] = 1

            with open('freq_tab_2_2.pkl', 'wb') as f:
                pickle.dump(freq_tab_2_2, f)

    # update root vocabulary, e.g. min_freq = 2
    # tuple (freq > 2) → root code

    # merge
    # update vocabulary: root code + merge code
    # tuple list → include tuple, root code, merge code

    # reshape dim (2, 2) tuple list → dim (2, 1) tuple list
    # count dim (2, 1) tuple list's tuple frequency
