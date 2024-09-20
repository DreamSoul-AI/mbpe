import os
import pickle

initial_freq_tab = {}
initial_prob_tab = {}

if not os.path.exists('freq_tab.pkl'):
    with open('freq_tab.pkl', 'wb') as f:
        pickle.dump(initial_freq_tab, f)

if not os.path.exists('prob_tab.pkl'):
    with open('prob_tab.pkl', 'wb') as f:
        pickle.dump(initial_prob_tab, f)

for tup in tuple_list_2_2:  # 1 image tuple list

    with open('freq_tab.pkl', 'rb') as f:
        freq_tab = pickle.load(f)

    with open('prob_tab.pkl', 'rb') as f:
        prob_tab = pickle.load(f)

    if tup in freq_tab:
        freq_tab[tup] += 1

        total_count = sum(freq_tab.values())

        prob_tab = {
            tup: count / total_count for tup, count in freq_tab.items()
        }

        with open('freq_tab.pkl', 'wb') as f:
            pickle.dump(freq_tab, f)

        with open('prob_tab.pkl', 'wb') as f:
            pickle.dump(prob_tab, f)


    else:
        freq_tab[tup] = 1
        total_count = sum(freq_tab.values())
        prob_tab = {
            tup: count / total_count for tup, count in freq_tab.items()
        }

        with open('freq_tab.pkl', 'wb') as f:
            pickle.dump(freq_tab, f)

        with open('prob_tab.pkl', 'wb') as f:
            pickle.dump(prob_tab, f)