
def total_tuples_counts(tuple_list):
    """
    Count the number of tuples in a list.
    """
    total_tuple = sum(1 for element in tuple_list if isinstance(element, tuple))

    return total_tuple


def tuples_counts(tuple_list):
    """
    Count the occurrences of each tuple in a list.
    """
    tuples = [element for element in tuple_list if isinstance(element, tuple)]
    counts_tuples = Counter(tuples)

    return counts_tuples


def process_root_vocabulary(tuple_list, state, min_root_freq):
    """
    Process the root vocabulary by calculating the frequency of each tuple in the list.
    Only tuples with frequency greater than min_root_freq will be added to the root_vocab dictionary.
    """
    total_tuples = total_tuples_counts(tuple_list)
    tuple_counts = tuples_counts(tuple_list)

    root_vocab = {}

    if state.shape == [1, 1, 1]:
        for tuple_item, count in tuple_counts.items():
            if count >= 1:
                root_vocab[tuple_item] = True

    else:
        for tuple_item, count in tuple_counts.items():
            frequency = count / total_tuples
            if frequency > min_root_freq:
                root_vocab[tuple_item] = True

    return root_vocab


def assign_code_to_vocab(vocab, start_value):
    """
    Assign values to the dictionary keys starting from start_value.
    """
    assigned_vocab = {}
    current_value = start_value

    for key in vocab:
        assigned_vocab[key] = current_value
        current_value += 1

    return assigned_vocab


def build_vocabulary(tuple_list, min_freq, root_vocab):

    vocab = set(root_vocab)

    while True:
        stats = freq_pair(tuple_list)

        if not stats:
            break

        pair = max(stats, key=stats.get)
        pair_freq = stats[pair] / sum(stats.values())

        if pair_freq > min_freq:
            vocab.add(pair)
            tuple_list = merge(tuple_list, pair)
        else:
            break

    return vocab


def encode(tuple_list, vocab, start_value):
    """
    Encode the tuple list by replacing each tuple with its corresponding code from vocab.
    """
    assigned_vocab = assign_code_to_vocab(vocab, start_value)
    encoded_tuple_list = [
        assigned_vocab.get(tuple_item, tuple_item) for tuple_item in tuple_list
    ]

    return encoded_tuple_list



