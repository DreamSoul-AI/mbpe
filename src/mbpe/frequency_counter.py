import pickle
import os
from collections import defaultdict


class FrequencyCounter:
    def __init__(self):
        pass

    def load_cumulative_frequency(self, file_path, reset=False):
        """
        Load the cumulative frequency dictionary from a pickle file.
        If the file does not exist or `reset=True`, return an empty dictionary.
        """
        if reset or not os.path.exists(file_path):  # Check if the file exists
            return defaultdict(int), 0  # Initialize: an empty tuple count dictionary and a total tuple count
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return defaultdict(int), 0  # Initialize: an empty tuple count dictionary and total tuple count

    def save_cumulative_frequency(self, cumulative_tuple_counts, total_tuple_count, file_path):
        """
        Save the cumulative frequency dictionary to a pickle file.

        Parameters:
        cumulative_tuple_counts (dict): Dictionary of cumulative tuple counts.
        total_tuple_count (int): Total count of tuples.
        file_path (str): File path to save the frequency dictionary.
        """
        with open(file_path, "wb") as f:
            pickle.dump((cumulative_tuple_counts, total_tuple_count), f)

    def save_vocab(self, vocab, file_path):
        """
        Save the vocabulary to a pickle file.

        Parameters:
        vocab (dict): A dictionary containing tuples and their corresponding frequencies.
        file_path (str): The file path to save the vocabulary.
        """
        with open(file_path, "wb") as f:
            pickle.dump(vocab, f)

    def calculate_and_filter_frequency(self, input_list, freq_file_path, vocab_file_path=None, reset=False, min_root_freq=None):
        """
        Calculate the frequency of tuples in the current list, save the frequency dictionary,
        and optionally filter tuples with frequencies greater than a threshold.

        Parameters:
        input_list (list): A list containing tuples and other types of data.
        freq_file_path (str): The file path to save the frequency dictionary.
        vocab_file_path (str): Optional. The file path to save the vocabulary dictionary.
        reset (bool): If True, clear the saved frequency data.
        min_root_freq (float): Optional. If provided, filter tuples with frequencies greater than this value.

        Returns:
        dict: A dictionary containing filtered vocabulary if min_root_freq is provided, otherwise the cumulative frequency dictionary.
        """
        # Load existing cumulative frequency data
        cumulative_tuple_counts, total_tuple_count = self.load_cumulative_frequency(freq_file_path, reset=reset)

        # Iterate through the current list, count the occurrences of tuples, and update
        for item in input_list:
            if isinstance(item, tuple):
                cumulative_tuple_counts[item] += 1
                total_tuple_count += 1

        # Save the updated cumulative frequency data
        self.save_cumulative_frequency(cumulative_tuple_counts, total_tuple_count, freq_file_path)

        # Calculate cumulative frequency
        cumulative_frequency = {key: count / total_tuple_count for key, count in cumulative_tuple_counts.items()}

        # Save the complete frequency table (before filtering)
        self.save_cumulative_frequency(cumulative_tuple_counts, total_tuple_count, freq_file_path)

        # If min_root_freq is provided, filter the frequency dictionary
        if min_root_freq is not None:
            filtered_vocab = {key: freq for key, freq in cumulative_frequency.items() if freq > min_root_freq}

            # Save the filtered vocabulary if a file path is provided
            if vocab_file_path:
                self.save_vocab(filtered_vocab, vocab_file_path)

            return cumulative_frequency, filtered_vocab

        return cumulative_frequency, None
