from collections import defaultdict
import torch

class RootFrequencyCounter:
    def __init__(self, root_entrance_freq, root_vocab_freq):
        """
        Initialize RootFrequencyCounter
        
        Args:
            root_entrance_freq: Minimum frequency threshold for tuples to enter root vocabulary
            root_vocab_freq: Final frequency threshold for vocabulary entry
        """
        self.root_entrance_freq = root_entrance_freq
        self.root_vocab_freq = root_vocab_freq
        self.root_global_freq_table = {}  # Store tuples and their global frequencies
        self.total_count = 0  # Total count of all processed tuples
        
    def initialize_batch_counts(self):
        """Step 1: Initialize batch_counts dictionary."""
        return {}

    def count_frequencies_in_tensor(self, tensor_data, batch_counts):
        """Step 2: Count frequency of each item in the current tensor."""
        output, counts = torch.unique(tensor_data, sorted=False, return_counts=True, dim=0)
        batch_size = len(tensor_data)
        
        for tuple_data, count in zip(output.tolist(), counts.tolist()):
            tuple_key = tuple(tuple_data)
            batch_freq = count / batch_size
            batch_counts[tuple_key] = {
                'count': count,
                'frequency': batch_freq
            }
        return batch_counts

    def update_existing_tuples(self, batch_counts, batch_size):
        """Step 3: Update global frequencies for existing tuples."""
        for tuple_key in list(self.root_global_freq_table.keys()):
            current_count = batch_counts.get(tuple_key, {'count': 0})['count']
            old_count = self.root_global_freq_table[tuple_key]['global_count']
            new_count = old_count + current_count
            new_freq = new_count / (self.total_count + batch_size)
            
            self.root_global_freq_table[tuple_key].update({
                'global_count': new_count,
                'global_frequency': new_freq
            })

    def process_new_tuples(self, batch_counts):
        """Step 4: Process new tuples that meet the entrance frequency threshold."""
        for tuple_key, info in batch_counts.items():
            if tuple_key not in self.root_global_freq_table and info['frequency'] >= self.root_entrance_freq:
                self.root_global_freq_table[tuple_key] = {
                    'global_count': info['count'],
                    'global_frequency': info['frequency']  # Use batch frequency as global frequency for new tuples
                }

    def process_batch(self, tensor_data):
        """
        Process a batch of data
        
        Args:
            tensor_data: Tensor data containing tuples
        """
        # Step 1: Initialize batch_counts
        batch_counts = self.initialize_batch_counts()
        
        # Step 2: Count frequencies in the current batch
        batch_counts = self.count_frequencies_in_tensor(tensor_data, batch_counts)
        
        # Step 3: Update global frequencies for existing tuples
        batch_size = len(tensor_data)
        self.update_existing_tuples(batch_counts, batch_size)
        
        # Step 4: Process new tuples
        self.process_new_tuples(batch_counts)
        
        # Step 5: Update total count
        self.total_count += batch_size
    
    def get_root_vocabulary(self):
        """
        Generate final vocabulary based on root_global_freq_table and root_vocab_freq
        
        Returns:
            dict: Root vocabulary with frequency information
        """
        root_vocabulary = {}
        for tuple_key, info in self.root_global_freq_table.items():
            if info['global_frequency'] >= self.root_vocab_freq:
                root_vocabulary[tuple_key] = info
        return root_vocabulary
        
    def train_single_epoch(self, data, epoch_num, verbose=True):
        """
        Train for a single epoch
        
        Args:
            data: List of batches of data
            epoch_num: Current epoch number (for display purposes)
            verbose: Whether to print detailed information
            
        Returns:
            dict: Root vocabulary after this epoch
        """
        if verbose:
            print(f"\nStarting Epoch {epoch_num}:")
        
        # Reset counter state
        self.root_global_freq_table = {}
        self.total_count = 0
        
        # Shuffle batch order
        batch_indices = torch.randperm(len(data))
        
        # Process each batch
        for i in batch_indices:
            batch = data[i]
            if verbose:
                print(f"\nEpoch {epoch_num} - Batch {i + 1}:")
                print("Current batch content:")
                for tuple_data in batch:
                    print(f"  {tuple_data}")
                print("---")
            
            # Convert batch to tensor and process
            batch_tensor = torch.tensor(batch)
            self.process_batch(batch_tensor)
        
        # Get vocabulary for current epoch
        root_vocab = self.get_root_vocabulary()
        
        if verbose:
            print(f"\nFinal vocabulary for Epoch {epoch_num}:")
            for tuple_key, info in root_vocab.items():
                print(f"Tuple: {tuple_key}")
                print(f"Frequency: {info['global_frequency']:.3f}")
                print(f"Count: {info['global_count']}")
                print("---")
                
        return root_vocab

    def calculate_average_vocab(self, all_epoch_vocabs):
        """
        Calculate average frequencies for tuples across all epochs
        
        Args:
            all_epoch_vocabs: List of vocabularies from all epochs
        
        Returns:
            dict: Dictionary containing average statistics for each tuple
        """
        # Collect all unique tuples
        all_tuples = set()
        for vocab in all_epoch_vocabs:
            all_tuples.update(vocab.keys())
        
        # Calculate average frequencies across epochs
        average_vocab = {}
        num_epochs = len(all_epoch_vocabs)
        
        for tuple_key in all_tuples:
            frequencies = []
            total_count = 0
            for vocab in all_epoch_vocabs:
                if tuple_key in vocab:
                    frequencies.append(vocab[tuple_key]['global_frequency'])
                    total_count += vocab[tuple_key]['global_count']
            
            avg_frequency = sum(frequencies) / num_epochs
            avg_count = total_count / num_epochs
            
            average_vocab[tuple_key] = {
                'average_frequency': avg_frequency,
                'average_count': avg_count,
                'epoch_appearances': len(frequencies),
                'frequencies': frequencies
            }
        
        return average_vocab
        
    def train_for_epochs(self, data, num_epochs, verbose=True):
        """
        Train for multiple epochs
        
        Args:
            data: List of batches of data
            num_epochs: Number of epochs to train
            verbose: Whether to print detailed information
            
        Returns:
            dict: Average vocabulary across all epochs
        """
        # Store vocabularies for each epoch
        all_epoch_vocabs = []
        
        # Train for each epoch
        for epoch in range(num_epochs):
            # Train for a single epoch and get vocabulary
            epoch_vocab = self.train_single_epoch(data, epoch + 1, verbose)
            all_epoch_vocabs.append(epoch_vocab)
        
        # Calculate average vocabulary across all epochs
        average_vocab = self.calculate_average_vocab(all_epoch_vocabs)
        
        # Print average vocabulary information
        if verbose:
            print("\nAverage Vocabulary Information Across All Epochs:")
            for tuple_key, info in average_vocab.items():
                print(f"Tuple: {tuple_key}")
                print(f"Average Frequency: {info['average_frequency']:.3f}")
                print(f"Average Count: {info['average_count']:.1f}")
                print(f"Epoch Appearances: {info['epoch_appearances']}/{num_epochs}")
                print(f"Frequencies per Epoch: {[f'{freq:.3f}' for freq in info['frequencies']]}")
                print("---")
                
        return average_vocab

if __name__ == "__main__":
    # Create example data - each sublist is a batch
    data = [
        [(0, 0, 0, 0), (0, 0, 0, 0), (0, 20, 0, 20), (0, 20, 20, 0),
         (0, 20, 0, 20), (0, 20, 20, 20), (0, 20, 20, 20), (0, 0, 0, 10),
         (0, 0, 0, 30), (0, 0, 0, 0), (0, 20, 20, 20)],
        [(0, 0, 0, 0), (0, 20, 0, 20), (0, 20, 0, 20), (0, 20, 0, 20),
         (0, 0, 0, 30), (0, 0, 0, 30), (0, 0, 0, 10), (0, 0, 0, 50),
         (0, 0, 0, 50), (0, 0, 0, 60), (0, 0, 0, 70)]
    ]

    # Create RootFrequencyCounter instance
    counter = RootFrequencyCounter(root_entrance_freq=0.2, root_vocab_freq=0.15)
    num_epochs = 3
    
    # Train for multiple epochs
    average_vocab = counter.train_for_epochs(data, num_epochs)