from collections import defaultdict

class FrequencyCounter:
    def __init__(self, min_freq=0):
        """Initialize FrequencyCounter with an optional min_freq parameter."""
        self.min_freq = min_freq
        self.global_freq_table = defaultdict(lambda: {"total_count": 0, "global_frequency": 0.0})
        self.total_batch_size = 0
    
    def initialize_local_freq_table(self):
        """Step 1: Initialize local_freq_table."""
        return defaultdict(lambda: {"count": 0, "frequency": 0.0})

    def count_frequencies_in_batch(self, batch, local_freq_table):
        """Step 2: Count frequency of each item in the current batch."""
        for item in batch:
            local_freq_table[item]["count"] += 1
        return local_freq_table

    def calculate_local_frequencies(self, local_freq_table, batch_size):
        """Step 3: Calculate frequency for each item in the batch."""
        for item in local_freq_table:
            local_freq_table[item]["frequency"] = local_freq_table[item]["count"] / batch_size
        return local_freq_table

    def update_global_freq_table(self, local_freq_table, batch_size):
        """Step 4: Update global_freq_table based on the local_freq_table."""
        for item, info in local_freq_table.items():
            frequency = info["frequency"]
            
            # Only process items whose frequency is greater than or equal to min_freq
            if item in self.global_freq_table:
                self.global_freq_table[item]["total_count"] += info["count"]
                self.global_freq_table[item]["global_frequency"] = (
                    self.global_freq_table[item]["total_count"] / (self.total_batch_size + batch_size)
                )
            else:
                if frequency >= self.min_freq:
                    self.global_freq_table[item] = {
                        "total_count": info["count"],
                        "global_frequency": info["count"] / (self.total_batch_size + batch_size)
                    }

    def update_freq_tables(self, batch):
        """Update both local and global frequency tables."""
        # Initialize the local frequency table for the current batch
        local_freq_table = self.initialize_local_freq_table()
        
        # Count frequencies in the current batch
        local_freq_table = self.count_frequencies_in_batch(batch, local_freq_table)
        
        # Calculate frequency for each item in the current batch
        local_freq_table = self.calculate_local_frequencies(local_freq_table, len(batch))
        
        # Update the global frequency table based on the local frequency tab
        # le
        self.update_global_freq_table(local_freq_table, len(batch))
        
        # Update the total_batch_size to reflect the total number of processed items
        self.total_batch_size += len(batch)
        
        return local_freq_table, self.global_freq_table
