import torch
import math
from collections import Counter


class CompressionMetrics:
    """
    Compute compression metrics for encoded sequences.
    """
    
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: The trained tokenizer with vocabulary
        """
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
    
    def compute_compression_ratio(self, original_data, encoded_sequence):
        """
        Compute compression ratio for a single sample.
        
        Args:
            original_data: Original image tensor (e.g., [1, 28, 28])
            encoded_sequence: Encoded 1D sequence (list of codewords)
        
        Returns:
            dict with compression metrics
        """
        # Original size in bits (assuming uint8)
        original_bits = original_data.numel() * 8
        
        # Encoded size - need bits to represent vocab
        vocab_size = len(self.vocab)
        bits_per_token = math.ceil(math.log2(vocab_size)) if vocab_size > 0 else 1
        encoded_bits = len(encoded_sequence) * bits_per_token
        
        # Compression ratio
        compression_ratio = original_bits / encoded_bits if encoded_bits > 0 else 0
        
        # Space savings percentage
        space_savings = (1 - encoded_bits / original_bits) * 100 if original_bits > 0 else 0
        
        return {
            'original_bits': original_bits,
            'encoded_bits': encoded_bits,
            'bits_per_token': bits_per_token,
            'vocab_size': vocab_size,
            'sequence_length': len(encoded_sequence),
            'compression_ratio': compression_ratio,
            'space_savings_pct': space_savings
        }
    
    def compute_entropy_based_compression(self, encoded_sequence):
        """
        Compute theoretical compression using entropy (accounts for token frequency).
        
        Args:
            encoded_sequence: Encoded 1D sequence (list of codewords)
        
        Returns:
            dict with entropy-based metrics
        """
        if not encoded_sequence:
            return {'entropy': 0, 'theoretical_bits': 0}
        
        # Count token frequencies
        token_counts = Counter(encoded_sequence)
        total_tokens = len(encoded_sequence)
        
        # Calculate entropy
        entropy = 0
        for count in token_counts.values():
            prob = count / total_tokens
            entropy -= prob * math.log2(prob)
        
        # Theoretical bits needed with optimal encoding
        theoretical_bits = entropy * total_tokens
        
        return {
            'entropy': entropy,
            'theoretical_bits': theoretical_bits,
            'unique_tokens': len(token_counts),
            'token_diversity': len(token_counts) / len(self.vocab.codewords) if len(self.vocab.codewords) > 0 else 0
        }
    
    def compute_batch_metrics(self, original_batch, encoded_batch):
        """
        Compute compression metrics for a batch of samples.
        
        Args:
            original_batch: Batch of original images [B, C, H, W] or [B, H, W]
            encoded_batch: List of encoded sequences
        
        Returns:
            dict with aggregated metrics
        """
        batch_metrics = []
        
        for i in range(len(encoded_batch)):
            original = original_batch[i:i+1]
            encoded = encoded_batch[i]
            
            basic_metrics = self.compute_compression_ratio(original, encoded)
            entropy_metrics = self.compute_entropy_based_compression(encoded)
            
            combined = {**basic_metrics, **entropy_metrics}
            batch_metrics.append(combined)
        
        # Aggregate statistics
        avg_metrics = {
            'avg_compression_ratio': sum(m['compression_ratio'] for m in batch_metrics) / len(batch_metrics),
            'avg_space_savings_pct': sum(m['space_savings_pct'] for m in batch_metrics) / len(batch_metrics),
            'avg_sequence_length': sum(m['sequence_length'] for m in batch_metrics) / len(batch_metrics),
            'avg_entropy': sum(m['entropy'] for m in batch_metrics) / len(batch_metrics),
            'min_compression_ratio': min(m['compression_ratio'] for m in batch_metrics),
            'max_compression_ratio': max(m['compression_ratio'] for m in batch_metrics),
            'total_original_bits': sum(m['original_bits'] for m in batch_metrics),
            'total_encoded_bits': sum(m['encoded_bits'] for m in batch_metrics),
        }
        
        return {
            'individual_metrics': batch_metrics,
            'aggregate_metrics': avg_metrics
        }
    
    def print_metrics(self, metrics, sample_idx=None):
        """
        Pretty print compression metrics.
        """
        prefix = f"Sample {sample_idx}: " if sample_idx is not None else ""
        
        print(f"\n{prefix}Compression Metrics:")
        print(f"  Original size: {metrics['original_bits']} bits")
        print(f"  Encoded size: {metrics['encoded_bits']} bits")
        print(f"  Sequence length: {metrics['sequence_length']} tokens")
        print(f"  Vocab size: {metrics['vocab_size']}")
        print(f"  Bits per token: {metrics['bits_per_token']}")
        print(f"  Compression ratio: {metrics['compression_ratio']:.3f}x")
        print(f"  Space savings: {metrics['space_savings_pct']:.2f}%")
        
        if 'entropy' in metrics:
            print(f"  Entropy: {metrics['entropy']:.3f} bits/token")
            print(f"  Theoretical bits: {metrics['theoretical_bits']:.1f}")
            print(f"  Unique tokens: {metrics['unique_tokens']}")
            print(f"  Token diversity: {metrics['token_diversity']:.3f}")
