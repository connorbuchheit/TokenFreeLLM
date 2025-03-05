import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
import math

class ByteNGramModel:
    def __init__(self, n: int = 8, smoothing: float = 0.1):
        """
        n: Context length (number of previous bytes to consider)
        smoothing: Laplace smoothing parameter
        """
        self.n = n
        self.smoothing = smoothing
        # Dictionary to store counts of n+1 grams (context + next byte)
        self.ngram_counts: Dict[bytes, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # Total counts for each context
        self.context_totals: Dict[bytes, int] = defaultdict(int)
        
    def train(self, texts: List[str]):
        """Train the model on a list of texts."""
        for text in texts:
            bytes_data = text.encode('utf-8')
            
            # Collect n-gram statistics
            for i in range(len(bytes_data) - self.n):
                context = bytes_data[i:i+self.n]
                next_byte = bytes_data[i+self.n]
                
                self.ngram_counts[context][next_byte] += 1
                self.context_totals[context] += 1
    
    def get_next_byte_distribution(self, context: bytes) -> np.ndarray:
        """
        Get probability distribution over next byte given context.
        Returns array of shape (256,) with probabilities.
        """
        if len(context) != self.n:
            raise ValueError(f"Context must be exactly {self.n} bytes long")
        
        # Get counts for this context
        byte_counts = self.ngram_counts[context]
        total = self.context_totals[context]
        
        # Calculate smoothed probabilities
        probs = np.zeros(256)
        for b in range(256):
            count = byte_counts[b]
            # Laplace smoothing
            probs[b] = (count + self.smoothing) / (total + 256 * self.smoothing)
            
        return probs
    
    def get_entropy(self, context: bytes) -> float:
        """Calculate Shannon entropy of next byte distribution given context."""
        probs = self.get_next_byte_distribution(context)
        # Only consider non-zero probabilities in entropy calculation
        nonzero_probs = probs[probs > 0]
        return -np.sum(nonzero_probs * np.log2(nonzero_probs))
    
    def predict_sequence_entropy(self, sequence: bytes, window: Optional[int] = None) -> List[float]:
        """
        Predict entropy for each position in a sequence.
        window: Optional sliding window size. If None, use all previous bytes as context.
        """
        entropies = []
        
        for i in range(self.n, len(sequence)):
            if window:
                start = max(i - window, 0)
                context = sequence[i-self.n:i]
            else:
                context = sequence[i-self.n:i]
            
            entropy = self.get_entropy(context)
            entropies.append(entropy)
            
        return entropies

if __name__ == "__main__":
    # Example usage
    model = ByteNGramModel(n=8)
    
    # Training data
    texts = [
        "Hello, this is some example text.",
        "More text to train the model.",
        "Even more training data here."
    ]
    
    # Train the model
    model.train(texts)
    
    # Test the model
    test_text = "Hello, how are you?"
    test_bytes = test_text.encode('utf-8')
    
    # Get entropies for each position
    entropies = model.predict_sequence_entropy(test_bytes)
    
    # Print results
    print("\nEntropy predictions:")
    for i, entropy in enumerate(entropies):
        context = test_bytes[i:i+8].decode('utf-8', errors='replace')
        print(f"Context: {context}, Entropy: {entropy:.4f}") 