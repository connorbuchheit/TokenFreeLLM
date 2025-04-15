import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional
import math

class ByteNGramModel:
    def __init__(self, n: int = 8, smoothing: float = 0.0001):
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
        print("Starting model training...") # Progress update: Training start
        total_texts = len(texts)
        for text_index, text in enumerate(texts):
            if (text_index + 1) % 1000 == 0: # Progress update every 1000 texts (adjust as needed)
                print(f"  Processed {text_index + 1}/{total_texts} texts for training...")
            bytes_data = text.encode('utf-8')

            # Collect n-gram statistics
            for i in range(len(bytes_data) - self.n):
                context = bytes_data[i:i+self.n]
                next_byte = bytes_data[i+self.n]

                self.ngram_counts[context][next_byte] += 1
                self.context_totals[context] += 1
        print("Model training complete.") # Progress update: Training end

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

        # Print probability distribution
        # if context == b'lo,':
        #     # Print context and probabilities in a more readable format
        #     print(f"\nContext: {context}")
        #     print("Byte  ASCII  Hex    Probability")
        #     print("-" * 35)
        #     for byte, prob in enumerate(probs):
        #         if prob > 0.001:  # Only show probabilities > 0.1%
        #             try:
        #                 ascii_char = chr(byte) if 32 <= byte <= 126 else '.'
        #             except ValueError:
        #                 ascii_char = '.'
        #             print(f"{byte:3d}  '{ascii_char}'    0x{byte:02x}   {prob:.6f}")

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
    window = 8
    model = ByteNGramModel(n=window)

    # Training data from parquet file
    parquet_file = 'data/a.parquet'  # Assuming 'a.parquet' is in the same directory
    num_texts_to_process = 20000 # Define the number of texts to process
    print(f"Loading text data from '{parquet_file}' (processing first {num_texts_to_process} texts)...") # Progress update: Start reading parquet
    try:
        df = pd.read_parquet(parquet_file)
        print(f"Successfully loaded parquet file '{parquet_file}'.") # Progress update: Parquet loaded

        # Assuming the parquet file has a column named 'text' containing the text data
        if 'text' in df.columns:
            texts = df['text'].tolist() # Extract text data into a list
        elif len(df.columns) > 0: # If 'text' column is not found, try to use the first column as text
            print(f"Warning: 'text' column not found in '{parquet_file}'. Using the first column '{df.columns[0]}' as text data.")
            texts = df[df.columns[0]].tolist()
        else:
            print(f"Error: No columns found in '{parquet_file}'. Please ensure the parquet file is not empty and contains text data.")
            texts = [] # Empty list if no text data is found
    except FileNotFoundError:
        print(f"Error: Parquet file '{parquet_file}' not found. Make sure the file is in the correct directory.")
        texts = []
    except Exception as e:
        print(f"Error reading parquet file '{parquet_file}': {e}")
        texts = []

    if texts: # Only train if we successfully loaded text data
        texts = texts[:num_texts_to_process] # Slice the list to take only the first num_texts_to_process
        print(f"Loaded {len(texts)} texts from '{parquet_file}'. Training on the first {len(texts)} texts.") # Progress update: Number of texts loaded
        # Train the model
        model.train(texts)

        # Test the model
        test_text = "Hello, how are you doing today?"
        test_bytes = test_text.encode('utf-8')

        # Get entropies for each position
        entropies = model.predict_sequence_entropy(test_bytes)

        # Print results
        print("\nEntropy predictions:")
        for i, entropy in enumerate(entropies):
            context = test_bytes[i:i+window].decode('utf-8', errors='replace')
            print(f"Context: {context}_ Entropy: {entropy:.4f}")
    else:
        print("No text data loaded. Exiting.")