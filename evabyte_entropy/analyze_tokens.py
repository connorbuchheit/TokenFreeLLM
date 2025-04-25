from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tiktoken
import sentencepiece as spm
import os

# Set matplotlib to use a non-interactive backend if running in a headless environment
matplotlib.use('Agg')

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")


def calculate_next_byte_entropy(prompt):
    """
    Calculate entropy for the next byte prediction given an input prompt.
    
    Args:
        prompt (str): Input text prompt
        
    Returns:
        float: Entropy value in bits
    """
    # Tokenize input using standard HF tokenizer interface
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    # Generate position_ids
    position_ids = torch.arange(len(input_ids[0]), dtype=torch.long, device="cuda").unsqueeze(0)
    
    # Get logits for the next token
    with torch.no_grad():
        outputs = model(input_ids, position_ids=position_ids)
        
    # Model returns a tuple with logits as the first element
    next_byte_logits = outputs[0][0, -1, :]  # shape: [vocab_size]
    
    # Convert to probabilities with softmax
    next_byte_probs = torch.nn.functional.softmax(next_byte_logits, dim=0)
    
    # Calculate entropy - more efficient implementation
    # For numerical stability, we use log_softmax and directly compute entropy
    log_probs = torch.nn.functional.log_softmax(next_byte_logits, dim=0)
    entropy = -torch.sum(next_byte_probs * log_probs) / torch.log(torch.tensor(2.0))  # Convert to bits
    
    return entropy.item()

def calculate_entropy_for_each_position(prompt):
    """
    Calculate entropy for each position in the input string with batching for improved efficiency.
    
    Args:
        prompt (str): Input text prompt
        batch_size (int): Number of positions to process in each batch
    
    Returns:
        list: List of entropy values for each position
    """
    entropies = []
    total_positions = len(prompt)
    
    print(f"Processing {total_positions} bytes")
    
    for i in range(1, total_positions + 1):
        start = max(0, i-200)
        prefix = prompt[start:i]
        entropy = calculate_next_byte_entropy(prefix)
        entropies.append(entropy)
        if i % 10 == 0:
            print(f"Processed {i}/{total_positions} bytes")
    
    return entropies

def generate_tokens(prompt, method):
    tokens = []
    
    # if method == "tiktoken":
    enc = tiktoken.get_encoding("cl100k_base")  # You can change to another encoding if needed
    token_ids = enc.encode(prompt)
    for token_id in token_ids:
        tokens.append(enc.decode([token_id]))
    return tokens
    
    # if method == "sentencepiece":
    #     # Use a pretrained model from HuggingFace that uses SentencePiece
    #     sp_tokenizer = AutoTokenizer.from_pretrained(sp_model)
        
    #     # Get tokens
    #     encoded = sp_tokenizer.encode(prompt, add_special_tokens=False)
    #     for token_id in encoded:
    #         tokens.append(sp_tokenizer.decode([token_id]))
    #     return tokens


def patch_threshold(values, prompt, threshold):
    """
    Helper function to create tokens based on threshold values
    
    Args:
        values (list): List of values to check against threshold
        prompt (str): Input text prompt
        threshold (float): Threshold value for splitting tokens
        
    Returns:
        list: List of tokens
    """
    tokens = []
    current_token = ""
    
    for i, value in enumerate(values):
        if i < len(prompt):  # Ensure we don't go out of bounds
            current_token += prompt[i]
            if value > threshold:
                tokens.append(current_token)
                current_token = ""
    
    # Add the last token if not empty
    if current_token:
        tokens.append(current_token)
        
    return tokens

def generate_patches(method, prompt, entropies, threshold=0.6):
    tokens = []

    if method == "static":
        return patch_threshold(entropies, prompt, threshold)

    if method == "derivative":
        derivatives = [entropies[i] - entropies[i-1] for i in range(1, len(entropies))]
        return patch_threshold(derivatives, prompt[1:], threshold) 

def get_token_stats(tokens):
    """
    Returns statistics about tokens
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        dict: Dictionary with token statistics
    """
    return {
        "total_tokens": len(tokens),
        "unique_tokens": len(set(tokens)),
        "avg_token_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0
    }

def jaccard_similarity(tokens1, tokens2):
    """
    Calculate the Jaccard similarity between two lists of tokens.

    Args:
        tokens1 (list): First list of tokens.
        tokens2 (list): Second list of tokens.

    Returns:
        float: Jaccard similarity score.
    """
    set1 = set(tokens1)
    set2 = set(tokens2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 1.0  # Both sets are empty, consider them identical

    return len(intersection) / len(union)

if __name__ == "__main__":
    with open("sample_text.txt", "r") as f:
        prompt = f.read().strip()
    print(f"Loaded text")

    entropies = calculate_entropy_for_each_position(prompt)
    derivatives = [entropies[i] - entropies[i-1] for i in range(1, len(entropies))]
    entropy_chars = prompt
    derivative_chars = prompt[1:]

    # for i, entropy in enumerate(entropies):
    #     print(f"{prompt[i]}: {entropy}")

    # Test tiktoken tokenization
    tiktoken_tokens = generate_tokens(prompt, "tiktoken")
    print("\nTiktoken tokens:")
    # print(tiktoken_tokens)
    tiktoken_stats = get_token_stats(tiktoken_tokens)
    print(f"Total: {tiktoken_stats['total_tokens']}, Unique: {tiktoken_stats['unique_tokens']}, Avg length: {tiktoken_stats['avg_token_length']:.2f}")
    
    # Test sentencepiece tokenization with different models
    # sp_tokens = generate_tokens(prompt, "sentencepiece", "google/mt5-base")
    # print(f"\nSentencePiece tokens:")
    # print(sp_tokens)
    # sp_stats = get_token_stats(sp_tokens)
    # print(f"Total: {sp_stats['total_tokens']}, Unique: {sp_stats['unique_tokens']}, Avg length: {sp_stats['avg_token_length']:.2f}")


    # Test static tokenization with entropy threshold - NOW LOOPING
    # static_tokens = generate_patches("static", prompt, entropies, 0.6)
    # print("\nEntropy tokens:")
    # print(static_tokens)
    # static_stats = get_token_stats(static_tokens)
    # print(f"Total: {static_stats['total_tokens']}, Unique: {static_stats['unique_tokens']}, Avg length: {static_stats['avg_token_length']:.2f}")

    # Keep derivative tokenization outside loop (using a fixed threshold for now)
    derivative_threshold = 1.0 
    derivative_tokens = generate_patches("derivative", prompt, entropies, derivative_threshold)
    print(f"\nDerivative tokens (threshold={derivative_threshold}):")
    derivative_stats = get_token_stats(derivative_tokens)
    print(f"Total: {derivative_stats['total_tokens']}, Unique: {derivative_stats['unique_tokens']}, Avg length: {derivative_stats['avg_token_length']:.2f}")

    # Loop through different static thresholds
    static_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Example thresholds
    print("\n--- Testing Different Static Thresholds ---")
    for threshold in static_thresholds:
        static_tokens = generate_patches("static", prompt, entropies, threshold)
        derivative_tokens = generate_patches("derivative", prompt, entropies, threshold)
        static_stats = get_token_stats(static_tokens)
        derivative_stats = get_token_stats(derivative_tokens)
        print(f" Threshold: {threshold}")
        print(f"Static:\n  Total: {static_stats['total_tokens']}, Unique: {static_stats['unique_tokens']}, Avg length: {static_stats['avg_token_length']:.2f}")
        print()
        print(f"Derivative:\n  Total: {derivative_stats['total_tokens']}, Unique: {derivative_stats['unique_tokens']}, Avg length: {derivative_stats['avg_token_length']:.2f}")

        # Calculate and print Jaccard similarities for this threshold
        jaccard_tiktoken_static = jaccard_similarity(tiktoken_tokens, static_tokens)
        jaccard_tiktoken_derivative = jaccard_similarity(tiktoken_tokens, derivative_tokens)
        print(f"  Jaccard (Tiktoken vs Static): {jaccard_tiktoken_static:.4f}")
        print(f"  Jaccard (Static vs Derivative): {jaccard_tiktoken_derivative:.4f}")


    # # Create figure with appropriate size
    # plt.figure(figsize=(20, 8))
    
    # # Plot entropy values starting at prompt index 0
    # plt.plot(list(range(len(entropies))), entropies, marker='.', markersize=10, linestyle='-', color='blue', label='Entropy')
    
    # # Plot derivatives starting at prompt index 1
    # plt.plot(list(range(1, len(derivatives) + 1)), derivatives, marker='.', markersize=10, linestyle='-', color='red', label='Derivative')
    
    # # Add labels and title
    # plt.ylabel('Value')
    
    # plt.title('Entropy and Derivatives of Input Text')
    # plt.grid(True)
    
    # # Add legend
    # plt.legend(loc='upper right')
    
    # # Set x-axis ticks to show characters (limit to reasonable number if text is long)
    # max_chars_to_display = 50
    # if len(prompt) <= max_chars_to_display:
    #     plt.xticks(range(len(prompt)), list(prompt))
    # else:
    #     # Show a subset of characters if the text is too long
    #     step = len(prompt) // max_chars_to_display
    #     plt.xticks(range(0, len(prompt), step), [prompt[i] for i in range(0, len(prompt), step)])
    
    # # Save plot to file
    # plt.tight_layout()
    # plt.savefig('entropy_derivative_plot.png')
    # print("Plot saved as 'entropy_derivative_plot.png'")
    
    