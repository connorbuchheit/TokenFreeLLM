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
        # if i % 10 == 0:
        #     print(f"Processed {i}/{total_positions} bytes")
    
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

if __name__ == "__main__":
    with open("sample_text.txt", "r") as f:
        prompt = f.read().strip()
    print(f"Loaded text")

    entropies = calculate_entropy_for_each_position(prompt)
    # for i, entropy in enumerate(entropies):
    #     print(f"{prompt[i]}: {entropy}")

    # Test tiktoken tokenization
    tiktoken_tokens = generate_tokens(prompt, "tiktoken")
    print("\nTiktoken tokens:")
    print(tiktoken_tokens)
    tiktoken_stats = get_token_stats(tiktoken_tokens)
    print(f"Total: {tiktoken_stats['total_tokens']}, Unique: {tiktoken_stats['unique_tokens']}, Avg length: {tiktoken_stats['avg_token_length']:.2f}")
    
    # Test sentencepiece tokenization with different models
    # sp_tokens = generate_tokens(prompt, "sentencepiece", "google/mt5-base")
    # print(f"\nSentencePiece tokens:")
    # print(sp_tokens)
    # sp_stats = get_token_stats(sp_tokens)
    # print(f"Total: {sp_stats['total_tokens']}, Unique: {sp_stats['unique_tokens']}, Avg length: {sp_stats['avg_token_length']:.2f}")


    # Test static tokenization with entropy threshold
    static_tokens = generate_patches("static", prompt, entropies, 0.6)
    print("\nEntropy tokens:")
    print(static_tokens)
    static_stats = get_token_stats(static_tokens)
    print(f"Total: {static_stats['total_tokens']}, Unique: {static_stats['unique_tokens']}, Avg length: {static_stats['avg_token_length']:.2f}")

    # Test derivative tokenization
    derivative_tokens = generate_patches("derivative", prompt, entropies, 1.0)
    print("\nDerivative tokens:")
    print(derivative_tokens)
    derivative_stats = get_token_stats(derivative_tokens)
    print(f"Total: {derivative_stats['total_tokens']}, Unique: {derivative_stats['unique_tokens']}, Avg length: {derivative_stats['avg_token_length']:.2f}")



    