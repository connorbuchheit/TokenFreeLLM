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

# --- Global Matplotlib Settings ---
fontsize = 14
font_settings = {
    'font.size': fontsize,
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': fontsize,
}
matplotlib.rcParams.update(font_settings)
# ----------------------------------

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

def strip_outer_spaces(tokens):
    """
    Removes leading and trailing whitespace from each token in a list.

    Args:
        tokens (list): A list of string tokens.

    Returns:
        list: A new list with processed tokens.
    """
    return [token.strip() for token in tokens if token not in ["", " "]]

def add_first_character(tokens, prompt):
    """
    Adds the first character of the prompt to the beginning of each token in the list.
    """
    return [prompt[0] + tokens[0]] + tokens[1:]

def create_entropy_threshold_plot(prompt, entropies, derivatives, static_threshold=0.6, derivative_threshold=0.6, filename="entropy_derivative_plot.png"):
    """
    Generates and saves a plot with two subplots: Entropy (top) and Derivative (bottom).
    Adds vertical lines based on respective thresholds for each subplot.
    
    Args:
        prompt (str): The input text prompt.
        entropies (list): List of entropy values corresponding to each character.
        derivatives (list): List of derivative values (entropy[i] - entropy[i-1]).
        static_threshold (float): Entropy threshold for drawing vertical lines on the top plot.
        derivative_threshold (float): Derivative threshold for drawing vertical lines on the bottom plot.
        filename (str): The name of the file to save the plot to.
    """
    # Create figure with two subplots, sharing the x-axis
    fig, ax = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    
    # --- Top Subplot: Entropy and Static Threshold Lines ---
    # Add vertical lines based on static threshold
    for i, entropy in enumerate(entropies):
        if entropy > static_threshold:
            # Add line between character i and i+1, extend below axis, disable clipping
            ax[0].axvline(i + 0.5, color='gray', linestyle='-', linewidth=2.5, alpha=0.8, ymin=-0.15, ymax=1, clip_on=False) 
            
    ax[0].plot(list(range(len(entropies))), entropies, marker='.', markersize=10, linestyle='-', color='blue', label='Entropy')
    ax[0].axhline(static_threshold, color='orange', linestyle=':', linewidth=1.5, label=f'Threshold ({static_threshold})')
    ax[0].set_ylabel('Entropy')
    ax[0].set_title(f'Next Byte Entropy - Static and Difference Thresholds')
    ax[0].grid(True)
    ax[0].legend(loc='upper right')
    ax[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    # --- Bottom Subplot: Derivative and Derivative Threshold Lines ---
    # Add vertical lines based on derivative threshold
    if derivatives: # Check if derivatives list is not empty
        for i, deriv in enumerate(derivatives):
             # derivative[i] corresponds to change between prompt[i] and prompt[i+1]
             # So the line goes after prompt[i+1], at index i+1 + 0.5
            if deriv > derivative_threshold:
                 # Add line between character i+1 and i+2, extend below axis, disable clipping
                ax[1].axvline(i + 1 + 0.5, color='dimgray', linestyle='-', linewidth=2.5, alpha=0.8, ymin=-0.15, ymax=1, clip_on=False) 
                
        ax[1].plot(list(range(1, len(derivatives) + 1)), derivatives, marker='.', markersize=10, linestyle='-', color='red', label='Entropy Difference')
        
    ax[1].axhline(derivative_threshold, color='purple', linestyle=':', linewidth=1.5, label=f'Threshold ({derivative_threshold})') # Use derivative threshold
    ax[1].set_ylabel('Entropy Difference')
    ax[1].grid(True)
    ax[1].legend(loc='upper right')

    # Set x-axis ticks and labels (applied to both due to sharex=True)
    max_chars_to_display = 50
    if len(prompt) <= max_chars_to_display:
        tick_indices = list(range(len(prompt)))
        tick_labels = [f"{prompt[i]}" for i in tick_indices]
        # Apply to bottom axis, which controls the shared axis
        ax[1].set_xticks(tick_indices)
        ax[1].set_xticklabels(tick_labels)
    else:
        step = len(prompt) // max_chars_to_display
        tick_indices = list(range(0, len(prompt), step))
        tick_labels = [f"{prompt[i]}" for i in tick_indices]
        # Apply to bottom axis, which controls the shared axis
        ax[1].set_xticks(tick_indices)
        ax[1].set_xticklabels(tick_labels)
        
    # Ensure top x-axis labels are visible if automatically hidden by sharex
    plt.setp(ax[0].get_xticklabels(), visible=True)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Add slight top margin for overall title if needed
    # fig.suptitle('Entropy Analysis', fontsize=16) # Optional overall title
    
    # Save plot to file
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")

def create_jaccard_similarity_plot(thresholds, jaccard_static, jaccard_derivative, filename="jaccard_similarity_plot.png"):
    """
    Generates and saves a plot of Jaccard similarity scores against thresholds.
    
    Args:
        thresholds (list): List of threshold values used.
        jaccard_static (list): List of Jaccard scores (tiktoken vs static) corresponding to thresholds.
        jaccard_derivative (list): List of Jaccard scores (tiktoken vs derivative) corresponding to thresholds.
        filename (str): The name of the file to save the plot to.
    """
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(thresholds, jaccard_static, linewidth=2, label='Tiktoken vs Static')
    plt.plot(thresholds, jaccard_derivative, linewidth=2, label='Tiktoken vs Difference')
    
    # Labels and title will use the global font settings
    plt.xlabel("Threshold") 
    plt.ylabel("Jaccard Similarity Score")
    plt.title("Jaccard Similarity - Tiktoken vs. Entropy Thresholds")
    plt.grid(True)
    plt.legend()
    # No need to set fontsize individually here anymore
    
    plt.tight_layout()
    plt.savefig(filename)
        
    print(f"Jaccard similarity plot saved as '{filename}'")

if __name__ == "__main__":
    with open("sample_text.txt", "r") as f:
        prompt = f.read().strip()
    print(f"Loaded text ({len(prompt)} bytes)")

    entropy_cache_file = "entropies_bee_movie.npy"
    if os.path.exists(entropy_cache_file):
        print(f"Loading entropies from file: {entropy_cache_file}")
        entropies = np.load(entropy_cache_file).tolist() # Load as numpy array and convert back to list
    else:
        print(f"Calculating entropies")
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")
        entropies = calculate_entropy_for_each_position(prompt)
        print(f"Saving calculated entropies to file: {entropy_cache_file}")
        np.save(entropy_cache_file, np.array(entropies))


    # Calculate derivatives based on loaded/calculated entropies
    derivatives = [entropies[i] - entropies[i-1] for i in range(1, len(entropies))]
    entropy_chars = prompt
    derivative_chars = prompt[1:]

    # for i, entropy in enumerate(entropies):
    #     print(f"{prompt[i]}: {entropy}")

    # Test tiktoken tokenization
    tiktoken_tokens = strip_outer_spaces(generate_tokens(prompt, "tiktoken"))
    # print(f"Tiktoken tokens: {tiktoken_tokens}")
    # print("\nTiktoken tokens:")
    # # print(tiktoken_tokens)
    # tiktoken_stats = get_token_stats(tiktoken_tokens)
    # print(f"Total: {tiktoken_stats['total_tokens']}, Unique: {tiktoken_stats['unique_tokens']}, Avg length: {tiktoken_stats['avg_token_length']:.2f}")
    
    # Loop through different static thresholds
    thresholds = np.linspace(0.01, 5.0, 1000) # Example thresholds
    # print("\n--- Testing Different Thresholds ---")
    
    # Lists to store results for plotting
    jaccard_static_scores = []
    jaccard_derivative_scores = []
    
    for threshold in thresholds:
        static_tokens = strip_outer_spaces(generate_patches("static", prompt, entropies, threshold))
        derivative_tokens = add_first_character(strip_outer_spaces(generate_patches("derivative", prompt, entropies, threshold)), prompt)
        static_stats = get_token_stats(static_tokens)
        derivative_stats = get_token_stats(derivative_tokens)
        # print(f" Threshold: {threshold:.2f}")
        # print(f"Static:\n  Total: {static_stats['total_tokens']}, Unique: {static_stats['unique_tokens']}, Avg length: {static_stats['avg_token_length']:.2f}")
        # print(f"Derivative:\n  Total: {derivative_stats['total_tokens']}, Unique: {derivative_stats['unique_tokens']}, Avg length: {derivative_stats['avg_token_length']:.2f}")
        # print("-")

        # Calculate and print Jaccard similarities for this threshold
        jaccard_tiktoken_static = jaccard_similarity(tiktoken_tokens, static_tokens)
        jaccard_tiktoken_derivative = jaccard_similarity(tiktoken_tokens, derivative_tokens)
        # print(f"Jaccard (tiktoken, static): {jaccard_tiktoken_static:.2f}")
        # print(f"Jaccard (tiktoken, derivative): {jaccard_tiktoken_derivative:.2f}")

        # Append scores for plotting
        jaccard_static_scores.append(jaccard_tiktoken_static)
        jaccard_derivative_scores.append(jaccard_tiktoken_derivative)

    # Create the entropy plot
    create_entropy_threshold_plot(prompt, entropies, derivatives)
    
    # Create the Jaccard similarity plot
    create_jaccard_similarity_plot(thresholds, jaccard_static_scores, jaccard_derivative_scores)
    
    