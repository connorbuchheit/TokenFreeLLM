from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set matplotlib to use a non-interactive backend if running in a headless environment
matplotlib.use('Agg')

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("evabyte/EvaByte", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("evabyte/EvaByte", torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to("cuda")

def calculate_byte_entropy(prompt):
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

def calculate_entropy_for_each_position(prompt, batch_size=10):
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
        start = max(0, i-100)
        prefix = prompt[start:i]
        entropy = calculate_byte_entropy(prefix)
        entropies.append(entropy)
        if i % 10 == 0:
            print(f"Processed {i}/{total_positions} bytes")
    
    return entropies

def calculate_patches(entropies, entropy_threshold):
    """
    Calculate patches based on entropy threshold.
    A new patch starts at the byte following one that exceeds the threshold.
    
    Args:
        entropies (list): List of entropy values for each position
        entropy_threshold (float): Threshold to start a new patch
        
    Returns:
        list: List of patch lengths
    """
    patches = []
    current_patch_length = 1  # Start with the first character
    
    # Find patch boundaries and calculate patch lengths
    for i in range(len(entropies) - 1):  # Exclude the last position
        if entropies[i] > entropy_threshold:
            # Entropy exceeds threshold, end current patch and start a new one
            patches.append(current_patch_length)
            current_patch_length = 1
        else:
            # Entropy is below threshold, continue current patch
            current_patch_length += 1
    
    # Add the last patch if it exists
    if current_patch_length > 0:
        patches.append(current_patch_length)
    
    return patches

def plot_entropy_with_patches(prompt, entropies, entropy_threshold=0.6, save_path=None):
    """
    Plot the entropy trend with patch boundaries for a given prompt.
    
    Args:
        prompt (str): The input prompt
        entropies (list): List of entropy values for each position
        entropy_threshold (float): Threshold value for entropy
        save_path (str, optional): Path to save the plot image
    """
    positions = list(range(1, len(prompt) + 1))
    chars = list(prompt)
    
    # Create figure with appropriate size
    plt.figure(figsize=(max(12, len(prompt)/3), 8))
    
    # Plot entropy values
    plt.plot(positions, entropies, marker='o', linestyle='-', color='blue')
    
    # Add labels and title
    plt.xlabel('Byte')
    plt.ylabel('Entropy')
    plt.title('Next Byte Entropy Calculation')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to show ALL characters
    plt.xticks(positions, chars) 
    
    # For longer strings, add more space for labels
    # if len(prompt) > 50:
    #     plt.subplots_adjust(bottom=0.25)
    
    # Add horizontal line for entropy threshold
    plt.axhline(y=entropy_threshold, color='red', linestyle='--', alpha=0.8, 
                label=f'Entropy Threshold: {entropy_threshold} bits')
    
    # Find and plot patch boundaries (at the following byte after exceeding threshold)
    patch_boundaries = []
    for i in range(1, len(entropies)):
        if entropies[i-1] > entropy_threshold and i < len(positions):
            patch_boundaries.append(positions[i])
            plt.axvline(x=positions[i], color='green', linestyle='-', alpha=0.5)
    
    # Add patch boundaries to legend if there are any
    if patch_boundaries:
        plt.plot([], [], color='green', linestyle='-', alpha=0.5, 
                label=f'Patch Boundaries (n={len(patch_boundaries)})')
    
    plt.legend()
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return patch_boundaries

def analyze_threshold_vs_patch_length(entropies, threshold_start=0.2, threshold_end=1.0, threshold_step=0.1, save_path=None):
    """
    Analyze and plot how patch length changes with different entropy thresholds.
    
    Args:
        entropies (list): List of entropy values
        threshold_start (float): Starting threshold value
        threshold_end (float): Ending threshold value
        threshold_step (float): Step size for threshold values
        save_path (str, optional): Path to save the plot
        
    Returns:
        tuple: (thresholds, avg_patch_lengths, patch_counts)
    """
    thresholds = np.arange(threshold_start, threshold_end + threshold_step, threshold_step)
    avg_patch_lengths = []
    patch_counts = []
    
    # Calculate patch metrics for each threshold
    for threshold in thresholds:
        patches = calculate_patches(entropies, threshold)
        avg_length = sum(patches) / len(patches) if patches else 0
        avg_patch_lengths.append(avg_length)
        patch_counts.append(len(patches))
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot average patch length vs threshold
    ax1.plot(thresholds, avg_patch_lengths, marker='o', linestyle='-', color='blue')
    ax1.set_ylabel('Average Patch Length')
    ax1.set_title('Average Patch Length vs Entropy Threshold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot number of patches vs threshold
    ax2.plot(thresholds, patch_counts, marker='s', linestyle='-', color='green')
    ax2.set_xlabel('Entropy Threshold')
    ax2.set_ylabel('Number of Patches')
    ax2.set_title('Number of Patches vs Entropy Threshold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Patch analysis plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return thresholds, avg_patch_lengths, patch_counts

if __name__ == "__main__":
    # Read the sample text from an external file
    try:
        with open("sample_text.txt", "r") as f:
            prompt = f.read().strip()
        print(f"Loaded text from file (length: {len(prompt)} characters)")
    except FileNotFoundError:
        print("Sample text file not found. Using a short example text instead.")
        prompt = "This is a short example text to analyze entropy patterns."
    
    # Set default entropy threshold
    entropy_threshold = 0.6
    
    print(f"\nAnalyzing entropy and patches for text")
    
    # Calculate entropies (this is the most time-consuming step)
    print("Calculating next-byte entropies")
    entropies = calculate_entropy_for_each_position(prompt)
    
    # Print statistics
    avg_entropy = sum(entropies) / len(entropies)
    # print(f"Average entropy: {avg_entropy:.4f} bits")
    # print(f"Min entropy: {min(entropies):.4f} bits")
    # print(f"Max entropy: {max(entropies):.4f} bits")
    
    # Plot entropy with patch boundaries
    print("Plotting entropy and patch boundaries")
    patch_boundaries = plot_entropy_with_patches(
        prompt, entropies, 
        entropy_threshold=entropy_threshold,
        save_path="entropy_plot_with_patches.png"
    )
    
    # Analyze and plot patch lengths vs threshold
    print("Analyzing how average patch length changes with threshold")
    thresholds, avg_lengths, patch_counts = analyze_threshold_vs_patch_length(
        entropies,
        threshold_start=0.1, 
        threshold_end=3.0, 
        threshold_step=0.01,
        save_path="patch_analysis.png"
    )
    
    print("Analysis complete!") 