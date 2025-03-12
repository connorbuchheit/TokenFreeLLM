import os
import argparse
import logging
import torch
import numpy as np
from typing import List, Optional, Tuple
import json

from EntropyProjects.LMTransformer.transformer import LMTransformer, LMTransformerArgs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Calculate Shannon entropy from logits"""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-10)  # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_length)
    return entropy

def load_entropy_model(checkpoint_dir: str, device: str = "cpu") -> LMTransformer:
    """Load a trained entropy model from checkpoint directory"""
    # Load model parameters
    with open(os.path.join(checkpoint_dir, "params.json"), "r") as f:
        params = json.load(f)
    
    # Create model with the same parameters
    model_args = LMTransformerArgs(
        dim=params["model"]["dim"],
        n_layers=params["model"]["n_layers"],
        n_heads=params["model"]["n_heads"],
        max_seqlen=params["model"]["max_length"],
        ffn_dim_multiplier=params.get("model", {}).get("ffn_dim_multiplier", None),
        vocab_size=params["model"]["vocab_size"],
        attn_bias_type="causal",
        attn_impl="sdpa",
        sliding_window=None,
    )
    
    model = LMTransformer(model_args)
    
    # Load state dict
    state_dict_path = os.path.join(checkpoint_dir, "model.pt")
    model.load_state_dict(torch.load(state_dict_path, map_location=device), strict=False)
    
    model.to(device)
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def compute_entropy_for_text(
    model: LMTransformer, 
    text: str, 
    seq_length: int = 512, 
    stride: int = 256,
    device: str = "cpu"
) -> Tuple[List[float], List[str]]:
    """Compute entropy for each position in the text"""
    bytes_data = text.encode('utf-8')
    entropies = []
    contexts = []
    
    # Process text in overlapping windows
    for i in range(0, max(1, len(bytes_data) - seq_length), stride):
        end_idx = min(i + seq_length, len(bytes_data))
        sequence = bytes_data[i:end_idx]
        
        # Pad if necessary
        if len(sequence) < seq_length:
            sequence = sequence + bytes([0] * (seq_length - len(sequence)))
        
        # Convert to tensor
        input_tensor = torch.tensor([[b for b in sequence]], dtype=torch.long).to(device)
        
        # Get model predictions
        with torch.no_grad():
            logits = model(input_tensor, target=None, mask="causal")
        
        # Calculate entropy
        entropy = calculate_entropy(logits)
        
        # Store results for valid positions
        valid_len = min(len(sequence), seq_length)
        for j in range(valid_len):
            if i + j < len(bytes_data):
                entropies.append(entropy[0, j].item())
                # Get context (previous bytes)
                context_start = max(0, j - 10)
                context = sequence[context_start:j+1]
                try:
                    context_str = context.decode('utf-8', errors='replace')
                except:
                    context_str = str(context)
                contexts.append(context_str)
    
    return entropies, contexts

def main():
    parser = argparse.ArgumentParser(description='Calculate entropy using a trained transformer model')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to model checkpoint directory')
    parser.add_argument('--input_file', type=str, help='Path to input text file')
    parser.add_argument('--input_text', type=str, help='Text to analyze')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length for processing')
    parser.add_argument('--stride', type=int, default=256, help='Stride for overlapping windows')
    
    args = parser.parse_args()
    
    if not args.input_file and not args.input_text:
        parser.error("Either --input_file or --input_text must be provided")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model from {args.checkpoint_dir} to {device}")
    model = load_entropy_model(args.checkpoint_dir, device)
    
    # Get input text
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.input_text
    
    # Compute entropy
    logger.info(f"Computing entropy for text of length {len(text)}")
    entropies, contexts = compute_entropy_for_text(
        model, text, args.seq_length, args.stride, device
    )
    
    # Output results
    results = []
    for i, (entropy, context) in enumerate(zip(entropies, contexts)):
        results.append({
            "position": i,
            "entropy": entropy,
            "context": context,
            "char": text[i] if i < len(text) else ""
        })
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")
    else:
        # Print a sample of results
        print("\nEntropy results (sample):")
        print("Position | Char | Entropy | Context")
        print("-" * 60)
        for result in results[:20]:  # Show first 20 results
            print(f"{result['position']:8d} | {result['char']:4s} | {result['entropy']:7.4f} | {result['context']}")

if __name__ == "__main__":
    main() 