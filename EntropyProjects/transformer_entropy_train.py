import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import numpy as np

from EntropyProjects.LMTransformer.transformer import LMTransformer, LMTransformerArgs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ByteDataset(Dataset):
    def __init__(self, texts: List[str], seq_length: int = 512, pad_token: int = 255):
        self.seq_length = seq_length
        self.pad_token = pad_token
        self.samples = []
        self.targets = []
        self.padding_masks = []
        
        for text in texts:
            bytes_data = text.encode('utf-8')
            
            # Handle texts shorter than seq_length
            if len(bytes_data) < seq_length + 1:
                padding_length = seq_length + 1 - len(bytes_data)
                padding_mask = [False] * len(bytes_data) + [True] * padding_length
                bytes_data = bytes_data + bytes([pad_token] * padding_length)
            else:
                padding_mask = [False] * (seq_length + 1)
            
            # Create sequences with shifted targets for next-byte prediction
            for i in range(0, len(bytes_data) - seq_length):
                sequence = bytes_data[i:i + seq_length]
                target = bytes_data[i + 1:i + seq_length + 1]  # Shifted by 1 for next-byte prediction
                curr_padding_mask = padding_mask[i:i + seq_length]
                
                self.samples.append(torch.tensor([b for b in sequence], dtype=torch.long))
                self.targets.append(torch.tensor([b for b in target], dtype=torch.long))
                self.padding_masks.append(torch.tensor(curr_padding_mask, dtype=torch.bool))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.targets[idx], self.padding_masks[idx]

def calculate_entropy(logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Calculate Shannon entropy from logits"""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-10)  # Add small epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_length)
    
    if padding_mask is not None:
        entropy = entropy.masked_fill(padding_mask, 0.0)
    
    return entropy

def train_model(
    model: LMTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir: str = './checkpoints',
    log_interval: int = 100
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore padding tokens in loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_entropy = 0
        num_train_batches = 0
        num_train_tokens = 0
        
        for batch_idx, (data, target, padding_mask) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            padding_mask = padding_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(data, target=None, mask="causal")
            
            # Calculate loss
            loss = criterion(logits.view(-1, 256), target.view(-1))
            
            # Calculate entropy
            entropy = calculate_entropy(logits, padding_mask)
            valid_tokens = (~padding_mask).float().sum()
            batch_entropy = entropy.sum() / valid_tokens
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_entropy += batch_entropy.item() * valid_tokens.item()
            num_train_batches += 1
            num_train_tokens += valid_tokens.item()
            
            if batch_idx % log_interval == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Entropy: {batch_entropy.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_entropy = 0
        num_val_batches = 0
        num_val_tokens = 0
        
        with torch.no_grad():
            for data, target, padding_mask in val_loader:
                data = data.to(device)
                target = target.to(device)
                padding_mask = padding_mask.to(device)
                
                logits = model(data, target=None, mask="causal")
                loss = criterion(logits.view(-1, 256), target.view(-1))
                
                entropy = calculate_entropy(logits, padding_mask)
                valid_tokens = (~padding_mask).float().sum()
                
                val_loss += loss.item()
                val_entropy += entropy.sum().item()
                num_val_batches += 1
                num_val_tokens += valid_tokens.item()
        
        avg_train_loss = train_loss / num_train_batches
        avg_train_entropy = train_entropy / num_train_tokens
        avg_val_loss = val_loss / num_val_batches
        avg_val_entropy = val_entropy / num_val_tokens
        
        logger.info(f'Epoch {epoch}:')
        logger.info(f'  Train loss: {avg_train_loss:.4f}, Train entropy: {avg_train_entropy:.4f}')
        logger.info(f'  Val loss: {avg_val_loss:.4f}, Val entropy: {avg_val_entropy:.4f}')
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'model_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_entropy': avg_val_entropy,
            }, checkpoint_path)
            logger.info(f'Saved best model checkpoint to {checkpoint_path}')
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_entropy': avg_val_entropy,
        }, checkpoint_path)
        
        # Update learning rate
        scheduler.step()

def load_data(data_path: str) -> List[str]:
    """Load text data from a file or directory"""
    texts = []
    
    if os.path.isfile(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    elif os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            if filename.endswith('.txt'):
                with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    
    return texts

def main():
    parser = argparse.ArgumentParser(description='Train a transformer model for byte entropy prediction')
    parser.add_argument('--data', type=str, required=True, help='Path to training data file or directory')
    parser.add_argument('--val_data', type=str, help='Path to validation data (if not provided, will use a portion of training data)')
    parser.add_argument('--model_dim', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--seq_length', type=int, default=512, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    train_texts = load_data(args.data)
    logger.info(f'Loaded {len(train_texts)} training texts')
    
    if args.val_data:
        val_texts = load_data(args.val_data)
        logger.info(f'Loaded {len(val_texts)} validation texts')
    else:
        # Use 10% of training data for validation
        split_idx = int(0.9 * len(train_texts))
        val_texts = train_texts[split_idx:]
        train_texts = train_texts[:split_idx]
        logger.info(f'Split data into {len(train_texts)} training and {len(val_texts)} validation texts')
    
    # Create datasets and dataloaders
    train_dataset = ByteDataset(train_texts, seq_length=args.seq_length)
    val_dataset = ByteDataset(val_texts, seq_length=args.seq_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    model_args = LMTransformerArgs(
        dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=256,  # Byte vocabulary
        max_seqlen=args.seq_length,
        seed=args.seed,
    )
    
    model = LMTransformer(model_args)
    model.init_weights()
    
    logger.info(f'Created model with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )

if __name__ == "__main__":
    main() 