import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from byte_lm import ByteLevelLM, ByteDataset

def train_model(
    model: ByteLevelLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token)  # Ignore padding tokens in loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_train_batches = 0
        
        for batch_idx, (data, target, padding_mask) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            padding_mask = padding_mask.to(device)
            
            optimizer.zero_grad()
            logits, entropy = model(data, src_key_padding_mask=padding_mask)
            
            # Reshape for CrossEntropyLoss
            loss = criterion(logits.view(-1, 256), target.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                # Calculate average entropy excluding padding tokens
                valid_entropy = entropy.masked_fill(padding_mask, 0.0)
                valid_tokens = (~padding_mask).float().sum()
                avg_entropy = valid_entropy.sum() / valid_tokens
                print(f'Average entropy (non-padding): {avg_entropy.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for data, target, padding_mask in val_loader:
                data = data.to(device)
                target = target.to(device)
                padding_mask = padding_mask.to(device)
                
                logits, entropy = model(data, src_key_padding_mask=padding_mask)
                loss = criterion(logits.view(-1, 256), target.view(-1))
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_train_loss = train_loss / num_train_batches
        avg_val_loss = val_loss / num_val_batches
        
        print(f'Epoch {epoch}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')

if __name__ == "__main__":
    model = ByteLevelLM()
    
    # Create sample data (replace with your actual dataset)
    sample_texts = ["Your training text here", "More training text"]
    
    train_dataset = ByteDataset(sample_texts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
    )
    
    train_model(model, train_loader, val_loader) 