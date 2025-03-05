import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional

class ByteLevelLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,  # Byte vocabulary
        d_model: int = 256, # Embedding dimension
        nhead: int = 4, # Number of heads in the multiheadattention models
        num_layers: int = 3, #  Number of sub-decoder-layers in the decoder
        dim_feedforward: int = 1024, # Dimension of the feedforward network model
        dropout: float = 0.1, # Dropout value
        max_seq_length: int = 512 # Maximum sequence length
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Create causal mask for autoregressive prediction
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_length, max_seq_length), diagonal=1).bool()
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.byte_predictor = nn.Linear(d_model, vocab_size)
        
    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # src shape: (batch_size, seq_length)
        x = self.embedding(src)
        x = self.pos_encoder(x)
        
        # Create causal mask for current sequence length
        seq_length = src.size(1)
        mask = self.causal_mask[:seq_length, :seq_length]
        
        # Use same sequence for encoder and decoder (like GPT)
        output = self.transformer(
            x,
            x,
            tgt_mask=mask,
            memory_mask=None,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        # Get logits for next byte prediction
        logits = self.byte_predictor(output)
        
        # Calculate entropy from probability distribution
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_length)
        
        return logits, entropy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ByteDataset(Dataset):
    def __init__(self, texts: List[str], seq_length: int = 512):
        self.seq_length = seq_length
        self.samples = []
        self.targets = []
        
        for text in texts:
            bytes_data = text.encode('utf-8')
            
            # Handle texts shorter than seq_length
            if len(bytes_data) < seq_length + 1:
                bytes_data = bytes_data + b'\0' * (seq_length + 1 - len(bytes_data))
            
            # Create sequences with shifted targets for next-byte prediction
            for i in range(0, len(bytes_data) - seq_length):
                sequence = bytes_data[i:i + seq_length]
                target = bytes_data[i + 1:i + seq_length + 1]  # Shifted by 1 for next-byte prediction
                
                self.samples.append(torch.tensor([b for b in sequence], dtype=torch.long))
                self.targets.append(torch.tensor([b for b in target], dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.targets[idx]

def train_model(
    model: ByteLevelLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_train_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            logits, entropy = model(data)
            
            # Reshape for CrossEntropyLoss
            loss = criterion(logits.view(-1, 256), target.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_train_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                print(f'Average entropy: {entropy.mean().item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, entropy = model(data)
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    train_model(model, train_loader, val_loader) 