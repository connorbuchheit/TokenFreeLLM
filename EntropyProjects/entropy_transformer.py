import torch
import torch.nn as nn
import math
from typing import Optional

class EntropyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,  # Assumes byte-level encoding
        d_model: int = 256, # Embedding dimension
        nhead: int = 4, # Number of heads in the multiheadattention models
        num_layers: int = 3, # Number of sub-encoder-layers in the encoder
        dim_feedforward: int = 1024, # Dimension of the feedforward network model
        dropout: float = 0.1, # Dropout value — change if need be
        max_seq_length: int = 512 # Maximum sequence length
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model) # Byte-level embedding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length) # Positional encoding
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Input and output tensors are provided as (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Modified output layer to predict probability distribution over bytes
        self.byte_predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, vocab_size),
            nn.LogSoftmax(dim=-1)  # Log probabilities for numerical stability
        )
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # src shape: (batch_size, seq_length)
        x = self.embedding(src)
        x = self.pos_encoder(x)
        
        # Transform the sequence
        transformer_output = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Get probability distribution for next byte
        log_probs = self.byte_predictor(transformer_output)
        probs = torch.exp(log_probs)
        
        # Calculate Shannon entropy: -sum of (p_i * log(p_i)) over all bytes i
        entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_length)
        
        return entropy  # Now only returning entropy values

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 