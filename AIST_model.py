import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from utils import *
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModule(nn.Module):
    def __init__(self, input_dim, seq_len, model_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        """
        Transformer module with embedding, positional encoding, and encoder layers.
        """
        super(TransformerModule, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for Transformer module.
        """
        # Project input to model dimension
        x = self.input_projection(x)  # [batch_size, seq_len, model_dim]
        # Add positional encoding
        x = x + self.positional_encoding  # [batch_size, seq_len, model_dim]
        # Pass through Transformer encoder
        x = self.dropout(self.transformer_encoder(x))  # [batch_size, seq_len, model_dim]
        return x


class AISTModelWithTransformer(nn.Module):
    def __init__(self, x_crime_seq_len, x_crime_input_dim, x_daily_input_dim, 
                 x_weekly_input_dim, transformer_dim=64, num_heads=4, ff_dim=128, 
                 num_layers=2, dropout=0.1):
        """
        AIST Model with Transformer for crime prediction.
        """
        super(AISTModelWithTransformer, self).__init__()
        
        # Transformer for crime data
        self.crime_transformer = TransformerModule(
            input_dim=x_crime_input_dim,
            seq_len=x_crime_seq_len,
            model_dim=transformer_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Daily and weekly feature MLPs
        self.daily_mlp = nn.Linear(x_daily_input_dim, transformer_dim)
        self.weekly_mlp = nn.Linear(x_weekly_input_dim, transformer_dim)
        
        # Combine outputs for final prediction
        self.fc1 = nn.Linear(3 * transformer_dim, 128)  # Combine crime, daily, weekly
        self.fc2 = nn.Linear(128, 1)  # Final prediction

    def forward(self, x_crime, x_crime_daily, x_crime_weekly):
        """
        Forward pass for AIST model with Transformer.
        """
        # Reshape and pass crime data through Transformer
        batch_size = x_crime.size(0)
        seq_len = int(x_crime.size(1) / 12)  # Assuming input_dim = 12
        x_crime = x_crime.view(batch_size, seq_len, 12)  # [batch_size, seq_len, input_dim]
        crime_features = self.crime_transformer(x_crime)  # [batch_size, seq_len, transformer_dim]
        crime_features = crime_features.mean(dim=1)  # Pooling to [batch_size, transformer_dim]
        
        # Pass daily and weekly data through MLPs
        daily_features = F.relu(self.daily_mlp(x_crime_daily))  # [batch_size, transformer_dim]
        weekly_features = F.relu(self.weekly_mlp(x_crime_weekly))  # [batch_size, transformer_dim]
        
        # Concatenate features
        combined_features = torch.cat([crime_features, daily_features, weekly_features], dim=-1)
        
        # Pass through final fully connected layers
        x = F.relu(self.fc1(combined_features))
        output = self.fc2(x)  # [batch_size, 1]
        
        return output


# Example Input Shapes
batch_size = 42
x_crime_seq_len = 10  # Sequence length derived from [120 / input_dim]
x_crime_input_dim = 12  # Derived from splitting x_crime (120 features) into 10 sequences
x_daily_input_dim = 20
x_weekly_input_dim = 3

# Instantiate the model
model = AISTModelWithTransformer(
    x_crime_seq_len=x_crime_seq_len,
    x_crime_input_dim=x_crime_input_dim,
    x_daily_input_dim=x_daily_input_dim,
    x_weekly_input_dim=x_weekly_input_dim
)

# Example Inputs
x_crime = torch.rand(batch_size, 120)  # Crime data [batch_size, total_features]
x_crime_daily = torch.rand(batch_size, 20)  # Daily data [batch_size, features]
x_crime_weekly = torch.rand(batch_size, 3)  # Weekly data [batch_size, features]

# Forward Pass
output = model(x_crime, x_crime_daily, x_crime_weekly)

# Output
print("Output shape:", output.shape)  # Expected: [batch_size, 1]
