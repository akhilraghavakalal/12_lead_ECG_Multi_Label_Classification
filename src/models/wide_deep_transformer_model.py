import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=750):  # Reduced max_len due to downsampling
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class WideDeepTransformer(nn.Module):
    def __init__(
        self,
        n_classes=27,
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        dropout=0.1,
        wide_dim=20,
    ):
        super().__init__()

        # Matching paper's CNN architecture exactly
        self.conv_layers = nn.Sequential(
            # First layer: 12 -> 128
            nn.Conv1d(12, 128, kernel_size=14, stride=3, padding=2),
            nn.ReLU(),
            # Second layer: 128 -> 256
            nn.Conv1d(128, 256, kernel_size=14, stride=3),
            nn.ReLU(),
            # Third layer: 256 -> 256
            nn.Conv1d(256, 256, kernel_size=10, stride=2),
            nn.ReLU(),
            # Fourth layer: 256 -> 256
            nn.Conv1d(256, 256, kernel_size=10, stride=2),
            nn.ReLU(),
            # Fifth layer: 256 -> 256
            nn.Conv1d(256, 256, kernel_size=10, stride=1),
            nn.ReLU(),
            # Sixth layer: 256 -> d_model
            nn.Conv1d(256, d_model, kernel_size=10, stride=1),
            nn.ReLU(),
        )

        self.pos_encoder = PositionalEncoding(d_model)

        # Using the paper's transformer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Wide network component
        self.wide_fc = nn.Linear(wide_dim, 64)

        # Final classification layers
        self.combine = nn.Sequential(
            nn.Linear(d_model + 64, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
            nn.Sigmoid(),
        )

    def forward(self, ecg, wide_features):
        # Process wide features
        wide = self.wide_fc(wide_features)

        # CNN feature extraction (approximately 20x downsampling as mentioned in paper)
        x = self.conv_layers(ecg)

        # Prepare for transformer
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Combine with wide features and classify
        combined = torch.cat([x, wide], dim=1)
        output = self.combine(combined)

        return output


def create_model(wide_dim, n_classes, sequence_length=7500, d_model=256):
    """
    Factory function to create an instance of WideDeepTransformer.

    Args:
        wide_dim (int): Dimension of wide (static) features
        n_classes (int): Number of output classes
        sequence_length (int): Length of input sequence
        d_model (int): Dimension of the model

    Returns:
        WideDeepTransformer: Instantiated model
    """
    model = WideDeepTransformer(
        n_classes=n_classes,
        d_model=d_model,
        nhead=8,
        num_layers=8,
        dim_feedforward=2048,
        dropout=0.1,
        wide_dim=wide_dim,
    )

    return model
