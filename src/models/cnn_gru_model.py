import torch
import torch.nn as nn

class CNNGRUModel(nn.Module):
    def __init__(self, input_channels=12, n_classes=27, wide_dim=20):
        super().__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=14, stride=3, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 256, kernel_size=14, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Conv1d(256, 256, kernel_size=10, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Conv1d(256, 256, kernel_size=10, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Wide features processing
        self.wide_fc = nn.Linear(wide_dim, 64)
        
        # Combine features
        self.combine = nn.Sequential(
            nn.Linear(512 + 64, 256),  # 512 from bidirectional GRU (256*2) + 64 from wide
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )
        
    def forward(self, ecg, wide_features):
        # CNN feature extraction
        x = self.cnn(ecg)  # (batch, 256, seq_len)
        
        # Prepare for GRU
        x = x.transpose(1, 2)  # (batch, seq_len, 256)
        
        # GRU processing
        output, _ = self.gru(x)  # (batch, seq_len, 512)
        
        # Global average pooling
        x = output.mean(dim=1)  # (batch, 512)
        
        # Process wide features
        wide = self.wide_fc(wide_features)  # (batch, 64)
        
        # Combine features
        combined = torch.cat([x, wide], dim=1)  # (batch, 576)
        output = self.combine(combined)
        
        return torch.sigmoid(output)