import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DRAMDataset(Dataset):
    def __init__(self, n_samples=10000, n_stat=20):
        super().__init__()
        self.x = torch.rand(n_samples, n_stat)  # 10,000개 샘플, 각 샘플은 20개의 stat
        self.y = torch.rand(n_samples)          # 각 샘플에 대한 utilization (0~1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
dataset = DRAMDataset()
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

class DRAMTransformer(nn.Module):
    def __init__(self, num_stats=20, d_model=64, n_head=4, num_layers=2, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_stats, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: (B, 20)
        x = x.unsqueeze(-1)                  # (B, 20, 1)
        x = self.embedding(x)                # (B, 20, d_model)
        x = x + self.pos_embedding           # (B, 20, d_model)
        x = x.transpose(0, 1)                # (20, B, d_model) for Transformer
        x = self.transformer_encoder(x)      # (20, B, d_model)
        x = x.transpose(0, 1)                # (B, 20, d_model)
        #x = x.mean(dim=1)                    # (B, d_model)
        x = x[:, 0, :]
        out = self.mlp(x)                    # (B, 1)
        return out.squeeze(1)                # (B,)
    
model = DRAMTransformer()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):  # epoch 수는 상황에 맞게 조절
    total_loss = 0
    for x_batch, y_batch in dataloader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
