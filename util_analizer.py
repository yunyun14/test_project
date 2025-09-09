# ----- 완성본 (Dataset 수정 + 전체 루프) -----

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# 1) 데이터 생성
N_SAMPLES = 500
N_STATS   = 30

rng = np.random.default_rng(SEED)
X = rng.normal(0, 1, size=(N_SAMPLES, N_STATS)).astype(np.float32)

w = np.zeros(N_STATS, dtype=np.float32)
w[[0, 3, 7, 12, 21]] = [1.6, -1.2, 2.0, 0.9, -1.5]
logit = X @ w + rng.normal(0, 0.5, size=N_SAMPLES)
y = (1 / (1 + np.exp(-logit))).astype(np.float32)

# 2) Dataset (values, feat_ids, label 모두 반환)
class StatDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, standardize: bool = True):
        X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        if standardize:
            mean = X.mean(dim=0, keepdim=True)
            std  = X.std(dim=0, keepdim=True)
            std[std == 0] = 1.0
            X = (X - mean) / std
        self.X = X
        self.feat_idx = torch.arange(self.X.shape[1], dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # values: [S], feat_ids: [S], label: []
        return self.X[idx], self.feat_idx, self.y[idx]

full_ds = StatDataset(X, y, standardize=False)
train_size = int(0.8 * len(full_ds))
test_size  = len(full_ds) - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size],
                                 generator=torch.Generator().manual_seed(SEED))

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# 3) 모델
class StatTransformer(nn.Module):
    def __init__(self, num_stats: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, use_sigmoid_head: bool = True):
        super().__init__()
        self.use_sigmoid_head = use_sigmoid_head
        self.feat_embed = nn.Embedding(num_stats, d_model)
        self.val_proj  = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, values: torch.Tensor, feat_ids: torch.Tensor):
        # values: [B, S], feat_ids: [S]
        B, S = values.shape
        
        # (핵심) feat_ids가 [B, S]로 들어오면 첫 행만 사용해서 [S]로 축소
        if feat_ids.dim() == 2:
            feat_ids = feat_ids[0]
        # dtype 보장
        feat_ids = feat_ids.to(values.device).long()

        val_emb  = self.val_proj(values.unsqueeze(-1))             # [B, S, d]
        feat_emb = self.feat_embed(feat_ids.to(values.device))     # [S, d]
        feat_emb = feat_emb.unsqueeze(0).expand(B, S, -1)          # [B, S, d]
        tokens = val_emb + feat_emb                                # [B, S, d]

        cls = self.cls_token.expand(B, 1, -1)                      # [B, 1, d]
        x = torch.cat([cls, tokens], dim=1)                        # [B, 1+S, d]
        x = self.encoder(x)                                        # [B, 1+S, d]

        cls_out = x[:, 0, :]                                       # [B, d]
        out = self.head(cls_out).squeeze(-1)                       # [B]
        if self.use_sigmoid_head:
            out = torch.sigmoid(out)
        return out

model = StatTransformer(num_stats=N_STATS, d_model=64, nhead=4, num_layers=2,
                        dim_feedforward=128, dropout=0.1, use_sigmoid_head=True).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# 4) 학습 / 검증
def run_epoch(loader, model, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total, n = 0.0, 0
    for vals, feat_ids, labels in loader:
        vals   = vals.to(device)        # [B, S]
        labels = labels.to(device)      # [B]
        preds  = model(vals, feat_ids.to(device))
        loss   = criterion(preds, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * vals.size(0)
        n     += vals.size(0)
    return total / max(n, 1)

EPOCHS = 100
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, model, optimizer)
    test_loss  = run_epoch(test_loader,  model, optimizer=None)
    print(f"[{epoch:02d}] train MSE: {train_loss:.4f} | test MSE: {test_loss:.4f}")

# 샘플 출력
vals, feat_ids, labels = next(iter(test_loader))
with torch.no_grad():
    preds = model(vals.to(device), feat_ids.to(device)).cpu()
print("예측 예시 (앞 5개):")
for i in range(5):
    print(f" pred={preds[i].item():.3f} | true={labels[i].item():.3f}")
