import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# =============================
# 0) 기본 설정
# =============================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 합성 데이터 설정
N_SAMPLES = 1000
N_STATS   = 30

# Ground-truth 선형 가중치 (실제 영향 feature)
TRUE_W = np.zeros(N_STATS, dtype=np.float32)
TRUE_W[[0, 1, 2, 3, 4]] = [1.6, -1.2, 2.0, 0.9, -1.5]
TRUE_IDX = {0, 1, 2, 3, 4}

# =============================
# 1) 데이터 생성 유틸
# =============================
def make_dataset(n_samples: int, n_stats: int, w: np.ndarray, noise_std: float = 0.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_stats)).astype(np.float32)
    logit = X @ w
    if noise_std > 0:
        logit = logit + rng.normal(0, noise_std, size=n_samples)
    y = (1 / (1 + np.exp(-logit))).astype(np.float32)
    return X, y

# 훈련/검증용 데이터
X_all, y_all = make_dataset(N_SAMPLES, N_STATS, TRUE_W, noise_std=0.0, seed=SEED)

# =============================
# 2) Dataset (훈련 스케일 보존)
# =============================
class StatDataset(Dataset):
    """
    features/targets를 (선택적으로) 표준화하여 보관하는 Dataset
    - mean/std를 외부에서 주입하면 그 기준으로 표준화 (훈련 스케일을 추론에도 재사용)
    - 주입하지 않으면 내부에서 계산하여 저장
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, standardize: bool = True, mean=None, std=None):
        X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        if standardize:
            if mean is None:
                mean = X.mean(dim=0, keepdim=True)
            if std is None:
                std = X.std(dim=0, keepdim=True)
            std = std.clone()
            std[std == 0] = 1.0
            X = (X - mean) / std
        self.X = X
        self.feat_idx = torch.arange(self.X.shape[1], dtype=torch.long)
        self.mean = mean
        self.std  = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # values: [S], feat_ids: [S], label: []
        return self.X[idx], self.feat_idx, self.y[idx]

    def get_scaler(self):
        return self.mean, self.std

# 훈련용 Dataset (표준화: 내부 계산)
full_ds = StatDataset(X_all, y_all, standardize=True)
mean, std = full_ds.get_scaler()

# Train/Test split
train_size = int(0.8 * len(full_ds))
test_size  = len(full_ds) - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size],
                                 generator=torch.Generator().manual_seed(SEED))

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# =============================
# 3) Transformer 레이어 (어텐션 weight 반환)
# =============================
class EncoderLayerWithAttn(nn.Module):
    """PyTorch TransformerEncoderLayer와 동일한 구조(+attn weight 반환)."""
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, norm_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.dropout   = nn.Dropout(dropout)
        self.linear2   = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [B, T, D]
        if self.norm_first:
            xn = self.norm1(x)
            attn_out, attn_weight = self.self_attn(
                xn, xn, xn, need_weights=True, average_attn_weights=False
            )  # attn_weight: [B, H, T, T]
            x = x + self.dropout1(attn_out)

            xn = self.norm2(x)
            y = self.linear2(self.dropout(self.activation(self.linear1(xn))))
            x = x + self.dropout2(y)
        else:
            attn_out, attn_weight = self.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            x = self.norm1(x + self.dropout1(attn_out))
            y = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = self.norm2(x + self.dropout2(y))
        return x, attn_weight

# =============================
# 4) 모델 정의
# =============================
class StatTransformer(nn.Module):
    def __init__(self, num_stats: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, use_sigmoid_head: bool = True):
        super().__init__()
        self.use_sigmoid_head = use_sigmoid_head
        self.feat_embed = nn.Embedding(num_stats, d_model)
        self.val_proj  = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            EncoderLayerWithAttn(d_model=d_model, nhead=nhead,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout, norm_first=True)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, values: torch.Tensor, feat_ids: torch.Tensor, return_attn: bool = False):
        # values: [B, S], feat_ids: [S] or [B, S]
        B, S = values.shape
        if feat_ids.dim() == 2:
            feat_ids = feat_ids[0]
        feat_ids = feat_ids.to(values.device).long()

        val_emb  = self.val_proj(values.unsqueeze(-1))             # [B, S, d]
        feat_emb = self.feat_embed(feat_ids)                       # [S, d]
        feat_emb = feat_emb.unsqueeze(0).expand(B, S, -1)          # [B, S, d]
        tokens = val_emb + feat_emb                                # [B, S, d]

        cls = self.cls_token.expand(B, 1, -1)                      # [B, 1, d]
        x = torch.cat([cls, tokens], dim=1)                        # [B, 1+S, d]

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)                                     # attn: [B, H, T, T]
            if return_attn:
                attn_maps.append(attn)

        cls_out = x[:, 0, :]                                       # [B, d]
        out = self.head(cls_out).squeeze(-1)                       # [B]
        if self.use_sigmoid_head:
            out = torch.sigmoid(out)
        return out, attn_maps

model = StatTransformer(num_stats=N_STATS, d_model=64, nhead=4, num_layers=2,
                        dim_feedforward=128, dropout=0.1, use_sigmoid_head=True).to(device)

# =============================
# 5) Optimizer/손실/지표
# =============================
#criterion = nn.MSELoss()
#criterion = nn.L1Loss()
class MAPELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        # preds, targets: [B]
        diff = torch.abs((targets - preds) / (targets.abs() + self.eps))
        #return diff.mean()
        return diff.mean()

class PiecewiseLoss(nn.Module):
    def __init__(self, y_thresh: float = 0.1, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.y_thresh = y_thresh
        self.reduction = reduction
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Piecewise loss:
          - if target < y_thresh: use relative absolute error (|y - y_hat| / (|y|+eps))
          - else: use absolute error (|y - y_hat|)
        """
        abs_err = torch.abs(preds - targets)
        rel_err = abs_err / (targets.abs() + self.eps)

        # element-wise choose: relative error if target < y_thresh, else absolute error
        loss_per_elem = torch.where(abs_err < rel_err, rel_err, abs_err)

        if self.reduction == "none":
            return loss_per_elem
        elif self.reduction == "sum":
            return loss_per_elem.sum()
        else:  # default "mean"
            return loss_per_elem.mean()
        
criterion = PiecewiseLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# =============================
# 6) Train/Eval 루프
# =============================
def run_epoch(loader, model, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total, n = 0.0, 0
    all_preds, all_labels = [], []
    for vals, feat_ids, labels in loader:
        vals   = vals.to(device)
        labels = labels.to(device)
        preds, _  = model(vals, feat_ids.to(device))
        loss   = criterion(preds, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * vals.size(0)
        n     += vals.size(0)
        all_preds.append(preds.detach())
        all_labels.append(labels.detach())
    epoch_loss = total / max(n, 1)
    
    return epoch_loss

EPOCHS = 3000
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, model, optimizer)
    test_loss  = run_epoch(test_loader,  model, optimizer=None)
    if (epoch % 10 == 0) or (epoch == 1):
        print(f"[{epoch:03d}] train MSE={train_loss:.4f} | "
              f"test MSE={test_loss:.4f}")

# =============================
# 7) IG
# =============================
def compute_ig_for_inputs(model, inputs, device, steps=64, baseline_mode="zero"):
    """
    inputs: torch.Tensor or np.ndarray (표준화된 공간!) shape [N, S]
    return: attributions [N, S] (torch.Tensor, CPU)
    """
    model.eval()

    if isinstance(inputs, np.ndarray):
        vals = torch.from_numpy(inputs.astype(np.float32))
    else:
        vals = inputs.clone().detach().float()
    vals = vals.to(device)

    S = vals.size(1)
    feat_ids = torch.arange(S, dtype=torch.long, device=device)

    # baseline (표준화된 공간에서 zero는 곧 각 feature의 평균)
    if baseline_mode in (None, "zero"):
        baseline = torch.zeros_like(vals)
    elif baseline_mode == "mean":
        baseline = vals.mean(dim=0, keepdim=True).expand_as(vals)
    else:
        raise ValueError("baseline_mode must be one of: None, 'zero', 'mean'")

    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1)  # [steps,1,1]
    path_points = baseline.unsqueeze(0) + alphas * (vals - baseline).unsqueeze(0)  # [steps,N,S]

    total_grads = torch.zeros_like(vals)
    for i in range(steps):
        x = path_points[i].clone().detach().requires_grad_(True)  # [N,S]
        preds, _ = model(x, feat_ids)                              # [N]
        preds_sum = preds.sum()
        grads = torch.autograd.grad(preds_sum, x, retain_graph=False)[0]  # [N,S]
        total_grads += grads

    avg_grads = total_grads / steps
    atts = (vals - baseline) * avg_grads                         # [N,S]
    return atts.detach().cpu()

@torch.no_grad()
def ig_completeness_check(model, inputs, device, atts, baseline_mode="zero"):
    model.eval()
    if isinstance(inputs, np.ndarray):
        vals = torch.from_numpy(inputs.astype(np.float32)).to(device)
    else:
        vals = inputs.to(device)
    S = vals.size(1)
    feat_ids = torch.arange(S, dtype=torch.long, device=device)

    if baseline_mode in (None, "zero"):
        baseline = torch.zeros_like(vals)
    elif baseline_mode == "mean":
        baseline = vals.mean(dim=0, keepdim=True).expand_as(vals)
    else:
        raise ValueError

    f_x, _ = model(vals, feat_ids)
    f_b, _ = model(baseline, feat_ids)

    lhs = atts.sum(dim=1).cpu().numpy()            # Σ IG_j
    rhs = (f_x - f_b).detach().cpu().numpy()       # f(x) - f(baseline)
    return lhs, rhs

# =============================
# 8) 어텐션 해석 (CLS -> stat token)
# =============================
@torch.no_grad()
def cls_to_stat_attention(attn_maps):
    """레이어별 CLS가 stat에 주는 평균 어텐션 [num_layers, S]"""
    per_layer = []
    for A in attn_maps:             # [B, H, T, T]
        A_mean = A.mean(dim=(0, 1)) # [T, T]
        cls_to_stat = A_mean[0, 1:] # [S]
        per_layer.append(cls_to_stat.detach().cpu().numpy())
    return per_layer

# =============================
# 9) IG 결과 출력/평가
# =============================
def print_per_sample_topk(atts, topk=5, stat_names=None, preds=None, util=None):
    A = atts.numpy()
    N, S = A.shape
    if stat_names is None:
        stat_names = [f"stat_{i+1}" for i in range(S)]
    for i in range(N):
        contrib = A[i]
        pos_order = np.argsort(-contrib)  # 큰 양수
        neg_order = np.argsort(contrib)   # 큰 음수
        print(f"\n=== Sample {i} ===")
        if preds is not None and util is not None:
            print(f" pred={preds[i].item():.3f} | true={util[i].item():.3f} | diff={(util[i].item()-preds[i].item())/util[i].item()*100:.3f}")
        print(f"[+IG Top-{topk}] (increase)")
        for r, j in enumerate(pos_order[:topk], 1):
            print(f" {r:2d}. {stat_names[j]} (idx={j}) IG={contrib[j]:.6f}")
        print(f"[-IG Top-{topk}] (decrease)")
        for r, j in enumerate(neg_order[:topk], 1):
            print(f" {r:2d}. {stat_names[j]} (idx={j}) IG={contrib[j]:.6f}")

@torch.no_grad()
def ig_topk_hit_rate(atts: torch.Tensor, true_idx: set, topk=5):
    A = atts.numpy()
    hits = 0
    for i in range(A.shape[0]):
        top = np.argsort(-A[i])[:topk]
        hits += len([t for t in top if t in true_idx])
    return hits/(A.shape[0]*topk)

# =============================
# 10) 추론/IG 실행 (훈련 스케일 재사용)
# =============================
# 새로운 샘플 생성 (분포 동일, 다른 시드)
X_infer, y_infer = make_dataset(n_samples=10, n_stats=N_STATS, w=TRUE_W, noise_std=0.0, seed=SEED+1)

# 훈련에서 저장한 mean/std로 동일 표준화 적용
inference_set = StatDataset(X_infer, y_infer, standardize=True, mean=mean, std=std)
inference_data, feat_ids, inference_util = inference_set.X.to(device), inference_set.feat_idx.to(device), inference_set.y

with torch.no_grad():
    preds, attn_maps = model(inference_data, feat_ids, return_attn=True)

# IG 계산 (표준화된 공간이므로 baseline=zero는 평균을 의미)
atts = compute_ig_for_inputs(model, inputs=inference_data, device=device, steps=256, baseline_mode="zero")
print("IG shape:", atts.shape)

# 완전성(completeness) 체크
lhs, rhs = ig_completeness_check(model, inference_data, device, atts, baseline_mode="zero")
print("Completeness check -> mean|ΣIG - Δf|:", float(np.mean(np.abs(lhs - rhs))))

# IG Top-k에서 정답 feature에 대한 hit-rate
hit_rate = ig_topk_hit_rate(atts, TRUE_IDX, topk=5)
print(f"IG Top-5 hit-rate vs TRUE_IDX: {hit_rate:.3f}")

# 주의(어텐션) 집계
per_layer_attn = cls_to_stat_attention(attn_maps)
for li, vec in enumerate(per_layer_attn):
    top5 = np.argsort(-vec)[:5]
    print(f"Layer {li} CLS->stat attention Top-5 idx:", top5)

# 샘플별 Top-k 인쇄
print_per_sample_topk(atts, topk=5,
                      stat_names=[f"stat_{i}" for i in range(N_STATS)],
                      preds=preds.cpu(), util=inference_util)
