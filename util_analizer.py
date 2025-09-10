# ----- 완성본 (Dataset 수정 + 전체 루프) -----

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# 1) 데이터 생성
N_SAMPLES = 1000
N_STATS   = 30

rng = np.random.default_rng(SEED)
X = rng.normal(0, 1, size=(N_SAMPLES, N_STATS)).astype(np.float32)

w = np.zeros(N_STATS, dtype=np.float32)
w[[0, 3, 7, 12, 21]] = [1.6, -1.2, 2.0, 0.9, -1.5]
#logit = X @ w + rng.normal(0, 0.5, size=N_SAMPLES)
logit = X @ w

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
    def get_stats(self):
        """ (mean[1,S], std[1,S]) 를 torch.Tensor로 반환 (없으면 None) """
        return self.X, self.feat_idx, self.y

full_ds = StatDataset(X, y, standardize=True)
train_size = int(0.8 * len(full_ds))
test_size  = len(full_ds) - train_size
train_ds, test_ds = random_split(full_ds, [train_size, test_size],
                                 generator=torch.Generator().manual_seed(SEED))

BATCH_SIZE = 64
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

class EncoderLayerWithAttn(nn.Module):
    """
    PyTorch TransformerEncoderLayer와 동일한 구조(+attn weight 반환).
    """
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
            # Self-attention block
            xn = self.norm1(x)
            attn_out, attn_weight = self.self_attn(
                xn, xn, xn, need_weights=True, average_attn_weights=False  # <-- 핵심
            )  # attn_weight: [B, num_heads, T, T]
            x = x + self.dropout1(attn_out)

            # FFN block
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
        return x, attn_weight  # 둘 다 반환

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

        '''
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        '''
        # (변경) PyTorch 기본 Encoder 대신, 우리가 만든 레이어 스택 사용
        self.layers = nn.ModuleList([
            EncoderLayerWithAttn(d_model=d_model, nhead=nhead,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout, norm_first=True)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, values: torch.Tensor, feat_ids: torch.Tensor, return_attn: bool = False):
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
        
        #x = self.encoder(x)                                        # [B, 1+S, d]
        attn_maps = []  # 각 레이어의 self-attn weight 저장
        for layer in self.layers:
            x, attn = layer(x)                                  # attn: [B, H, T, T]
            if return_attn:
                attn_maps.append(attn)

        cls_out = x[:, 0, :]                                       # [B, d]
        out = self.head(cls_out).squeeze(-1)                       # [B]
        if self.use_sigmoid_head:
            out = torch.sigmoid(out)
        return out, attn_maps

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
        preds, _  = model(vals, feat_ids.to(device))
        loss   = criterion(preds, labels)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * vals.size(0)
        n     += vals.size(0)
    return total / max(n, 1)

EPOCHS = 300
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, model, optimizer)
    test_loss  = run_epoch(test_loader,  model, optimizer=None)
    print(f"[{epoch:02d}] train MSE: {train_loss:.4f} | test MSE: {test_loss:.4f}")


def compute_ig_for_inputs(model, inputs, device, steps=64, baseline_mode="zero"):
    """
    inputs: torch.Tensor 또는 numpy.ndarray, shape [N, S]
            (학습 때와 동일한 전처리 상태여야 함)
    return: attributions [N, S] (torch.Tensor, CPU로 반환)
    """
    model.eval()

    # to tensor
    if isinstance(inputs, np.ndarray):
        vals = torch.from_numpy(inputs.astype(np.float32))
    else:
        vals = inputs.clone().detach().float()
    vals = vals.to(device)  # [N, S]

    # feat_ids: [S] 자동 생성 (0..S-1)
    S = vals.size(1)
    feat_ids = torch.arange(S, dtype=torch.long, device=device)

    # baseline 만들기
    if baseline_mode == "zero" or baseline_mode is None:
        baseline = torch.zeros_like(vals)
    elif baseline_mode == "mean":
        baseline = vals.mean(dim=0, keepdim=True).expand_as(vals)
    else:
        raise ValueError("baseline_mode must be one of: None, 'zero', 'mean'")

    # IG 본체
    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1)  # [steps,1,1]
    path_points = baseline.unsqueeze(0) + alphas * (vals - baseline).unsqueeze(0)  # [steps,N,S]

    total_grads = torch.zeros_like(vals)
    for i in range(steps):
        x = path_points[i].clone().detach().requires_grad_(True)  # [N,S]
        preds,_ = model(x, feat_ids)                                # [N]
        preds_sum = preds.sum()
        grads = torch.autograd.grad(preds_sum, x, retain_graph=False)[0]  # [N,S]
        total_grads += grads

    avg_grads = total_grads / steps
    atts = (vals - baseline) * avg_grads                         # [N,S]
    return atts.detach().cpu()

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1) 데이터 생성
N_SAMPLES = 10

rng = np.random.default_rng(SEED)
X = rng.normal(0, 1, size=(N_SAMPLES, N_STATS)).astype(np.float32)

#logit = X @ w + rng.normal(0, 0.5, size=N_SAMPLES)
logit = X @ w
y = (1 / (1 + np.exp(-logit))).astype(np.float32)

inference_set = StatDataset(X, y, standardize=True)
inference_data, feat_ids, inference_util = inference_set.get_stats()

preds = None
attn_maps = None
with torch.no_grad():
    preds, attn_maps = model(inference_data, feat_ids, return_attn=True)

atts = compute_ig_for_inputs(
    model,
    inputs=inference_data,         # [10, S]
    device=device,
    steps=256,
    baseline_mode="zero"    # or "mean"
)                           # -> [10, S]

print("IG shape:", atts.shape)  # torch.Size([10, S])

def print_per_sample_topk(atts, topk=5, stat_names=None, preds=None, util=None):
    """
    atts: [N, S] torch.Tensor (compute_ig_for_inputs 결과)
    """
    A = atts.numpy()
    N, S = A.shape
    if stat_names is None:
        stat_names = [f"stat_{i+1}" for i in range(S)]

    for i in range(N):
        contrib = A[i]  # [S]
        pos_order = np.argsort(-contrib)  # 큰 양수부터
        neg_order = np.argsort(contrib)   # 큰 음수부터

        print(f"\n=== Sample {i} ===")
        if preds != None:
            print(f" pred={preds[i].item():.3f} | true={util[i].item():.3f}")
        print(f"[+IG Top-{topk}] (increase)")
        for r, j in enumerate(pos_order[:topk], 1):
            print(f" {r:2d}. {stat_names[j]} (idx={j}) IG={contrib[j]:.6f}")

        print(f"[-IG Top-{topk}] (decrease)")
        for r, j in enumerate(neg_order[:topk], 1):
            print(f" {r:2d}. {stat_names[j]} (idx={j}) IG={contrib[j]:.6f}")

# 호출
print_per_sample_topk(atts, topk=5, stat_names=[f"stat_{i+1}" for i in range(N_STATS)], preds=preds, util=y)




'''
def _normalize_feat_ids(feat_ids, device):
    # [S] 또는 [B,S]로 올 수 있으니 [S]로 정규화
    if feat_ids.dim() == 2:
        feat_ids = feat_ids[0]
    return feat_ids.to(device).long()

@torch.no_grad()
def _make_baseline(vals, mode="zero"):
    # vals: [B,S] (CPU/GPU 무관)
    if mode == "zero":
        return torch.zeros_like(vals)
    elif mode == "mean":
        mean = vals.mean(dim=0, keepdim=True)
        return mean.expand_as(vals)
    else:
        raise ValueError("mode must be 'zero' or 'mean'")

def integrated_gradients(model, vals, feat_ids, device, baseline=None, steps=64):
    """
    IG for regression output.
    vals: [B,S]  (requires_grad 내부에서 설정)
    feat_ids: [S] or [B,S]
    baseline: [B,S] or None(=zero)
    return: attributions [B,S]
    """
    model.eval()
    vals = vals.to(device)
    feat_ids = _normalize_feat_ids(feat_ids, device)

    if baseline is None:
        baseline = _make_baseline(vals, mode="zero").to(device)
    else:
        baseline = baseline.to(device)

    # alphas: 0..1
    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1)  # [steps,1,1]
    path_points = baseline.unsqueeze(0) + alphas * (vals - baseline).unsqueeze(0)  # [steps,B,S]

    total_grads = torch.zeros_like(vals, device=device)
    for i in range(steps):
        x = path_points[i].clone().detach().requires_grad_(True)  # [B,S]
        preds, _ = model(x, feat_ids)                 # [B]
        preds_sum = preds.sum()                    # 배치 합으로 스칼라화
        grads = torch.autograd.grad(preds_sum, x, retain_graph=False)[0]  # [B,S]
        total_grads += grads

    avg_grads = total_grads / steps               # [B,S]
    attributions = (vals - baseline) * avg_grads  # [B,S]
    print(attributions)
    return attributions.detach()

def ig_rank_signed(atts, stat_names=None, topk=30, mode="abs"):
    """
    atts: [B,S]  (Integrated Gradients 결과)
    mode: "abs" | "pos" | "neg"
      - abs: |IG| 기준 Top-K (방향 무시)
      - pos: +기여(예측 증가) Top-K
      - neg: -기여(예측 감소) Top-K
    """
    import numpy as np
    import torch

    # 표본 평균(부호 유지)
    contrib = atts.mean(dim=0).detach().cpu().numpy()  # shape [S]
    S = contrib.shape[0]
    if stat_names is None:
        stat_names = [f"stat_{i+1}" for i in range(S)]

    if mode == "abs":
        scores = np.abs(contrib)
        order  = np.argsort(-scores)
        title  = "Top-K by |IG| (magnitude)"
    elif mode == "pos":
        scores = contrib
        order  = np.argsort(-scores)   # 큰 양수부터
        title  = "Top-K by +IG (increasing utilization)"
    elif mode == "neg":
        scores = contrib
        order  = np.argsort(scores)    # 가장 음수부터
        title  = "Top-K by -IG (decreasing utilization)"
    else:
        raise ValueError("mode must be one of: 'abs', 'pos', 'neg'")

    top_idx = order[:topk]
    print(title)
    for r, j in enumerate(top_idx, 1):
        print(f"{r:2d}. {stat_names[j]} (idx={j})  score={scores[j]:.6f}  raw_IG_mean={contrib[j]:.6f}")

    return top_idx, scores, contrib

# 샘플 출력
vals, feat_ids, labels = next(iter(test_loader))
vals = vals.to(device); feat_ids = feat_ids.to(device)

with torch.no_grad():
    preds, attn_maps = model(vals, feat_ids, return_attn=True)

atts = integrated_gradients(
    model,
    vals=vals,
    feat_ids=feat_ids,
    device=device,
    baseline=None,   # or _make_baseline(vals, "mean")
    steps=64
)

# 랭킹/플롯
stat_names = [f"stat_{i+1}" for i in range(N_STATS)]
print("\n=== 증가(예측 올리는) Top-K ===")
pos_idx, pos_scores, contrib = ig_rank_signed(atts, stat_names, topk=10, mode="pos")

print("\n=== 감소(예측 내리는) Top-K ===")
neg_idx, neg_scores, _ = ig_rank_signed(atts, stat_names, topk=30, mode="neg")

print("\n=== 방향 무시 (절댓값) Top-K ===")
abs_idx, abs_scores, _ = ig_rank_signed(atts, stat_names, topk=10, mode="abs")
'''

'''
layer_idx = 0       # 0 ~ num_layers-1
head_idx  = 0       # 0 ~ nhead-1
sample_ix = 0       # 배치 내 샘플 인덱스


print("예측 예시 (앞 5개):")
for i in range(5):
    print(f" pred={preds[i].item():.3f} | true={labels[i].item():.3f}")

attn = attn_maps[layer_idx]              # [B, H, T, T]
attn = attn[sample_ix, head_idx].cpu()   # [T, T], T=1+S (CLS + S개 stat)

# 토큰 라벨 (행/열)
S = vals.size(1)
tokens = ["CLS"] + [f"stat_{i+1}" for i in range(S)]

plt.figure(figsize=(8, 7))
plt.imshow(attn, aspect="auto")
plt.colorbar(label="attention weight")
plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=90)
plt.yticks(ticks=np.arange(len(tokens)), labels=tokens)
plt.title(f"Layer {layer_idx}, Head {head_idx}, Sample {sample_ix}")
plt.tight_layout()
plt.show()
'''