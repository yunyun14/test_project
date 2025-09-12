# ============================================================
# SoC Transformer Demo (Master + Global Stats, NO Config)
#  - Inputs:
#       Xm: [B,L,M,Fm]  (per-master stats)
#       Xg: [B,L,Fg]    (global DRAM/bus/controller stats)
#  - Outputs:
#       per-time:  util_k [B,L,1], latency_k [B,L,M]
#       future:    util_next [B,1], latency_next [B,M]
#  - Includes: causal mask, IG for future util (attribution on Xm and Xg)
# PyTorch >= 2.0
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Toy shapes
# -----------------------------
BATCH = 8        # batch size
L     = 12       # window length (time steps)
M     = 5        # # of masters
Fm    = 30       # # of per-master features
Fg    = 6        # # of global features
D     = 128      # model hidden dim
N_TRAIN = 256
N_VAL   = 32
EPOCHS  = 5
LR      = 2e-3

# -----------------------------
# Random dataset (with weak structure)
# Xm: [N,L,M,Fm], Xg: [N,L,Fg]
# Labels:
#   util_k:        [N, L, 1]  (depends on master+global)
#   latency_k:     [N, L, M]  (depends on master)
#   util_next:     [N, 1]     (depends on last-step master+global)
#   latency_next:  [N, M]     (depends on last-step master)
# -----------------------------
def make_random_dataset(n):
    Xm = torch.randn(n, L, M, Fm)
    Xg = torch.randn(n, L, Fg)

    # util_k: simple function of early master features + some global stats
    util_k_base = Xm[..., :4].mean(dim=(-1, -2))          # (n, L)
    util_k_glob = 0.3 * Xg[..., :2].mean(dim=-1)          # (n, L)
    util_k = (util_k_base + util_k_glob).unsqueeze(-1)     # (n, L, 1)
    util_k = util_k + 0.05*torch.randn(n, L, 1)

    # latency_k (per master): depend on per-master features
    latency_k = Xm[..., :6].mean(dim=-1) + 0.05*torch.randn(n, L, M)  # (n, L, M)

    # future labels from last step
    util_next_base = Xm[:, -1, ...].mean(dim=(-1, -2))     # (n,)
    util_next_glob = 0.3 * Xg[:, -1, :2].mean(dim=-1)      # (n,)
    util_next = (util_next_base + util_next_glob).unsqueeze(-1) + 0.05*torch.randn(n, 1)

    latency_next = Xm[:, -1, ...].mean(dim=-1)[:, :M] + 0.05*torch.randn(n, M)

    return Xm, Xg, util_k, latency_k, util_next, latency_next

Xm_tr, Xg_tr, u_tr, l_tr, un_tr, ln_tr = make_random_dataset(N_TRAIN)
Xm_va, Xg_va, u_va, l_va, un_va, ln_va = make_random_dataset(N_VAL)

# -----------------------------
# DataLoader
# -----------------------------
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, Xm, Xg, u, lat, unext, lnext):
        self.Xm = Xm; self.Xg = Xg; self.u = u; self.lat = lat; self.un = unext; self.ln = lnext
    def __len__(self): return self.Xm.shape[0]
    def __getitem__(self, i):
        return (self.Xm[i], self.Xg[i], self.u[i], self.lat[i], self.un[i], self.ln[i])

tr_loader = torch.utils.data.DataLoader(
    TensorDataset(Xm_tr, Xg_tr, u_tr, l_tr, un_tr, ln_tr),
    batch_size=BATCH, shuffle=True
)
va_loader = torch.utils.data.DataLoader(
    TensorDataset(Xm_va, Xg_va, u_va, l_va, un_va, ln_va),
    batch_size=BATCH, shuffle=False
)

# -----------------------------
# Model blocks
# -----------------------------
class MasterStatEncoder(nn.Module):
    """Per-master stat MLP: [B,L,M,Fm] -> [B,L,M,D]"""
    def __init__(self, Fm, D):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Fm, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )
    def forward(self, xm):  # [B,L,M,Fm]
        return self.mlp(xm) # [B,L,M,D]

class GlobalStatEncoder(nn.Module):
    """Global stat MLP: [B,L,Fg] -> [B,L,1,D] (single token per time)"""
    def __init__(self, Fg, D):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Fg, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )
    def forward(self, xg):  # [B,L,Fg]
        g = self.mlp(xg)    # [B,L,D]
        return g.unsqueeze(2)  # [B,L,1,D]

class IntraTimeEncoder(nn.Module):
    """Within-time self-attn over [GlobalToken + MasterTokens] -> TimeToken + refined master tokens
       Vectorized over time by flattening [B*L, (1+M), D].
    """
    def __init__(self, D, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, nhead, batch_first=True)
        self.ln = nn.LayerNorm(D)
    def forward(self, g_tok, m_tok):  # g_tok: [B,L,1,D], m_tok: [B,L,M,D]
        B,L,M,D = m_tok.shape
        tokens = torch.cat([g_tok, m_tok], dim=2)          # [B,L,(1+M),D]
        x = tokens.reshape(B*L, 1+M, D)                    # [B*L,1+M,D]
        y,_ = self.attn(x, x, x)                           # [B*L,1+M,D]
        y = self.ln(y)
        y = y.reshape(B, L, 1+M, D)
        time_token = y[:, :, :1, :]                        # [B,L,1,D]  (use global slot as time summary)
        master_ref = y[:, :, 1:, :]                        # [B,L,M,D]
        return time_token, master_ref

class TemporalEncoder(nn.Module):
    """Causal Transformer encoder over time tokens: [B,L,1,D] -> [B,L,D]"""
    def __init__(self, D, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=D, nhead=nhead, dim_feedforward=4*D,
                dropout=dropout, batch_first=True, norm_first=True
            ) for _ in range(nlayers)
        ])
    def forward(self, z_time, causal_mask=True):  # z_time: [B,L,1,D]
        B,L,_,D = z_time.shape
        x = z_time.squeeze(2)  # [B,L,D]
        mask = None
        if causal_mask:
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, mask)
        return x  # [B,L,D]

class Heads(nn.Module):
    """Outputs:
       - contemporaneous: util per time from time tokens; latency per master per time from master tokens
       - future (t+1): util, latency from last time token
    """
    def __init__(self, D, M):
        super().__init__()
        self.util_head = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))   # per time
        self.lat_head  = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))   # per master per time
        self.future_util = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1)) # from last time token
        self.future_lat  = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, M))
    def forward(self, time_seq, master_seq_ref):
        util_k = self.util_head(time_seq)                        # [B,L,1]
        lat_k  = self.lat_head(master_seq_ref).squeeze(-1)       # [B,L,M]
        z_last = time_seq[:, -1, :]                               # [B,D]
        util_next = self.future_util(z_last)                     # [B,1]
        lat_next  = self.future_lat(z_last)                      # [B,M]
        return util_k, lat_k, util_next, lat_next

class SoCForecastModel(nn.Module):
    def __init__(self, Fm, Fg, M, D):
        super().__init__()
        self.enc_m   = MasterStatEncoder(Fm, D)
        self.enc_g   = GlobalStatEncoder(Fg, D)
        self.intra   = IntraTimeEncoder(D, nhead=4)
        self.temp    = TemporalEncoder(D, nhead=4, nlayers=2, dropout=0.1)
        self.heads   = Heads(D, M)
    def forward(self, Xm, Xg):
        """
        Xm: [B,L,M,Fm], Xg: [B,L,Fg]
        returns: util_k [B,L,1], latency_k [B,L,M], util_next [B,1], latency_next [B,M]
        """
        m_tok = self.enc_m(Xm)                         # [B,L,M,D]
        g_tok = self.enc_g(Xg)                         # [B,L,1,D]
        time_tok, master_ref = self.intra(g_tok, m_tok)# [B,L,1,D], [B,L,M,D]
        time_seq = self.temp(time_tok)                 # [B,L,D] (causal)
        util_k, lat_k, util_next, lat_next = self.heads(time_seq, master_ref)
        return util_k, lat_k, util_next, lat_next

# -----------------------------
# Train
# -----------------------------
model = SoCForecastModel(Fm, Fg, M, D).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

def step_batch(batch, train=True):
    Xm, Xg, u, lat, un, ln = batch
    Xm = Xm.to(device).float()
    Xg = Xg.to(device).float()
    u  = u.to(device).float()
    lat= lat.to(device).float()
    un = un.to(device).float()
    ln = ln.to(device).float()

    util_k, lat_k, util_next, lat_next = model(Xm, Xg)

    loss_util_k = F.mse_loss(util_k, u)
    loss_lat_k  = F.mse_loss(lat_k, lat)
    loss_util_n = F.mse_loss(util_next, un)
    loss_lat_n  = F.mse_loss(lat_next, ln)

    loss = loss_util_k + loss_lat_k + 0.5*loss_util_n + 0.5*loss_lat_n

    if train:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        mae_util = (util_k - u).abs().mean().item()
        mae_lat  = (lat_k - lat).abs().mean().item()
        mae_un   = (util_next - un).abs().mean().item()
        mae_ln   = (lat_next - ln).abs().mean().item()

    return loss.item(), mae_util, mae_lat, mae_un, mae_ln

print("Training...")
for ep in range(1, EPOCHS+1):
    model.train()
    tr_stats = []
    for batch in tr_loader:
        tr_stats.append(step_batch(batch, train=True))
    tr_stats = torch.tensor(tr_stats).mean(dim=0).tolist()

    model.eval()
    va_stats = []
    with torch.no_grad():
        for batch in va_loader:
            va_stats.append(step_batch(batch, train=False))
    va_stats = torch.tensor(va_stats).mean(dim=0).tolist()

    print(f"[Epoch {ep:02d}] "
          f"TrainLoss={tr_stats[0]:.4f} | "
          f"MAE(Util_k/Lat_k/Util_n/Lat_n)={tr_stats[1]:.3f}/{tr_stats[2]:.3f}/{tr_stats[3]:.3f}/{tr_stats[4]:.3f}  ||  "
          f"ValLoss={va_stats[0]:.4f} | "
          f"ValMAE={va_stats[1]:.3f}/{va_stats[2]:.3f}/{va_stats[3]:.3f}/{va_stats[4]:.3f}"
    )

# =========================
# IG: per-time / per-time&master attribution
# =========================

@torch.no_grad()
def _baseline_like(x, kind="mean"):
    """x: [1,L,M,Fm] or [1,L,Fg]"""
    if kind == "zero":
        return torch.zeros_like(x)
    if x.ndim == 4:   # master: [1,L,M,Fm]
        mean_t = x.mean(dim=(0,1), keepdim=True)  # [1,1,M,Fm]
    elif x.ndim == 3: # global: [1,L,Fg]
        mean_t = x.mean(dim=(0,1), keepdim=True)  # [1,1,Fg]
    else:
        mean_t = x.mean().expand_as(x)
    return mean_t.expand_as(x)

def ig_util_at_time(model, Xm_sample, Xg_sample, k: int, steps: int = 64, baseline="mean"):
    """
    시점 k의 utilization(û_k)을 타깃으로 IG 수행.
    Returns:
      attr_m: [L,M,Fm]  (master 입력에 대한 기여도)
      attr_g: [L,Fg]    (global 입력에 대한 기여도)
    """
    model.eval()
    xm = Xm_sample.unsqueeze(0).to(device).float()  # [1,L,M,Fm]
    xg = Xg_sample.unsqueeze(0).to(device).float()  # [1,L,Fg]
    bm = _baseline_like(xm, baseline)
    bg = _baseline_like(xg, baseline)

    igm = torch.zeros_like(xm)
    igg = torch.zeros_like(xg)

    for s in range(1, steps+1):
        alpha = s / steps
        xm_s = bm + alpha * (xm - bm); xm_s.requires_grad_(True)
        xg_s = bg + alpha * (xg - bg); xg_s.requires_grad_(True)

        util_k, lat_k, util_next, lat_next = model(xm_s, xg_s)
        # 타깃: 시점 k의 utilization (배치=1이므로 sum으로 스칼라화)
        y = util_k[:, k, :].sum()
        grads = torch.autograd.grad(y, (xm_s, xg_s), retain_graph=False, create_graph=False)
        igm += grads[0]
        igg += grads[1]

    attr_m = (xm - bm) * (igm / steps)   # [1,L,M,Fm]
    attr_g = (xg - bg) * (igg / steps)   # [1,L,Fg]
    return attr_m.squeeze(0).cpu(), attr_g.squeeze(0).cpu()

def ig_latency_at_time_master(model, Xm_sample, Xg_sample, k: int, i: int, steps: int = 64, baseline="mean"):
    """
    시점 k, 마스터 i의 latency(ℓ_{k,i})를 타깃으로 IG 수행.
    Returns:
      attr_m: [L,M,Fm]
      attr_g: [L,Fg]
    """
    model.eval()
    xm = Xm_sample.unsqueeze(0).to(device).float()
    xg = Xg_sample.unsqueeze(0).to(device).float()
    bm = _baseline_like(xm, baseline)
    bg = _baseline_like(xg, baseline)

    igm = torch.zeros_like(xm)
    igg = torch.zeros_like(xg)

    for s in range(1, steps+1):
        alpha = s / steps
        xm_s = bm + alpha * (xm - bm); xm_s.requires_grad_(True)
        xg_s = bg + alpha * (xg - bg); xg_s.requires_grad_(True)

        util_k, lat_k, util_next, lat_next = model(xm_s, xg_s)
        # 타깃: 시점 k, 마스터 i의 latency
        y = lat_k[:, k, i].sum()
        grads = torch.autograd.grad(y, (xm_s, xg_s), retain_graph=False, create_graph=False)
        igm += grads[0]
        igg += grads[1]

    attr_m = (xm - bm) * (igm / steps)
    attr_g = (xg - bg) * (igg / steps)
    return attr_m.squeeze(0).cpu(), attr_g.squeeze(0).cpu()

def topk_util_attr_combined(attr_m, attr_g, k_top=5):
    """
    attr_m: [L,M,Fm]
    attr_g: [L,Fg]
    Returns:
      top_k: list of (val, time, "master"/"global", master_idx, feat_idx)
      bottom_k: same but for smallest abs value
    """
    entries = []

    # master part
    L, M, Fm = attr_m.shape
    for t in range(L):
        for m in range(M):
            for f in range(Fm):
                v = attr_m[t,m,f].item()
                entries.append((v, v, t, "master", m, f))

    # global part
    Lg, Fg = attr_g.shape
    for t in range(Lg):
        for g in range(Fg):
            v = attr_g[t,g].item()
            entries.append((v, v, t, "global", None, g))

    # sort by absolute value
    entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)

    # top-k
    top_k = [(val, raw, t, kind, m, f) for (val, raw, t, kind, m, f) in entries_sorted[:k_top]]
    # bottom-k (주의: 여기서는 abs 기준 제일 작은 것들 → 영향 거의 없는 것)
    bottom_k = [(val, raw, t, kind, m, f) for (val, raw, t, kind, m, f) in entries_sorted[-k_top:]]

    return top_k, bottom_k

# 샘플 하나 집어오기
Xm_sample = Xm_va[0]   # [L,M,Fm]
Xg_sample = Xg_va[0]   # [L,Fg]


# 예: 시점 k=10의 utilization attribution
attr_m, attr_g = ig_util_at_time(model, Xm_sample, Xg_sample, k=10, steps=512, baseline="mean")

top_k, bottom_k = topk_util_attr_combined(attr_m, attr_g, k_top=5)

print("=== Top-5 influential stats ===")
for val, raw, t, kind, m, f in top_k:
    if kind == "master":
        print(f"time={t}, master={m}, feat={f}, value={raw:.4f}, |val|={val:.4f}")
    else:
        print(f"time={t}, global_feat={f}, value={raw:.4f}, |val|={val:.4f}")

print("\n=== Bottom-5 (least influential) stats ===")
for val, raw, t, kind, m, f in bottom_k:
    if kind == "master":
        print(f"time={t}, master={m}, feat={f}, value={raw:.4f}, |val|={val:.4f}")
    else:
        print(f"time={t}, global_feat={f}, value={raw:.4f}, |val|={val:.4f}")

# 예: 시점 k=10, master i=2 의 latency attribution
attr_m_l, attr_g_l = ig_latency_at_time_master(model, Xm_sample, Xg_sample, k=10, i=2, steps=512, baseline="mean")

top_k_l, bottom_k_l = topk_util_attr_combined(attr_m_l, attr_g_l, k_top=5)

print("\n=== Top-5 influential stats for Latency(k=10, master=2) ===")
for val, raw, t, kind, m, f in top_k_l:
    if kind == "master":
        print(f"time={t}, master={m}, feat={f}, value={raw:.4f}, |val|={val:.4f}")
    else:
        print(f"time={t}, global_feat={f}, value={raw:.4f}, |val|={val:.4f}")
