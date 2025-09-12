# =========================
# Single-Encoder SOC Performance (Full Script)
# - One TransformerEncoder shared by Analysis & Rollout
# - Block-causal mask across time
# - Random synthetic dataset (N=300)
# - Mini-batch training loop
# - Quick eval: analysis vs rollout
# =========================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------
# 0) Repro & device
# ---------------------
SEED = 5678
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")  # 필요하면 "cuda"로 바꾸세요

# ---------------------
# 1) Problem sizes
# ---------------------
T   = 8    # time steps per sequence
M   = 4    # masters
SG  = 8    # global stat dim
SM  = 6    # master stat dim (0th = remaining BW)
CG  = 5    # global cfg dim
CGD = 5    # global delta cfg dim
CM  = 5    # per-master cfg dim
CMD = 5    # per-master delta cfg dim
REMAIN_IDX = 0

D      = 32   # token dim (모델 너비)
LAYERS = 1    # Transformer encoder layers
HEADS  = 4    # attention heads

TOKENS_PER_T = 3 + 3*M  # [SG, CG, CGD, SM×M, CM×M, CMD×M]

# ---------------------
# 2) Synthetic label generator (hidden dynamics)
# ---------------------
# 학습용 정답(y)을 만들기 위한 은닉 선형계/동역학 (랜덤 고정)
W_util_sg  = torch.randn(SG, 1) * 0.6
W_util_cg  = torch.randn(CG, 1) * 0.4
W_util_cgd = torch.randn(CGD,1) * 0.2

W_lat_sm   = torch.randn(SM, 1) * 0.7
W_lat_cm   = torch.randn(CM, 1) * 0.5
W_lat_cgd  = torch.randn(CGD,1) * 0.2

A_sg    = torch.randn(SG, SG) * 0.05
A_sm    = torch.randn(SM, SM) * 0.05
B_g_cg  = torch.randn(CG,  SG) * 0.03
B_g_cgd = torch.randn(CGD, SG) * 0.03
B_m_cm  = torch.randn(CM,  SM) * 0.03
B_m_cmd = torch.randn(CMD, SM) * 0.03

def synth_labels(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd):
    """
    Return:
      y_util[T,1], y_lat[T,M,1], y_next_sg[T,SG], y_next_sm[T,M,SM]
    """
    Tn = X_sg.shape[0]; Mn = X_sm.shape[1]
    y_util = torch.zeros(Tn, 1)
    y_lat  = torch.zeros(Tn, Mn, 1)
    y_next_sg = torch.zeros(Tn, SG)
    y_next_sm = torch.zeros(Tn, Mn, SM)
    with torch.no_grad():
        for t in range(Tn):
            y_util[t] = torch.sigmoid(X_sg[t] @ W_util_sg + X_cg[t] @ W_util_cg + X_cgd[t] @ W_util_cgd)
            lat_m = (X_sm[t] @ W_lat_sm + X_cm[t] @ W_lat_cm + (X_cgd[t] @ W_lat_cgd).view(1,1)).squeeze(-1)
            y_lat[t] = F.softplus(lat_m).unsqueeze(-1)
            if t < Tn-1:
                y_next_sg[t] = X_sg[t] + X_sg[t] @ A_sg + X_cg[t] @ B_g_cg + X_cgd[t] @ B_g_cgd
                ns = X_sm[t] + X_sm[t] @ A_sm + X_cm[t] @ B_m_cm + X_cmd[t] @ B_m_cmd
                # 간단 소비 모델: 남은 BW 감소 (0..1로 클램프)
                served = torch.sigmoid((X_sm[t] @ W_lat_sm).squeeze(-1)) * 0.15
                R_next = torch.clamp(X_sm[t,:,REMAIN_IDX] - served, 0.0, 1.0)
                ns[:, REMAIN_IDX] = R_next
                y_next_sm[t] = ns
            else:
                y_next_sg[t] = X_sg[t]
                y_next_sm[t] = X_sm[t]
    return y_util, y_lat, y_next_sg, y_next_sm

# ---------------------
# 3) Dataset
# ---------------------
class SocDataset(Dataset):
    def __init__(self, N=300, rollout_deltas=False):
        self.N = N
        self.rollout_deltas = rollout_deltas
        self.data = []
        for _ in range(N):
            X_sg  = torch.randn(T, SG)
            X_sm  = torch.randn(T, M, SM) * 0.3
            X_sm[..., REMAIN_IDX] = 1.0  # start remaining BW
            X_cg  = torch.randn(T, CG) * 0.3
            X_cm  = torch.randn(T, M, CM) * 0.2
            if rollout_deltas:
                X_cgd = torch.randn(T, CGD) * 0.2
                X_cmd = torch.randn(T, M, CMD) * 0.2
            else:
                X_cgd = torch.zeros(T, CGD)
                X_cmd = torch.zeros(T, M, CMD)
            y_util, y_lat, y_next_sg, y_next_sm = synth_labels(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd)
            self.data.append((X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd, y_util, y_lat, y_next_sg, y_next_sm))
    def __len__(self): return self.N
    def __getitem__(self, i): return self.data[i]

# ---------------------
# 4) Model (single encoder, block-causal)
# ---------------------
class MLP(nn.Module):
    def __init__(self, i, o, h=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i,h), nn.GELU(), nn.Linear(h,o))
    def forward(self,x): return self.net(x)

class OneEncoder(nn.Module):
    """
    한 개의 TransformerEncoder만 사용.
    시점 t의 토큰 순서는: [SG, CG, CGD, SM×M, CM×M, CMD×M]  -> 시점당 3+3M개
    - 분석 모드: 모든 시점 토큰을 동시에 넣고 block-causal mask로 미래 차단
    - 롤아웃 모드: k 스텝에서는 0..k만 실제 토큰, 나머지는 PAD 토큰으로 채우고 같은 마스크 사용
    """
    def __init__(self, T, M, D, SG, SM, CG, CGD, CM, CMD, layers=1, heads=4):
        super().__init__()
        self.T, self.M, self.D = T, M, D
        self.S_PER_T = 3 + 3*M
        # embedders
        self.f_sg  = MLP(SG,  D); self.f_cg  = MLP(CG,  D); self.f_cgd = MLP(CGD, D)
        self.f_sm  = MLP(SM,  D); self.f_cm  = MLP(CM,  D); self.f_cmd = MLP(CMD, D)
        # embeddings
        self.type_emb   = nn.Embedding(7, D)   # 0..5 types, 6=PAD
        self.master_emb = nn.Embedding(M, D)
        self.time_emb   = nn.Embedding(T, D)
        self.pad_token  = nn.Parameter(torch.zeros(1,1,D))
        # single encoder
        enc_layer = nn.TransformerEncoderLayer(d_model=D, nhead=heads, dim_feedforward=128, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        # heads
        self.head_util    = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,96), nn.GELU(), nn.Linear(96,1), nn.Sigmoid())
        self.head_latency = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,96), nn.GELU(), nn.Linear(96,1), nn.Softplus())
        self.head_next_sg = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,96), nn.GELU(), nn.Linear(96,SG))
        self.head_next_sm = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,96), nn.GELU(), nn.Linear(96,SM))

    def _pack(self, X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd):
        Tn, Mn, D = self.T, self.M, self.D
        seq = []
        for t in range(Tn):
            pos = self.time_emb(torch.tensor(t))
            z_sg  = self.f_sg(X_sg[t])   + self.type_emb.weight[0] + pos
            z_cg  = self.f_cg(X_cg[t])   + self.type_emb.weight[1] + pos
            z_cgd = self.f_cgd(X_cgd[t]) + self.type_emb.weight[2] + pos
            ids = torch.arange(Mn)
            idv = self.master_emb(ids)
            z_sm  = self.f_sm(X_sm[t]).view(Mn, D)   + self.type_emb.weight[3] + pos + idv
            z_cm  = self.f_cm(X_cm[t]).view(Mn, D)   + self.type_emb.weight[4] + pos + idv
            z_cmd = self.f_cmd(X_cmd[t]).view(Mn, D) + self.type_emb.weight[5] + pos + idv
            seq.append(torch.cat([z_sg.unsqueeze(0), z_cg.unsqueeze(0), z_cgd.unsqueeze(0), z_sm, z_cm, z_cmd], dim=0))
        return torch.cat(seq, dim=0).unsqueeze(0)  # [1, L, D], L=T*(3+3M)

    def _mask_full(self):
        L = self.T * self.S_PER_T
        mask = torch.full((L, L), float('-inf'))
        for t in range(self.T):
            for k in range(t+1):
                mask[t*self.S_PER_T:(t+1)*self.S_PER_T, k*self.S_PER_T:(k+1)*self.S_PER_T] = 0.0
        return mask

    def _time_indices(self, t):
        base = t * self.S_PER_T
        idx_sg  = base + 0
        idx_cgd = base + 2
        idx_sm0 = base + 3
        idx_cmd0= base + 3 + 2*self.M
        return idx_sg, idx_cgd, idx_sm0, idx_cmd0

    # ---- 분석 모드: 전체 시점 병렬
    def analysis(self, X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd):
        X = self._pack(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd)
        H = self.encoder(X, mask=self._mask_full())
        util = []; lat = []; nsg = []; nsm = []
        for t in range(self.T):
            idx_sg, idx_cgd, idx_sm0, idx_cmd0 = self._time_indices(t)
            h_sg  = H[:, idx_sg, :]
            h_cgd = H[:, idx_cgd, :]
            util.append(self.head_util(h_sg).squeeze(0))           # [1] -> []
            h_sm = H[:, idx_sm0:idx_sm0+self.M, :]
            lat.append(self.head_latency(h_sm).squeeze(0))         # [M,1]
            nsg.append(self.head_next_sg(h_cgd).squeeze(0))        # [SG]
            h_cmd = H[:, idx_cmd0:idx_cmd0+self.M, :]
            nsm.append(self.head_next_sm(h_cmd).squeeze(0))        # [M,SM]
        util = torch.stack(util, dim=0)  # [T,1]
        lat  = torch.stack(lat,  dim=0)  # [T,M,1]
        nsg  = torch.stack(nsg,  dim=0)  # [T,SG]
        nsm  = torch.stack(nsm,  dim=0)  # [T,M,SM]
        return util, lat, nsg, nsm

    # ---- 롤아웃 모드: 프리픽스 k까지 실제 + 나머지 PAD, 같은 인코더/같은 마스크
    def rollout(self, X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd):
        X_sg = X_sg.clone(); X_sm = X_sm.clone()
        X_cg = X_cg.clone(); X_cgd = X_cgd.clone()
        X_cm = X_cm.clone(); X_cmd = X_cmd.clone()

        utils, lats, R_hist = [], [], []
        R = X_sm[0,:,REMAIN_IDX].clone(); R_hist.append(R.numpy().copy())

        for k in range(self.T):
            X_sm[k,:,REMAIN_IDX] = R  # 현재 남은 BW 주입

            # 토큰 빌드: 0..k는 실제, k+1..T-1은 PAD 토큰
            seq = []
            for t in range(self.T):
                if t <= k:
                    pos = self.time_emb(torch.tensor(t))
                    z_sg  = self.f_sg(X_sg[t])   + self.type_emb.weight[0] + pos
                    z_cg  = self.f_cg(X_cg[t])   + self.type_emb.weight[1] + pos
                    z_cgd = self.f_cgd(X_cgd[t]) + self.type_emb.weight[2] + pos
                    ids = torch.arange(self.M); idv = self.master_emb(ids)
                    z_sm  = self.f_sm(X_sm[t]).view(self.M, self.D)   + self.type_emb.weight[3] + pos + idv
                    z_cm  = self.f_cm(X_cm[t]).view(self.M, self.D)   + self.type_emb.weight[4] + pos + idv
                    z_cmd = self.f_cmd(X_cmd[t]).view(self.M, self.D) + self.type_emb.weight[5] + pos + idv
                    step_tok = torch.cat([z_sg.unsqueeze(0), z_cg.unsqueeze(0), z_cgd.unsqueeze(0), z_sm, z_cm, z_cmd], dim=0)
                else:
                    step_tok = self.pad_token.expand(self.S_PER_T, self.D)
                seq.append(step_tok.unsqueeze(0))
            Xtok = torch.cat(seq, dim=0).view(1, self.T*self.S_PER_T, self.D)

            # 마스크: k 시점 이하만 보게
            L = self.T * self.S_PER_T
            mask = torch.full((L, L), float('-inf'))
            for t in range(k+1):
                for kk in range(t+1):
                    mask[t*self.S_PER_T:(t+1)*self.S_PER_T, kk*self.S_PER_T:(kk+1)*self.S_PER_T] = 0.0

            H = self.encoder(Xtok, mask=mask)

            # k 시점 출력 읽기
            idx_sg, idx_cgd, idx_sm0, idx_cmd0 = self._time_indices(k)
            h_sg  = H[:, idx_sg, :]
            h_sm  = H[:, idx_sm0:idx_sm0+self.M, :]
            h_cgd = H[:, idx_cgd, :]
            h_cmd = H[:, idx_cmd0:idx_cmd0+self.M, :]

            util_k = self.head_util(h_sg).squeeze(0).item()
            lat_k  = self.head_latency(h_sm).squeeze(0).detach().numpy()
            utils.append(util_k); lats.append(lat_k)

            # next state → k+1 입력 갱신
            if k < self.T - 1:
                X_sg[k+1] = self.head_next_sg(h_cgd).squeeze(0).detach()
                next_sm = self.head_next_sm(h_cmd).squeeze(0).detach()
                next_sm[:, REMAIN_IDX] = torch.clamp(next_sm[:, REMAIN_IDX], 0.0, 1.0)
                X_sm[k+1] = next_sm
                R = next_sm[:, REMAIN_IDX]; R_hist.append(R.numpy().copy())

        return np.array(utils), np.stack(lats), np.stack(R_hist)

# ---------------------
# 5) Instantiate & data
# ---------------------
model = OneEncoder(T, M, D, SG, SM, CG, CGD, CM, CMD, layers=LAYERS, heads=HEADS).to(device)

train_ds = SocDataset(N=300, rollout_deltas=False)  # Δ=0 (분석 스타일)로 학습
val_ds   = SocDataset(N=40,  rollout_deltas=False)

# collate: 각 batch는 "샘플 튜플들의 리스트"로 그대로 전달
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  collate_fn=lambda b: b)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, collate_fn=lambda b: b)

opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

# ---------------------
# 6) Training loop (teacher forcing on analysis)
# ---------------------
EPOCHS = 3
for ep in range(EPOCHS):
    model.train(); total=0; steps=0
    for batch in train_loader:
        loss = 0.0
        for (X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd, y_util, y_lat, y_next_sg, y_next_sm) in batch:
            util, lat, nsg, nsm = model.analysis(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd)
            l_util = F.mse_loss(util, y_util)
            l_lat  = F.mse_loss(lat,  y_lat)
            l_nsg  = F.mse_loss(nsg[:-1], y_next_sg[:-1])  # 마지막 t는 제외
            l_nsm  = F.mse_loss(nsm[:-1], y_next_sm[:-1])
            # soft physics: remaining ∈ [0,1]
            R_pred = torch.clamp(nsm[:,:,REMAIN_IDX], 0.0, 1.0)
            l_phys = F.mse_loss(R_pred, nsm[:,:,REMAIN_IDX])
            loss = loss + (l_util + l_lat + 0.7*l_nsg + 0.7*l_nsm + 0.1*l_phys)
        loss = loss / len(batch)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); steps += 1

    # validation
    model.eval(); vtotal=0; vsteps=0
    with torch.no_grad():
        for batch in val_loader:
            loss = 0.0
            for (X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd, y_util, y_lat, y_next_sg, y_next_sm) in batch:
                util, lat, nsg, nsm = model.analysis(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd)
                l_util = F.mse_loss(util, y_util)
                l_lat  = F.mse_loss(lat,  y_lat)
                l_nsg  = F.mse_loss(nsg[:-1], y_next_sg[:-1])
                l_nsm  = F.mse_loss(nsm[:-1], y_next_sm[:-1])
                R_pred = torch.clamp(nsm[:,:,REMAIN_IDX], 0.0, 1.0)
                l_phys = F.mse_loss(R_pred, nsm[:,:,REMAIN_IDX])
                loss = loss + (l_util + l_lat + 0.7*l_nsg + 0.7*l_nsm + 0.1*l_phys)
            vtotal += loss.item()/len(batch); vsteps += 1
    print(f"Epoch {ep+1}/{EPOCHS} | train {total/steps:.4f} | val {vtotal/vsteps:.4f}")

# ---------------------
# 7) Quick evaluation: analysis vs rollout on a fresh sample
# ---------------------
# 새 샘플 하나 만들고, 롤아웃은 Δ≠0로 반사실 시나리오
X_sg  = torch.randn(T, SG)
X_sm  = torch.randn(T, M, SM) * 0.3; X_sm[..., REMAIN_IDX] = 1.0
X_cg  = torch.randn(T, CG) * 0.3
X_cm  = torch.randn(T, M, CM) * 0.2
X_cgd = torch.zeros(T, CGD); X_cmd = torch.zeros(T, M, CMD)

X_cgd_roll = torch.randn(T, CGD) * 0.2
X_cmd_roll = torch.randn(T, M, CMD) * 0.2

model.eval()
with torch.no_grad():
    util_ana, lat_ana, nsg_ana, nsm_ana = model.analysis(X_sg, X_sm.clone(), X_cg, X_cgd, X_cm, X_cmd)

utils_roll, lats_roll, R_hist = model.rollout(
    X_sg.clone(), X_sm.clone(), X_cg.clone(), X_cgd_roll.clone(), X_cm.clone(), X_cmd_roll.clone()
)

print("\n=== Analysis (sample) ===")
print("Util (first 5):", np.round(util_ana[:5,0].detach().numpy(), 3))
print("Latency t0 per master:", np.round(lat_ana[0].squeeze(-1).detach().numpy(), 3))

print("\n=== Rollout (counterfactual) ===")
print("Util (first 5):", np.round(utils_roll[:5], 3))
print("Remaining BW final:", np.round(R_hist[-1], 3))
