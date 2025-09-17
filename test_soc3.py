# =========================
# Single-Encoder SOC Performance (Batch-only)
# - GPU 자동 사용
# - analysis_batch / rollout_batch 만 사용
# - Block-causal mask
# - Random synthetic dataset (N=300)
# - Mini-batch training
# - Inference: B=1 또는 B>1 모두 동일 경로
# =========================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib import colors

# ---------------------
# 0) Repro & device
# ---------------------
SEED = 5678
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ---------------------
# 1) Problem sizes
# ---------------------
T   = 8   # time steps per sequence
M   = 6    # masters
SG  = 4    # global stat dim
SM  = 4    # master stat dim (0th = remaining BW)
CG  = 4    # global cfg dim
CGD = 4    # global delta cfg dim
CM  = 4    # per-master cfg dim
CMD = 4    # per-master delta cfg dim
REMAIN_IDX = 0

D      = 128  # token dim (model width)
LAYERS = 2    # Transformer layers
HEADS  = 8    # attention heads

# ---------------------
# 2) Synthetic label generator
# ---------------------
important_idx = torch.tensor([0, 1, 2, 3])
important_w   = torch.tensor([2.0, -1.0, 2.0, -1.0])

W_util_sg  = torch.zeros(SG, 1)
W_util_sg[important_idx, 0] = important_w

W_lat_sm   = torch.zeros(SM, 1)
W_lat_sm[important_idx, 0] = important_w

@torch.no_grad()
def synth_labels(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd):
    """
    Return:
      y_util[T,1], y_lat[T,M,1], y_next_sg[T,SG], y_next_sm[T,M,SM]
    """
    Tn = X_sg.shape[0]
    Mn = X_sm.shape[1]
    y_util = torch.zeros(Tn, 1)
    y_lat  = torch.zeros(Tn, Mn, 1)
    y_next_sg = torch.zeros(Tn, SG)
    y_next_sm = torch.zeros(Tn, Mn, SM)
    for t in range(Tn):
        # util: SG 특정 feature만 영향
        y_util[t] = torch.sigmoid(X_sg[t] @ W_util_sg)
        # latency: SM 특정 feature만 영향(항상 양수 되도록 abs), scale up
        y_lat[t] = torch.abs((X_sm[t] @ W_lat_sm) * 1000.0)
        # next stat: t+1 입력을 정답으로 사용 (teacher forcing)
        if t < Tn-1:
            y_next_sg[t] = X_sg[t + 1]
            y_next_sm[t] = X_sm[t + 1]
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
            X_sg  = torch.rand(T, SG)
            X_sm  = torch.rand(T, M, SM)
            X_sm[..., REMAIN_IDX] = 1.0  # 남은 BW를 1.0에서 시작
            X_cg  = torch.rand(T, CG)
            X_cm  = torch.rand(T, M, CM)
            if rollout_deltas:
                X_cgd = torch.rand(T, CGD) * 0.2
                X_cmd = torch.rand(T, M, CMD) * 0.2
            else:
                X_cgd = torch.zeros(T, CGD)
                X_cmd = torch.zeros(T, M, CMD)
            y_util, y_lat, y_next_sg, y_next_sm = synth_labels(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd)
            self.data.append((X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd, y_util, y_lat, y_next_sg, y_next_sm))
    def __len__(self): return self.N
    def __getitem__(self, i): return self.data[i]

# ---------------------
# 4) Model (single encoder, block-causal, batch-only)
# ---------------------
class MLP(nn.Module):
    def __init__(self, i, o, h=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i,h), nn.GELU(), nn.Linear(h,o))
    def forward(self,x): return self.net(x)

class OneEncoder(nn.Module):
    """
    - 단일 TransformerEncoder
    - 시점 t 토큰 순서: [SG, CG, CGD, SM×M, CM×M, CMD×M]  -> 3+3M개
    - 분석/학습: analysis_batch (모든 시점 + block-causal mask)
    - 롤아웃: rollout_batch (0..k만 실제, k+1..T-1은 PAD + 부분 마스크)
    """
    def __init__(self, T, M, D, SG, SM, CG, CGD, CM, CMD, layers=1, heads=4):
        super().__init__()
        self.T, self.M, self.D = T, M, D
        self.S_PER_T = 3 + 3*M
        # embedders
        self.f_sg  = MLP(SG,  D) 
        self.f_cg  = MLP(CG,  D)
        self.f_cgd = MLP(CGD, D)
        self.f_sm  = MLP(SM,  D)
        self.f_cm  = MLP(CM,  D)
        self.f_cmd = MLP(CMD, D)
        # embeddings
        self.type_emb   = nn.Embedding(7, D)   # 0..5 types, 6=PAD(미사용)
        self.master_emb = nn.Embedding(M, D)
        self.time_emb   = nn.Embedding(T, D)
        # encoder
        enc_layer = nn.TransformerEncoderLayer(d_model=D, nhead=heads, dim_feedforward=4*D, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        # heads
        self.head_util    = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,96), nn.GELU(), nn.Linear(96,1), nn.Sigmoid())
        self.head_latency = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,96), nn.GELU(), nn.Linear(96,1), nn.Softplus())
        self.head_next_sg = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,96), nn.GELU(), nn.Linear(96,SG))
        self.head_next_sm = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,96), nn.GELU(), nn.Linear(96,SM))

    def _build_block_causal_mask(self, device):
        L = self.T * self.S_PER_T
        mask = torch.full((L, L), float('-inf'), device=device)
        for t in range(self.T):
            for k in range(t+1):
                mask[t*self.S_PER_T:(t+1)*self.S_PER_T, k*self.S_PER_T:(k+1)*self.S_PER_T] = 0.0
        return mask

    # ----- (batch) pack
    def _pack_batch(self, X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd):
        # X_* shapes:
        #   X_sg:  [B,T,SG],   X_sm:  [B,T,M,SM],  X_cg:  [B,T,CG]
        #   X_cgd: [B,T,CGD],  X_cm:  [B,T,M,CM],  X_cmd:[B,T,M,CMD]
        B, Tn, M, D = X_sg.size(0), self.T, self.M, self.D
        tokens = []
        dev = self.time_emb.weight.device

        # 타입 임베딩: 1D/2D 두 가지 버전
        type_sg_1d  = self.type_emb.weight[0].view(1, D)     # for [B,D]
        type_cg_1d  = self.type_emb.weight[1].view(1, D)
        type_cgd_1d = self.type_emb.weight[2].view(1, D)

        type_sm_2d  = self.type_emb.weight[3].view(1, 1, D)  # for [B,M,D]
        type_cm_2d  = self.type_emb.weight[4].view(1, 1, D)
        type_cmd_2d = self.type_emb.weight[5].view(1, 1, D)

        ids = torch.arange(M, device=dev)
        idv = self.master_emb(ids).view(1, M, D)  # [1,M,D]

        for t in range(Tn):
            pos_1d = self.time_emb.weight[t].view(1, D)      # for [B,D]
            pos_2d = pos_1d.view(1, 1, D)                    # for [B,M,D]

            # ---- 전역 토큰: [B,D] + [1,D]
            z_sg  = self.f_sg(X_sg[:, t])   + type_sg_1d  + pos_1d   # [B,D]
            z_cg  = self.f_cg(X_cg[:, t])   + type_cg_1d  + pos_1d   # [B,D]
            z_cgd = self.f_cgd(X_cgd[:, t]) + type_cgd_1d + pos_1d   # [B,D]

            # ---- 마스터 토큰: [B,M,D] + [1,1,D] + [1,M,D]
            z_sm  = self.f_sm(X_sm[:, t]).view(B,M,D)   + type_sm_2d  + pos_2d + idv
            z_cm  = self.f_cm(X_cm[:, t]).view(B,M,D)   + type_cm_2d  + pos_2d + idv
            z_cmd = self.f_cmd(X_cmd[:, t]).view(B,M,D) + type_cmd_2d + pos_2d + idv

            step = torch.cat([
                z_sg.unsqueeze(1),   # [B,1,D]
                z_cg.unsqueeze(1),   # [B,1,D]
                z_cgd.unsqueeze(1),  # [B,1,D]
                z_sm,                # [B,M,D]
                z_cm,                # [B,M,D]
                z_cmd                # [B,M,D]
            ], dim=1)  # [B, S_PER_T, D]

            tokens.append(step)
       
        return torch.cat(tokens, dim=1)  # [B, T*S_PER_T, D]

    # ---- 분석(배치)
    def analysis_batch(self, X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB):
        Xtok = self._pack_batch(X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB)       # [B,L,D]
        mask = self._build_block_causal_mask(Xtok.device)                         # [L,L]
        H = self.encoder(Xtok, mask=mask)                                         # [B,L,D]

        B = X_sgB.size(0)
        H = H.view(B, self.T, self.S_PER_T, self.D)
        h_sg   = H[:,:,0,:]                               # [B,T,D]
        h_cgd  = H[:,:,2,:]                               # [B,T,D]
        h_sm   = H[:,:,3:3+self.M,:]                      # [B,T,M,D]
        h_cmd  = H[:,:,3+2*self.M:3+3*self.M,:]           # [B,T,M,D]

        util = self.head_util(h_sg)                       # [B,T,1]
        lat  = self.head_latency(h_sm)                    # [B,T,M,1]
        nsg  = self.head_next_sg(h_cgd)                   # [B,T,SG]
        nsm  = self.head_next_sm(h_cmd)                   # [B,T,M,SM]
        return util, lat, nsg, nsm

    # ---- 롤아웃(배치)
    def rollout_batch(self, X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB):
        """
        Returns:
        utils : [B,T]
        lats  : [B,T,M]
        R_hist: [B,T,M]
        """
        B, Tn, M, D = X_sgB.size(0), self.T, self.M, self.D
        dev = self.time_emb.weight.device

        # clone (덮어쓰기 방지)
        X_sg = X_sgB.clone()
        X_sm = X_smB.clone()
        X_cg = X_cgB.clone()
        X_cgd= X_cgdB.clone()
        X_cm = X_cmB.clone()
        X_cmd= X_cmdB.clone()

        # 초기 남은 BW (배치)
        R = X_sm[:, 0, :, REMAIN_IDX].clone()          # [B,M]
        R_hist = []
        utils = []
        lats  = []

        # 미리 타입/마스터 임베딩 준비
        type_sg_1d  = self.type_emb.weight[0].view(1, D)
        type_cg_1d  = self.type_emb.weight[1].view(1, D)
        type_cgd_1d = self.type_emb.weight[2].view(1, D)

        type_sm_2d  = self.type_emb.weight[3].view(1, 1, D)
        type_cm_2d  = self.type_emb.weight[4].view(1, 1, D)
        type_cmd_2d = self.type_emb.weight[5].view(1, 1, D)

        ids = torch.arange(M, device=dev)
        idv = self.master_emb(ids).view(1, M, D)   # [1,M,D]

        for k in range(Tn):
            # 현재 step k의 remaining BW 주입
            X_sm[:, k, :, REMAIN_IDX] = R          # [B,M]

            # ----- prefix (0..k) 토큰만 빌드 -----
            blocks = []
            for t in range(k+1):
                pos_1d = self.time_emb.weight[t].view(1, D)
                pos_2d = pos_1d.view(1, 1, D)
                z_sg  = self.f_sg(X_sg[:,t])   + type_sg_1d  + pos_1d     # [B,D]
                z_cg  = self.f_cg(X_cg[:,t])   + type_cg_1d  + pos_1d     # [B,D]
                z_cgd = self.f_cgd(X_cgd[:,t]) + type_cgd_1d + pos_1d     # [B,D]
                z_sm  = self.f_sm(X_sm[:,t]).view(B,M,D)   + type_sm_2d + pos_2d + idv
                z_cm  = self.f_cm(X_cm[:,t]).view(B,M,D)   + type_cm_2d + pos_2d + idv
                z_cmd = self.f_cmd(X_cmd[:,t]).view(B,M,D) + type_cmd_2d + pos_2d + idv
                step = torch.cat([z_sg.unsqueeze(1), z_cg.unsqueeze(1), z_cgd.unsqueeze(1),
                              z_sm, z_cm, z_cmd], dim=1)        # [B,S_PER_T,D]
                blocks.append(step)

            # prefix 시퀀스
            Xtok = torch.cat(blocks, dim=1)  # [B, (k+1)*S_PER_T, D]
            Lk = (k+1) * self.S_PER_T

            # ----- prefix용 block-causal mask (slice) -----
            # 풀마스크 만들기보다, 여기서 직접 만들거나, 빌드해둔 풀마스크를 [:Lk,:Lk]로 슬라이스해도 됨
            mask = torch.full((Lk, Lk), float('-inf'), device=dev)
            # strictly causal: 각 시점 블록이 과거 블록들만 보도록
            for t in range(k+1):
                for kk in range(t+1):
                    mask[t*self.S_PER_T:(t+1)*self.S_PER_T, kk*self.S_PER_T:(kk+1)*self.S_PER_T] = 0.0
            
            H = self.encoder(Xtok, mask=mask)  # [B,Lk,D]

            # k 시점 출력 읽기 (prefix 기준 index)
            base = k * self.S_PER_T
            idx_sg  = base + 0
            idx_cgd = base + 2
            idx_sm0 = base + 3
            idx_cmd0= base + 3 + 2*M

            h_sg  = H[:, idx_sg, :]                 # [B,D]
            h_sm  = H[:, idx_sm0:idx_sm0+M, :]      # [B,M,D]
            h_cgd = H[:, idx_cgd, :]                # [B,D]
            h_cmd = H[:, idx_cmd0:idx_cmd0+M, :]    # [B,M,D]

            util_k = self.head_util(h_sg).squeeze(-1)   # [B]
            lat_k  = self.head_latency(h_sm).squeeze(-1)# [B,M]

            utils.append(util_k)
            lats.append(lat_k)
            R_hist.append(X_sm[:, k, :, REMAIN_IDX].clone())  # [B,M]

            # next state → k+1 입력 갱신 (배치)
            if k < Tn - 1:
                X_sg[:, k+1] = self.head_next_sg(h_cgd)                 # [B,SG]
                next_sm = self.head_next_sm(h_cmd)                       # [B,M,SM]
                next_sm[:,:,REMAIN_IDX] = torch.clamp(next_sm[:,:,REMAIN_IDX], 0.0, 1.0)
                X_sm[:, k+1] = next_sm
                R = next_sm[:,:,REMAIN_IDX]                              # [B,M]

        utils = torch.stack(utils, dim=1)        # [B,T]
        lats  = torch.stack(lats,  dim=1)        # [B,T,M]
        R_hist= torch.stack(R_hist, dim=1)       # [B,T,M]
        return utils, lats, R_hist

# ---------------------
# 5) Instantiate & data
# ---------------------
model = OneEncoder(T, M, D, SG, SM, CG, CGD, CM, CMD, layers=LAYERS, heads=HEADS).to(device)

print("Make dataset...")
full_ds = SocDataset(N=1000, rollout_deltas=False)   # 필요시 1000으로 변경
n_total = len(full_ds)
n_train = int(n_total * 0.8)
n_val   = n_total - n_train
train_ds, val_ds = random_split(full_ds, [n_train, n_val])

def collate_stack(batch):
    cols = list(zip(*batch))
    return [torch.stack(c, dim=0) for c in cols]

pin = (device.type == "cuda")
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  collate_fn=collate_stack, pin_memory=pin)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, collate_fn=collate_stack, pin_memory=pin)

opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

# ---------------------
# 6) Training loop (teacher forcing on analysis; batched)
# ---------------------
class MAPELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, preds, targets):
        return torch.mean(torch.abs((targets - preds) / (targets.abs() + self.eps)))

criterion = MAPELoss()

def to_device_batch(batch, device, non_blocking=True):
    return [x.to(device, non_blocking=non_blocking) for x in batch]

EPOCHS = 2  # 데모: 작게, 성능 확인 후 늘리세요
for ep in range(EPOCHS):
    model.train()
    total=0
    steps=0
    total_util = 0
    total_lat = 0
    total_nsg = 0
    total_nsm = 0
    for batch in train_loader:
        X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB, y_utilB, y_latB, y_nsgB, y_nsmB = to_device_batch(batch, device)
        util, lat, nsg, nsm = model.analysis_batch(X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB)

        l_util = criterion(util, y_utilB)
        l_lat  = criterion(lat,  y_latB)
        l_nsg  = criterion(nsg[:,:-1], y_nsgB[:,:-1])  # 마지막 t 제외
        l_nsm  = criterion(nsm[:,:-1], y_nsmB[:,:-1])

        loss = l_util + l_lat + l_nsg + l_nsm
        opt.zero_grad(); loss.backward(); opt.step()

        total += loss.item()
        steps += 1
        
        total_util += l_util.item()
        total_lat += l_lat.item()
        total_nsg += l_nsg.item()
        total_nsm += l_nsm.item()

    # validation
    model.eval(); vtotal=0; vsteps=0
    with torch.no_grad():
        for batch in val_loader:
            X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB, y_utilB, y_latB, y_nsgB, y_nsmB = to_device_batch(batch, device)
            util, lat, nsg, nsm = model.analysis_batch(X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB)

            l_util = criterion(util, y_utilB)
            l_lat  = criterion(lat,  y_latB)
            l_nsg  = criterion(nsg[:,:-1], y_nsgB[:,:-1])
            l_nsm  = criterion(nsm[:,:-1], y_nsmB[:,:-1])

            vtotal += (l_util + l_lat + l_nsg + l_nsm).item(); vsteps += 1

    print(f"Epoch {ep+1}/{EPOCHS} | train {total/steps:.4f} util {total_util/steps:.4f} lat {total_lat/steps:.4f} SG {total_nsg/steps:.4f} SM {total_nsm/steps:.4f} | val {vtotal/vsteps:.4f}")

# ============================================================
# 7) IG (Integrated Gradients) : 통합 축(모든 설정/스탯)
# ============================================================
def build_feature_axis_labels(SG, CG, CGD, M, SM, CM, CMD):
    labels = []
    for i in range(SG):  labels.append(f"SG[{i}]")
    for i in range(CG):  labels.append(f"CG[{i}]")
    for i in range(CGD): labels.append(f"CGD[{i}]")
    for m in range(M):
        for i in range(SM):  labels.append(f"SM[{m}][{i}]")
    for m in range(M):
        for i in range(CM):  labels.append(f"CM[{m}][{i}]")
    for m in range(M):
        for i in range(CMD): labels.append(f"CMD[{m}][{i}]")
    return labels

def _make_baselines_like(inputs, mode="zero"):
    outs = []
    for x in inputs:
        if mode == "zero":
            outs.append(torch.zeros_like(x))
        elif mode == "mean":
            if x.ndim >= 2:
                mean = x.mean(dim=(0,1), keepdim=True)
                outs.append(mean.expand_as(x))
            else:
                outs.append(torch.zeros_like(x))
        else:
            outs.append(torch.zeros_like(x))
    return outs

def _integrated_gradients(model, inputs, forward_fn, steps=64, baseline_mode="zero"):
    baselines = _make_baselines_like(inputs, mode=baseline_mode)
    deltas = [x - b for x, b in zip(inputs, baselines)]
    grads_sum = [torch.zeros_like(x) for x in inputs]

    for s in range(1, steps+1):
        alpha = float(s) / steps
        x_interp = [b + alpha * d for b, d in zip(baselines, deltas)]
        for xi in x_interp:
            xi.requires_grad_(True)
        out_scalar = forward_fn(x_interp)  # scalar
        grads = torch.autograd.grad(out_scalar, x_interp, retain_graph=False, create_graph=False)
        grads_sum = [gs + g for gs, g in zip(grads_sum, grads)]

    attributions = [(gs / steps) * d for gs, d in zip(grads_sum, deltas)]
    return attributions  # list aligned to inputs

def concat_attrs_timewise(A_sg, A_sm, A_cg, A_cgd, A_cm, A_cmd, signed=True):
    if not signed:
        A_sg = A_sg.abs(); A_sm = A_sm.abs(); A_cg = A_cg.abs()
        A_cgd = A_cgd.abs(); A_cm = A_cm.abs(); A_cmd = A_cmd.abs()

    B, Tn, SGd = A_sg.shape
    M  = A_sm.shape[2]
    SMd= A_sm.shape[-1]
    CGd = A_cg.shape[-1]
    CGDd= A_cgd.shape[-1]
    CMd = A_cm.shape[-1]
    CMDd= A_cmd.shape[-1]

    Dtot = SGd + CGd + CGDd + M*SMd + M*CMd + M*CMDd
    mat = torch.zeros(Tn, Dtot, device=A_sg.device)
    for t in range(Tn):
        chunks = [
            A_sg[0, t],             # [SG]
            A_cg[0, t],             # [CG]
            A_cgd[0, t],            # [CGD]
            A_sm[0, t].reshape(-1), # [M*SM]
            A_cm[0, t].reshape(-1), # [M*CM]
            A_cmd[0, t].reshape(-1) # [M*CMD]
        ]
        mat[t] = torch.cat(chunks, dim=0)
    return mat  # [T, Dtot]

def ig_util_combined_matrix(model, X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB,
                            steps=64, baseline_mode="zero", signed=True):
    model.eval()
    Tn = X_sgB.size(1)
    mats = []
    for t in range(Tn):
        inputs = [X_sgB.clone().detach(), X_smB.clone().detach(),
                  X_cgB.clone().detach(), X_cgdB.clone().detach(),
                  X_cmB.clone().detach(), X_cmdB.clone().detach()]
        def forward_fn(x_list):
            X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd = x_list
            util, _, _, _ = model.analysis_batch(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd)
            return util[0, t, 0]
        A_sg, A_sm, A_cg, A_cgd, A_cm, A_cmd = _integrated_gradients(model, inputs, forward_fn, steps=steps, baseline_mode=baseline_mode)
        mat_t = concat_attrs_timewise(A_sg, A_sm, A_cg, A_cgd, A_cm, A_cmd, signed=signed)
        mats.append(mat_t[t].unsqueeze(0))
    mat_all = torch.cat(mats, dim=0)  # [T, Dtot]
    return mat_all

def ig_latency_combined_matrices_all_masters(model, X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB,
                                             steps=48, baseline_mode="zero", signed=True):
    model.eval()
    M = X_smB.size(2)
    mats_per_m = []
    for m in range(M):
        Tn = X_sgB.size(1)
        mats = []
        for t in range(Tn):
            inputs = [X_sgB.clone().detach(), X_smB.clone().detach(),
                      X_cgB.clone().detach(), X_cgdB.clone().detach(),
                      X_cmB.clone().detach(), X_cmdB.clone().detach()]
            def forward_fn(x_list, tt=t, mm=m):
                X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd = x_list
                _, lat, _, _ = model.analysis_batch(X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd)
                return lat[0, tt, mm, 0]
            A_sg, A_sm, A_cg, A_cgd, A_cm, A_cmd = _integrated_gradients(model, inputs, lambda z: forward_fn(z, t, m), steps=steps, baseline_mode=baseline_mode)
            mat_t = concat_attrs_timewise(A_sg, A_sm, A_cg, A_cgd, A_cm, A_cmd, signed=signed)
            mats.append(mat_t[t].unsqueeze(0))
        mats_per_m.append(torch.cat(mats, dim=0))  # [T, Dtot]
    return mats_per_m

# ---------------------
# 8) Plot (save to files; background)
# ---------------------
def save_util_heatmap(matrix_TxD, labels, out_path="util_ig_heatmap.png", title="Utilization IG Heatmap"):
    data = matrix_TxD.detach().cpu().numpy() if torch.is_tensor(matrix_TxD) else matrix_TxD
    vmax = np.nanmax(np.abs(data))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    plt.figure(figsize=(10, max(4, 0.04*len(labels) + 3)))
    im = plt.imshow(data.T, aspect='auto', origin='lower',
                    cmap='RdYlGn', norm=norm)  # + green, - red
    cbar = plt.colorbar(im, label='IG attribution (+ green, − red)')
    cbar.ax.set_ylabel('IG attribution (+ green, − red)', rotation=90, va='center')
    plt.xlabel('time')
    plt.ylabel('features (stats & configs)')
    plt.title(title)
    step = max(1, len(labels) // 30)
    idxs = np.arange(0, len(labels), step)
    plt.yticks(idxs, [labels[i] for i in idxs], fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")

def save_latency_heatmaps_grid(mats_per_m, labels, cols=4, out_path="latency_ig_heatmaps_grid.png", title_prefix="Latency IG"):
    M = len(mats_per_m)
    rows = int(np.ceil(M / cols))
    all_data = np.concatenate([ (m.detach().cpu().numpy() if torch.is_tensor(m) else m) for m in mats_per_m ], axis=0)
    vmax = np.nanmax(np.abs(all_data))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig = plt.figure(figsize=(4.8*cols, 3.2*rows + 0.04*len(labels)))
    for m, mat in enumerate(mats_per_m):
        mat_np = mat.detach().cpu().numpy() if torch.is_tensor(mat) else mat
        ax = plt.subplot(rows, cols, m+1)
        im = ax.imshow(mat_np.T, aspect='auto', origin='lower',
                       cmap='RdYlGn_r', norm=norm)  # + red, - green
        ax.set_title(f"{title_prefix} (m={m})")
        ax.set_xlabel('time')
        if m % cols == 0:
            step = max(1, len(labels) // 20)
            idxs = np.arange(0, len(labels), step)
            ax.set_yticks(idxs)
            ax.set_yticklabels([labels[i] for i in idxs], fontsize=7)
        else:
            ax.set_yticks([])
    # 공용 컬러바
    cbar = fig.colorbar(im, ax=fig.axes, orientation='horizontal', fraction=0.035, pad=0.03)
    cbar.set_label('IG attribution (+ red, − green)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[saved] {out_path}")

# ---------------------
# 9) Top/Bottom5 printer
# ---------------------
def print_top_bottom_over_time(matrix_TxD, labels, k=5, tag="Util"):
    """
    matrix_TxD: [T, Dtot] (signed IG)
    각 time t에 대해 상위/하위 k개 피처와 값을 출력
    """
    data = matrix_TxD.detach().cpu().numpy() if torch.is_tensor(matrix_TxD) else matrix_TxD
    Tn, Dtot = data.shape
    print(f"\n==== {tag}: Top{k} / Bottom{k} per time ====")
    for t in range(Tn):
        row = data[t]
        # Top-k positive
        top_idx = np.argsort(-row)[:k]
        # Bottom-k negative
        bot_idx = np.argsort(row)[:k]
        top = [(labels[i], float(row[i])) for i in top_idx]
        bot = [(labels[i], float(row[i])) for i in bot_idx]
        print(f"[t={t}] Top{k}: {top}")
        print(f"[t={t}] Bottom{k}: {bot}")

def print_top_bottom_latency_all_masters(mats_per_m, labels, k=5):
    """
    mats_per_m: list of [T, Dtot] (signed)
    각 master m, time t에 대해 Top/Bottom k 출력
    """
    print(f"\n==== Latency: Top{ k } / Bottom{ k } per time & master ====")
    for m, mat in enumerate(mats_per_m):
        data = mat.detach().cpu().numpy() if torch.is_tensor(mat) else mat
        Tn, Dtot = data.shape
        print(f"\n-- Master m={m} --")
        for t in range(Tn):
            row = data[t]
            top_idx = np.argsort(-row)[:k]
            bot_idx = np.argsort(row)[:k]
            top = [(labels[i], float(row[i])) for i in top_idx]
            bot = [(labels[i], float(row[i])) for i in bot_idx]
            print(f"[t={t}] Top{k}: {top}")
            print(f"[t={t}] Bottom{k}: {bot}")

# ============================================================
# 10) Inference + IG + Save + Print
# ============================================================
@torch.no_grad()
def make_one_sample(device):
    X_sg  = torch.rand(T, SG, device=device)
    X_sm  = torch.rand(T, M, SM, device=device); X_sm[..., REMAIN_IDX] = 1.0
    X_cg  = torch.rand(T, CG, device=device)
    X_cm  = torch.rand(T, M, CM, device=device)
    X_cgd = torch.zeros(T, CGD, device=device)
    X_cmd = torch.zeros(T, M, CMD, device=device)
    return X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd

# 샘플 1개로 분석/IG
model.eval()
X_sg, X_sm, X_cg, X_cgd, X_cm, X_cmd = make_one_sample(device)
with torch.no_grad():
    utilB, latB, nsgB, nsmB = model.analysis_batch(
        X_sg.unsqueeze(0), X_sm.unsqueeze(0), X_cg.unsqueeze(0), X_cgd.unsqueeze(0),
        X_cm.unsqueeze(0), X_cmd.unsqueeze(0)
    )
print("\n=== Quick Analysis (B=1) ===")
print("Util t0..4:", np.round(utilB[0,:5,0].detach().cpu().numpy(), 3))
print("Lat  t0 m0..3:", np.round(latB[0,0,:4,0].detach().cpu().numpy(), 3))

# IG 계산
X_sgB  = X_sg.unsqueeze(0)
X_smB  = X_sm.unsqueeze(0)
X_cgB  = X_cg.unsqueeze(0)
X_cgdB = X_cgd.unsqueeze(0)
X_cmB  = X_cm.unsqueeze(0)
X_cmdB = X_cmd.unsqueeze(0)
labels_all = build_feature_axis_labels(SG, CG, CGD, M, SM, CM, CMD)

util_mat = ig_util_combined_matrix(model, X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB,
                                   steps=64, baseline_mode="zero", signed=True)
lat_mats = ig_latency_combined_matrices_all_masters(model, X_sgB, X_smB, X_cgB, X_cgdB, X_cmB, X_cmdB,
                                                    steps=48, baseline_mode="zero", signed=True)

# 파일 저장(백그라운드)
save_util_heatmap(util_mat, labels_all, out_path="util_ig_heatmap.png",
                  title="Utilization IG (signed: + green, − red)")
save_latency_heatmaps_grid(lat_mats, labels_all, cols=4,
                           out_path="latency_ig_heatmaps_grid.png",
                           title_prefix="Latency IG (signed: + red, − green)")

# Top5 / Bottom5 출력
print_top_bottom_over_time(util_mat, labels_all, k=5, tag="Util")
print_top_bottom_latency_all_masters(lat_mats, labels_all, k=5)

print("\nDone. Check the saved files: 'util_ig_heatmap.png', 'latency_ig_heatmaps_grid.png'")