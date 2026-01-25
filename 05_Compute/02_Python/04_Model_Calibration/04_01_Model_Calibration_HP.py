#!/usr/bin/env python
# coding: utf-8

# # 1. Import Data

# In[ ]:


# Global Import and Directories
import sys
sys.path.append("STGNN/02_Code")
import Setup
from Setup import *


# In[ ]:


# =============================
# a) Parameters
# =============================

SEED = 42
ATTN_HEADS_DEFAULT = 4
EMBED_DIM = 32
HIDDEN_DIM = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 10
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

WINDOW_SIZE = 672 # high value due to different encoder
HORIZON_STEPS = 4  # Horizont per Target
LENGTH_ENCODER_DEFAULT = 16 # Target Length after Strides


@dataclass
class STWindowDataMulti:
    x: torch.Tensor
    x_mask: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    time_feat: np.ndarray
    train_end_idx: np.ndarray
    val_end_idx: np.ndarray
    test_end_idx: np.ndarray
    valid_nodes: np.ndarray
    target_cols: List[str]
    condat: Optional[torch.Tensor] = None
        
@dataclass
class STFeatureFlags:
    use_attention: bool = True
    use_spatial: bool = True
    use_time: bool = True
    use_temp_day: bool = True
    use_temp_week: bool = True
    use_condat: bool = True
        
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# -------------------------------------------------
# 1) Temporaler Downsampling-Encoder (Conv1d)
# -------------------------------------------------

class TemporalDownsamplingEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        kernel1: int = 3,
        stride1: int = 1,
        kernel2: int = 3,
        stride2: int = 2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=mid_dim,
            kernel_size=kernel1,
            stride=stride1,
            padding=kernel1 // 2,
        )
        self.bn1 = nn.BatchNorm1d(mid_dim)

        self.conv2 = nn.Conv1d(
            in_channels=mid_dim,
            out_channels=out_dim,
            kernel_size=kernel2,
            stride=stride2,
            padding=0,
        )
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, N, F]
        """
        T, N, F = x.shape

        x_ = x.permute(1, 2, 0)   # [N, F, T]

        x_ = self.conv1(x_)       # [N, mid_dim, T']
        x_ = self.bn1(x_)
        x_ = self.act(x_)

        x_ = self.conv2(x_)       # [N, out_dim, T_out]
        x_ = self.bn2(x_)
        x_ = self.act(x_)

        x_ = x_.permute(2, 0, 1)  # [T_out, N, out_dim]
        return x_


# -------------------------------------------------
# 2) Multi-Scale Temporaler Encoder
# -------------------------------------------------

def _downsample_stride(win: int, out_len: int) -> int:
    if win % out_len != 0:
        raise ValueError(f"win={win} must be divisible by out_len={out_len}.")
    return win // out_len


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim_per_scale_week: int,
        out_dim_per_scale_day: int,
        win_day: int,
        win_week: int,
        length_encoder: int = 16, 
        use_day: bool = True,
        use_week: bool = True,
    ):
        super().__init__()
        self.win_day = win_day
        self.win_week = win_week
        self.length_encoder = length_encoder
        self.use_day = use_day
        self.use_week = use_week
        
        self.enc_day = None
        self.enc_week = None

        if self.use_day:
            stride_day = _downsample_stride(win_day, length_encoder)
            self.enc_day = TemporalDownsamplingEncoder(
                in_dim=in_dim,
                mid_dim=hidden_dim,
                out_dim=out_dim_per_scale_day,
                kernel1=3,
                stride1=1,
                kernel2=stride_day,
                stride2=stride_day,
            )

        if self.use_week:
            stride_week = _downsample_stride(win_week, length_encoder)
            self.enc_week = TemporalDownsamplingEncoder(
                in_dim=in_dim,
                mid_dim=hidden_dim,
                out_dim=out_dim_per_scale_week,
                kernel1=3,
                stride1=1,
                kernel2=stride_week,
                stride2=stride_week,
            )

        if (self.enc_day is None) and (self.enc_week is None):
            raise ValueError("MultiScaleTemporalEncoder: at least one of use_day/use_week must be True.")

    @property
    def out_dim(self) -> int:
        out = 0
        if self.enc_day is not None:
            out += self.enc_day.conv2.out_channels
        if self.enc_week is not None:
            out += self.enc_week.conv2.out_channels
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [W_global, N, F]
        """
        W_global, N, F = x.shape

        outs = []

        if self.enc_day is not None:
            if self.win_day > W_global:
                raise ValueError(f"win_day={self.win_day} > W_global={W_global}")
            x_day = x[-self.win_day:]
            h_day = self.enc_day(x_day)  # [P_day, N, C_day]
            outs.append(h_day)

        if self.enc_week is not None:
            if self.win_week > W_global:
                raise ValueError(f"win_week={self.win_week} > W_global={W_global}")
            x_week = x[-self.win_week:]
            h_week = self.enc_week(x_week)  # [P_week, N, C_week]
            outs.append(h_week)

        if len(outs) == 1:
            return outs[0]

        # zwei aktiv -> auf gemeinsame Länge trimmen
        P = min(outs[0].size(0), outs[1].size(0))
        outs = [h[-P:] for h in outs]
        return torch.cat(outs, dim=-1)  # [P, N, C_day + C_week]


# -------------------------------------------------
# 3) Local spatial convolution
# -------------------------------------------------

class LocalSpatialEncoder(nn.Module):
    def __init__(self, hidden_dim: int, cheb_K: int = 3, n_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            [ChebConv(hidden_dim, hidden_dim, K=cheb_K) for _ in range(n_layers)]
        )
        self.act = nn.ReLU()
   
    def forward(
        self,
        h_seq: torch.Tensor,              # [P, N, H]
        edge_index: torch.Tensor,         # [2, E]
        edge_weight: Optional[torch.Tensor] = None,  # [E]
    ) -> torch.Tensor:
        P, N, H = h_seq.shape
        out_list = []

        for t in range(P):
            ht = h_seq[t]  # [N, H]
            for li, conv in enumerate(self.layers):
                h_in = ht
                h_sp = conv(ht, edge_index, edge_weight=edge_weight)

                # Aktivierung nur zwischen Layers
                if li != len(self.layers) - 1:
                    h_sp = self.act(h_sp)

                ht = h_in + h_sp  # Residual

            out_list.append(ht)

        return torch.stack(out_list, dim=0)

# -------------------------------------------------
# 4) STGNN-Modell
# -------------------------------------------------

class TemporalAttnPoolBlock(nn.Module):
    def __init__(self, hidden_dim: int, attn_heads: int, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=attn_heads, batch_first=False, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_mult * hidden_dim, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q:  [1, N, H]
        kv: [P, N, H]
        """
        attn_out, _ = self.attn(q, kv, kv)     # [1, N, H]
        q = self.ln1(q + attn_out)             # Residual

        ff_out = self.ff(q)                    # [1, N, H]
        q = self.ln2(q + ff_out)               # Residual
        return q


class STGNNWindowMultiRegressor(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embed_dim: int,
        hidden_dim: int,
        out_dim: int,
        in_dim: int = 1,
        gnn_type: str = "sage",
        temporal_type: str = "gru",
        spatial_pool: str = "mean",
        temp_hidden_dim: int = 32,
        temp_out_per_scale_week: int = 16,
        temp_out_per_scale_day: int = 16,
        time_in_dim: int = 0,
        cond_in_dim: int = 0,
        win_hour: int = 16,
        win_day: int = 64,
        win_week: int = 128,
        use_attention: bool = True,
        use_spatial: bool = True,
        use_time: bool = True,
        use_temp_day: bool = True,
        use_temp_week: bool = True,
        attn_heads: int = 4,  
        cheb_K: int = 3,
        cheb_layers: int = 1,
        length_encoder: int = 16,
        temporal_attn_layers: int = 1,
    ):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_spatial = use_spatial
        self.use_time = use_time
        self.use_temp_day = use_temp_day
        self.use_temp_week = use_temp_week
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.time_in_dim = time_in_dim
        self.cond_in_dim = cond_in_dim
        self.win_hour = win_hour
        self.win_day = win_day
        self.win_week = win_week

        self.emb = nn.Embedding(num_nodes, embed_dim)
        self.act = nn.ReLU()
        
        in_dim_eff = 2 * in_dim
        
        self.temp_enc = None
        self.temp_out_dim = 0

        if self.use_temp_day or self.use_temp_week:
            self.temp_enc = MultiScaleTemporalEncoder(
                in_dim=in_dim_eff + embed_dim,
                hidden_dim=temp_hidden_dim,
                out_dim_per_scale_week=temp_out_per_scale_week,
                out_dim_per_scale_day=temp_out_per_scale_day,
                win_day=win_day,
                win_week=win_week,
                length_encoder=length_encoder,
                use_day=self.use_temp_day,
                use_week=self.use_temp_week,
            )
            self.temp_out_dim = self.temp_enc.out_dim


        self.hour_proj = nn.Linear(in_dim_eff  + embed_dim, hidden_dim)
        self.spatial_enc = None
        if self.use_spatial:
            self.spatial_enc = LocalSpatialEncoder(hidden_dim=hidden_dim, cheb_K=cheb_K, n_layers=cheb_layers)

        self.spatial_out_dim = hidden_dim  # bleibt hidden_dim, weil "no spatial" dann identity ist


        self.core_st_dim = self.spatial_out_dim + self.temp_out_dim

        if self.use_time and time_in_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.core_st_dim),
                nn.ReLU(),
            )
        else:
            self.time_mlp = None

        if cond_in_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.core_st_dim),
                nn.ReLU(),
            )
        else:
            self.cond_mlp = None

        n_blocks = 1
        if self.time_mlp is not None:
            n_blocks += 1
        if self.cond_mlp is not None:
            n_blocks += 1

        self.st_proj = nn.Linear(n_blocks * self.core_st_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        
        # Self-attention pooling at the end
        self.temporal_attn_layers = temporal_attn_layers

        if self.use_attention:
            self.global_query = nn.Parameter(torch.randn(1, hidden_dim))

            self.attn_blocks = nn.ModuleList([
                TemporalAttnPoolBlock(hidden_dim=hidden_dim, attn_heads=attn_heads)
                for _ in range(temporal_attn_layers)
            ])


        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        x_window: torch.Tensor,              # [W, N, F]
        x_mask: Optional[torch.Tensor],      # [W, N, F]
        edge_index: torch.Tensor,            # [2, E]
        edge_weight: Optional[torch.Tensor] = None, 
        time_feat: Optional[torch.Tensor] = None,  # [W, time_in_dim]
        condat: Optional[torch.Tensor] = None,     # [N, cond_in_dim]
    ) -> torch.Tensor:
        
        if x_mask is None:
            raise ValueError("x_mask must be provided (1=valid, 0=missing).")

        x_mask = x_mask.to(dtype=x_window.dtype, device=x_window.device)
    
        # Features erweitern: [W,N,F] -> [W,N,2F]
        x_in = torch.cat([x_window, x_mask], dim=-1)

        device = x_window.device
        W, N, F2 = x_in.shape

        node_emb = self.emb.weight.to(device)                        # [N, embed_dim]
        node_emb_exp = node_emb.unsqueeze(0).expand(W, -1, -1)       # [W, N, embed_dim]
        x_with_emb = torch.cat([x_in, node_emb_exp], dim=-1)         # [W, N, F+embed_dim]

        if self.win_hour > W:
            raise ValueError(f"win_hour={self.win_hour} > Window Size W={W}")
        x_hour = x_with_emb[-self.win_hour:]                         # [Wh, N, F+emb]

        h_hour = self.hour_proj(x_hour)                              # [Wh, N, hidden_dim]
        h_hour = self.act(h_hour)

        if self.spatial_enc is None:
            h_spat_hour = h_hour
        else:
            h_spat_hour = self.spatial_enc(h_hour, edge_index, edge_weight=edge_weight)


        if self.temp_enc is None:
            P = h_spat_hour.size(0)
            st_core = h_spat_hour  # [P, N, hidden_dim]
        else:
            st_tw = self.temp_enc(x_with_emb)  # [P_tw, N, temp_out_dim]
            P = min(h_spat_hour.size(0), st_tw.size(0))
            h_spat_hour = h_spat_hour[-P:]
            st_tw = st_tw[-P:]
            st_core = torch.cat([h_spat_hour, st_tw], dim=-1)  # [P, N, hidden_dim + temp_out_dim]

        blocks = [st_core]

        if self.time_mlp is not None:
            if time_feat is None:
                raise ValueError("time_feat cannot be none if time_in_dim > 0.")

            if time_feat.size(0) < P:
                raise ValueError(f"time_feat too short: {time_feat.size(0)} < P={P}")

            time_feat = time_feat[-P:]             # [P, time_in_dim]

            t_emb = self.time_mlp(time_feat.to(device))   # [P, core_st_dim]
            t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # [P, N, core_st_dim]
            blocks.append(t_emb)

        if self.cond_mlp is not None:
            if condat is None:
                raise ValueError("condat cannot be none if cond_in_dim > 0.")
            if condat.size(0) != N:
                raise ValueError(f"condat expects N={N}, receives {condat.size(0)}.")
            c_emb = self.cond_mlp(condat.to(device))                 # [N, core_st_dim]
            c_emb = c_emb.unsqueeze(0).expand(P, -1, -1)             # [P, N, core_st_dim]
            blocks.append(c_emb)

        st_cat = torch.cat(blocks, dim=-1)                           # [P, N, n_blocks * core_st_dim]
        h_seq = self.st_proj(st_cat)                                 # [P, N, hidden_dim]
        h_seq = self.act(h_seq)

        out_seq, h_n = self.gru(h_seq)          # out_seq: [P, N, H], h_n: [1, N, H]

        if self.use_attention:
            P, N, H = out_seq.shape
            q = self.global_query.unsqueeze(1).expand(-1, N, -1)  # [1, N, H]

            for block in self.attn_blocks:
                q = block(q, out_seq)  # bleibt [1, N, H]

            h_final = q[0]  # [N, H]
        else:
            h_final = h_n[-1]

        out = self.head(h_final)          # [N, out_dim]
        return out



# =============================
# e) Training and evaluation (multi-step, multi-target)
# =============================

@dataclass
class STTrainConfig:
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    epochs: int = EPOCHS
    patience: int = PATIENCE
    horizon_steps: int = HORIZON_STEPS  # per Target

    # Modell-Hyperparameter (tunable)
    embed_dim: int = EMBED_DIM
    hidden_dim: int = HIDDEN_DIM
    temp_hidden_dim: int = 32
    temp_out_per_scale_week: int = 16
    temp_out_per_scale_day: int = 16
    win_hour: int = 16
    win_day: int = 64
    win_week: int = 128
    cheb_K: int = 3
    cheb_layers: int = 1
    length_encoder: int = LENGTH_ENCODER_DEFAULT
    attn_heads: int = ATTN_HEADS_DEFAULT
    temporal_attn_layers: int = 1


def _compute_window_metrics_multi_per_horizon(
    model: nn.Module,
    data: STWindowDataMulti,
    end_indices: np.ndarray,
    device: torch.device,
    loss_nodes: torch.Tensor,
    eval_horizons: List[int],
    time_full: torch.Tensor,
    condat_full: Optional[torch.Tensor],
) -> Tuple[dict, float]:

    x_full = data.x.to(device)
    x_mask_full = data.x_mask.to(device)
    y_full = data.y.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device)
    time_full = time_full.to(device)
    W = WINDOW_SIZE  

    out_dim = y_full.shape[2]      # H * M
    M = len(data.target_cols)
    H_tot = out_dim // M

    sum_abs = {h: 0.0 for h in eval_horizons}
    sum_sq  = {h: 0.0 for h in eval_horizons}
    sum_pct = {h: 0.0 for h in eval_horizons}
    count   = {h: 0   for h in eval_horizons}

    model.eval()
    with torch.no_grad():
        for end_t in end_indices:
            start_t = end_t - W + 1
            x_win = x_full[start_t:end_t+1]        # [W, N, F]
            x_mask_win = x_mask_full[start_t:end_t+1]
            y_target = y_full[end_t]               # [N, H*M]

            time_arg = time_full[start_t:end_t+1] if model.time_mlp is not None else None

            pred = model(
                x_win,
                x_mask_win,
                edge_index,
                edge_weight=edge_weight,
                time_feat=time_arg,
                condat=condat_full,
            )

            y_nodes = y_target[loss_nodes]         # [N_sel, H*M]
            p_nodes = pred[loss_nodes]             # [N_sel, H*M]

            if not torch.isfinite(y_nodes).any():
                continue

            N_sel = y_nodes.shape[0]

            y_nodes_resh = y_nodes.view(N_sel, M, H_tot)
            p_nodes_resh = p_nodes.view(N_sel, M, H_tot)

            for h_step in eval_horizons:
                h_idx = h_step - 1
                if h_idx < 0 or h_idx >= H_tot:
                    continue

                y_h = y_nodes_resh[..., h_idx]     # [N_sel, M]
                p_h = p_nodes_resh[..., h_idx]     # [N_sel, M]

                mask_h = torch.isfinite(y_h)
                if mask_h.sum() == 0:
                    continue

                y_valid = y_h[mask_h]
                p_valid = p_h[mask_h]

                diff = p_valid - y_valid

                sum_abs[h_step] += diff.abs().sum().item()
                sum_sq[h_step]  += (diff ** 2).sum().item()
                sum_pct[h_step] += (diff.abs() / y_valid).sum().item()
                count[h_step]   += mask_h.sum().item()

    metrics_per_h = {}
    mae_values_for_agg = []

    for h_step in eval_horizons:
        if count[h_step] == 0:
            metrics_per_h[h_step] = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
            continue

        mae_h = sum_abs[h_step] / count[h_step]
        rmse_h = math.sqrt(sum_sq[h_step] / count[h_step])
        mape_h = sum_pct[h_step] / count[h_step]

        mae_values_for_agg.append(mae_h)
        metrics_per_h[h_step] = {"MAE": float(mae_h), "RMSE": float(rmse_h), "MAPE": float(mape_h)}

    if len(mae_values_for_agg) == 0:
        agg_mae = float("nan")
    else:
        agg_mae = float(np.mean(mae_values_for_agg))

    return metrics_per_h, agg_mae


def collect_window_predictions(
    model: nn.Module,
    data: STWindowDataMulti,
    end_indices: np.ndarray,
    device: torch.device,
    loss_nodes: torch.Tensor,
    time_full: torch.Tensor,
    condat_full: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """
    Sammelt Predictions und Targets für bestimmte Window-Endpunkte.
    Output:
      preds:   [num_windows, N_sel, H*M]
      targets: [num_windows, N_sel, H*M]
      end_idx: [num_windows]
    """
    x_full = data.x.to(device)
    x_mask_full = data.x_mask.to(device)
    y_full = data.y.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device)

    W = WINDOW_SIZE

    preds_list = []
    targs_list = []
    ends_list = []

    model.eval()
    with torch.no_grad():
        for end_t in end_indices:
            start_t = end_t - W + 1

            x_win = x_full[start_t:end_t+1]                 # [W, N, F]
            x_mask_win = x_mask_full[start_t:end_t+1]       # [W, N, F]
            y_target = y_full[end_t]                        # [N, H*M]
            time_arg = time_full[start_t:end_t+1] if model.time_mlp is not None else None

            pred = model(
                x_win,
                x_mask_win,
                edge_index,
                edge_weight=edge_weight,
                time_feat=time_arg,
                condat=condat_full,
            )

            y_nodes = y_target[loss_nodes]                  # [N_sel, H*M]
            p_nodes = pred[loss_nodes]                      # [N_sel, H*M]

            mask = torch.isfinite(y_nodes)
            if mask.sum() == 0:
                continue

            # wir speichern trotzdem volle Matrizen, aber setzen non-finite in target/pred auf nan
            y_nodes_clean = y_nodes.clone()
            p_nodes_clean = p_nodes.clone()
            y_nodes_clean[~mask] = float("nan")
            p_nodes_clean[~mask] = float("nan")

            preds_list.append(p_nodes_clean.detach().cpu())
            targs_list.append(y_nodes_clean.detach().cpu())
            ends_list.append(int(end_t))

    if len(preds_list) == 0:
        preds = torch.empty((0, len(loss_nodes), y_full.shape[2]), dtype=torch.float32)
        targs = torch.empty((0, len(loss_nodes), y_full.shape[2]), dtype=torch.float32)
        ends = np.array([], dtype=int)
    else:
        preds = torch.stack(preds_list, dim=0)
        targs = torch.stack(targs_list, dim=0)
        ends = np.array(ends_list, dtype=int)

    return {"preds": preds, "targets": targs, "end_idx": ends}



def train_stgnn_window_multi(
    data: STWindowDataMulti,
    cfg: STTrainConfig,
    gnn_type: str = "sage",
    temporal_type: str = "gru",
    eval_horizons: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    flags: Optional[STFeatureFlags] = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if flags is None:
        flags = STFeatureFlags()

    x_full = data.x.to(device)
    x_mask_full = data.x_mask.to(device)
    y_full = data.y.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device)
    W = WINDOW_SIZE
    
    time_full = torch.tensor(data.time_feat, dtype=torch.float32, device=device)
    
    # ---- condat vorbereiten (Node-Condition-Features) ----
    condat_full = None
    cond_in_dim = 0
    if flags.use_condat and getattr(data, "condat", None) is not None:
        condat_full = data.condat.to(device)
        if condat_full.dim() != 2:
            raise ValueError(f"condat must be 2D [N, C], got shape={tuple(condat_full.shape)}")
        cond_in_dim = int(condat_full.shape[1])


    if flags.use_time:
        time_in_dim = time_full.shape[1]
    else:
        time_in_dim = 0


    M = len(data.target_cols)
    H_per_target = HORIZON_STEPS
    out_dim = H_per_target * M
    in_dim = x_full.shape[2]  

    if eval_horizons is None:
        eval_horizons = [1, 2, 4]

    model = STGNNWindowMultiRegressor(
        num_nodes=x_full.shape[1],
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=out_dim,
        in_dim=in_dim,
        gnn_type=gnn_type,
        temporal_type=temporal_type,
        time_in_dim=time_in_dim,
        temp_hidden_dim=cfg.temp_hidden_dim,
        temp_out_per_scale_week=cfg.temp_out_per_scale_week,
        temp_out_per_scale_day=cfg.temp_out_per_scale_day,
        win_hour=cfg.win_hour,
        win_day=cfg.win_day,
        win_week=cfg.win_week,
        cheb_K=cfg.cheb_K,
        cheb_layers=cfg.cheb_layers,
        length_encoder=cfg.length_encoder, 
        attn_heads=cfg.attn_heads,
        temporal_attn_layers=cfg.temporal_attn_layers,
        use_attention=flags.use_attention,
        use_spatial=flags.use_spatial,
        use_time=flags.use_time,
        use_temp_day=flags.use_temp_day,
        use_temp_week=flags.use_temp_week,
        cond_in_dim=cond_in_dim,
    ).to(device)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    loss_nodes = torch.tensor(data.valid_nodes, dtype=torch.long, device=device)

    best_state = None
    best_val = float("inf")
    wait = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        count_windows = 0

        for end_t in data.train_end_idx:
            start_t = end_t - W + 1

            x_win = x_full[start_t:end_t+1]
            y_target = y_full[end_t]

            x_mask_win = x_mask_full[start_t:end_t+1]
            
            time_arg = time_full[start_t:end_t+1] if flags.use_time else None

            pred = model(
                x_win,
                x_mask_win,
                edge_index,
                edge_weight=edge_weight,
                time_feat=time_arg,
                condat=condat_full,
            )


            y_nodes = y_target[loss_nodes]
            p_nodes = pred[loss_nodes]

            mask = torch.isfinite(y_nodes)
            if mask.sum() == 0:
                continue

            loss = loss_fn(p_nodes[mask], y_nodes[mask])

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            count_windows += 1

        avg_train_loss = epoch_loss / max(count_windows, 1)

        val_per_h, val_agg_mae = _compute_window_metrics_multi_per_horizon(
            model, data, data.val_end_idx, device, loss_nodes,
            eval_horizons, time_full, condat_full
        )

        if val_agg_mae < best_val:
            best_val = val_agg_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_per_h, _ = _compute_window_metrics_multi_per_horizon(
        model, data, data.val_end_idx, device, loss_nodes,
        eval_horizons, time_full, condat_full
    )
    test_per_h, _ = _compute_window_metrics_multi_per_horizon(
        model, data, data.test_end_idx, device, loss_nodes,
        eval_horizons, time_full, condat_full
    )

    return {
        "model": model,
        "metrics": {
            "val_per_h": val_per_h,
            "test_per_h": test_per_h,
        },
        "val_agg_mae": float(best_val),
    }


# =============================
# f) High-level API
# =============================

def train_stgnn_from_data(
    data: STWindowDataMulti,
    *,
    seed: int = SEED,
    eval_horizons: Optional[List[int]] = None,
    hp_overrides: Optional[Dict[str, Any]] = None, 
    device: Optional[torch.device] = None,
    return_model: bool = False,
    flags: Optional[STFeatureFlags] = None,
) -> dict:

    set_seed(seed)
    if eval_horizons is None:
        eval_horizons = [1, 2, 4]

    cfg = STTrainConfig()

    if hp_overrides is not None:
        for k, v in hp_overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            else:
                raise ValueError(f"Unknown Hyperparameters in hp_overrides: {k}")

    result = train_stgnn_window_multi(
        data,
        cfg,
        eval_horizons=eval_horizons,
        device=device,
        flags=flags,
    )

    out = {
        "metrics": result["metrics"],
        "val_agg_mae": result["val_agg_mae"],
    }
    if return_model:
        out["model"] = result["model"]
    return out


# =============================
# g) Multi-Seed Experiments
# =============================

def run_multi_seed_experiment(
    *,
    data: STWindowDataMulti,
    seeds: List[int],
    eval_horizons: Optional[List[int]] = None,
    hp_overrides: Optional[Dict[str, Any]] = None,   # <--- neu
    device: Optional[torch.device] = None,
    flags: Optional[STFeatureFlags] = None,
) -> dict:

    if eval_horizons is None:
        eval_horizons = [1, 2, 4]

    all_results = []

    metrics_collect = {
        "val_per_h": {h: {"MAE": [], "RMSE": [], "MAPE": []} for h in eval_horizons},
        "test_per_h": {h: {"MAE": [], "RMSE": [], "MAPE": []} for h in eval_horizons},
    }

    for s in seeds:
        res = train_stgnn_from_data(
            data,
            seed=s,
            eval_horizons=eval_horizons,
            hp_overrides=hp_overrides,
            device=device,
            flags=flags,
        )

    
        all_results.append(res)

        for split_key in ["val_per_h", "test_per_h"]:
            for h in eval_horizons:
                for metric in ["MAE", "RMSE", "MAPE"]:
                    metrics_collect[split_key][h][metric].append(
                        res["metrics"][split_key][h][metric]
                    )

    metrics_stats = {
        "val_per_h": {},
        "test_per_h": {},
    }

    for split_key in ["val_per_h", "test_per_h"]:
        for h in eval_horizons:
            metrics_stats[split_key][h] = {}
            for metric in ["MAE", "RMSE", "MAPE"]:
                values = np.array(metrics_collect[split_key][h][metric], dtype=float)
                metrics_stats[split_key][h][metric + "_mean"] = float(values.mean())
                metrics_stats[split_key][h][metric + "_var"] = float(values.var(ddof=1))

    return {
        "seeds": seeds,
        "all_results": all_results,
        "metrics_raw": metrics_collect,
        "metrics_stats": metrics_stats,
    }


# =============================
# h) Hyperparameter Tuning
# =============================

def sample_hyperparams(
    rng: np.random.Generator,
) -> Dict[str, Any]:
    embed_dim_choices = [16, 32, 64]
    hidden_dim_choices = [64, 128, 256]
    temp_hidden_choices = [32, 64, 128]
    temp_out_choices_week = [32, 64, 128]
    temp_out_choices_day = [16, 32, 64]

    win_hour_choices = [4, 8]
    length_encoder = int(rng.choice([12, 16, 32]))
    win_day_candidates = [16, 32, 64, 96]
    win_week_candidates = [96, 192, 288, 480, 672]
    win_day_choices = [w for w in win_day_candidates if w % length_encoder == 0]
    win_week_choices = [w for w in win_week_candidates if w % length_encoder == 0]
    win_day = int(rng.choice(win_day_choices))
    win_week = int(rng.choice(win_week_choices))
  
    cheb_K_choices = [2, 3, 4]
    cheb_layers_choices = [1, 2, 3, 4, 5, 6]
    attn_heads_choices = [2, 4, 8] 
    temporal_attn_layers_choices = [1, 2, 3, 4, 5, 6]

    def log_uniform(low, high):
        log_low, log_high = np.log10(low), np.log10(high)
        val = rng.uniform(log_low, log_high)
        return float(10 ** val)

    hp = {
        "lr": log_uniform(1e-5, 5e-3),
        "weight_decay": log_uniform(1e-6, 1e-3),
        "embed_dim": int(rng.choice(embed_dim_choices)),
        "hidden_dim": int(rng.choice(hidden_dim_choices)),
        "temp_hidden_dim": int(rng.choice(temp_hidden_choices)),
        "temp_out_per_scale_week": int(rng.choice(temp_out_choices_week)),
        "temp_out_per_scale_day": int(rng.choice(temp_out_choices_day)),
        "win_hour": int(rng.choice(win_hour_choices)),
        "win_day": win_day,
        "win_week": win_week,
        "cheb_K": int(rng.choice(cheb_K_choices)),
        "cheb_layers": int(rng.choice(cheb_layers_choices)),
        "length_encoder": length_encoder,
        "attn_heads": int(rng.choice(attn_heads_choices)),
        "temporal_attn_layers": int(rng.choice(temporal_attn_layers_choices)),
    }
    return hp

def get_trial_id(cli_trial: int | None) -> int:
    if cli_trial is not None:
        return int(cli_trial)
    if "PBS_ARRAY_INDEX" in os.environ:
        return int(os.environ["PBS_ARRAY_INDEX"]) - 1
    return 0  # fallback: lokaler Testlauf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["tune", "train"], required=True)

    # tune args
    parser.add_argument("--trial", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=42)

    # train args
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[SEED])

    # shared
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="trial_results/E5_4N")

    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    blob = torch.load(args.data_path, map_location="cpu")
    data = STWindowDataMulti(**blob)
    
    flags = STFeatureFlags() 


    if args.mode == "tune":
        trial = get_trial_id(args.trial)

        rng = np.random.default_rng(args.base_seed + trial)
        hp = sample_hyperparams(rng)

        res = run_multi_seed_experiment(
            data=data,
            seeds=[SEED],
            eval_horizons=[1, 2, 4],
            hp_overrides=hp,
            device=device,
            flags=flags,
        )

        stats = res["metrics_stats"]
        score = float(np.mean([stats["val_per_h"][h]["MAE_mean"] for h in [1, 2, 4]]))

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        out = {"trial": trial, "score": score, "hyperparams": hp, "metrics_stats": stats}
        with open(outdir / f"trial_{trial:05d}.json", "w") as f:
            json.dump(out, f, indent=2)

        print(f"trial={trial} score={score}", flush=True)


    elif args.mode == "train":
        if args.config_path is None:
            raise ValueError("--config-path is required in train mode.")

        # config laden
        with open(args.config_path, "r") as f:
            cfg_blob = json.load(f)
            
        flag_blob = {}
        if isinstance(cfg_blob, dict):
            flag_blob = cfg_blob.get("flags", {})
        flags = STFeatureFlags(**flag_blob)

        # akzeptiere zwei Formate:
        # 1) direkt hyperparam dict
        # 2) tuning output dict mit key "hyperparams"
        if isinstance(cfg_blob, dict) and "hyperparams" in cfg_blob:
            hp = cfg_blob["hyperparams"]
        else:
            hp = cfg_blob

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # time_feat in torch, genau wie im training
        time_full = torch.tensor(data.time_feat, dtype=torch.float32, device=device)
        loss_nodes = torch.tensor(data.valid_nodes, dtype=torch.long, device=device)
        
        # condat (Node-Features) auf device ziehen
        condat_full = data.condat.to(device) if getattr(data, "condat", None) is not None else None


        per_seed_results = []

        for s in args.seeds:
            res = train_stgnn_from_data(
                data,
                seed=s,
                eval_horizons=[1, 2, 4],
                hp_overrides=hp,
                device=device,
                return_model=True,
                flags=flags,
            )

            model = res["model"]

            # predictions fuer test set
            pred_pack = collect_window_predictions(
                model=model,
                data=data,
                end_indices=data.test_end_idx,
                device=device,
                loss_nodes=loss_nodes,
                time_full=time_full,
                condat_full=condat_full, 
            )

            # speichern: model + preds
            model_path = outdir / f"model_seed{s}.pt"
            torch.save(
                {
                    "seed": s,
                    "hyperparams": hp,
                    "state_dict": model.state_dict(),
                },
                model_path,
            )

            preds_path = outdir / f"preds_test_seed{s}.pt"
            torch.save(
                {
                    "seed": s,
                    "hyperparams": hp,
                    "preds": pred_pack["preds"],
                    "targets": pred_pack["targets"],
                    "end_idx": pred_pack["end_idx"],
                },
                preds_path,
            )

            per_seed_results.append(
                {
                    "seed": s,
                    "val_agg_mae": res["val_agg_mae"],
                    "metrics": res["metrics"],
                    "model_path": str(model_path),
                    "preds_path": str(preds_path),
                }
            )

        # optional: Metriken über seeds aggregieren (simple mean/var)
        # wir nehmen MAE_mean über horizons im test/val wie beim tune score
        def _extract_mae(metrics, split, h):
            return float(metrics[split][h]["MAE"])

        summary = {
            "mode": "train",
            "seeds": args.seeds,
            "hyperparams": hp,
            "per_seed": per_seed_results,
        }

        with open(outdir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"train done. saved to {outdir}", flush=True)


# In[ ]:


if __name__ == "__main__":
    main()

