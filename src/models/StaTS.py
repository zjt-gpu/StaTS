import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_timeseries.nn.embedding import DataEmbedding
import math
from einops import rearrange

import src.layer.mu_backbone as ns_Transformer

from torch.nn import MultiheadAttention
import torch.fft

import os
import datetime

class FrequencyGuidanceLite(nn.Module):
    def __init__(self, n_freq, use_log=True):
        super().__init__()
        self.use_log = use_log
        self.a = nn.Parameter(torch.zeros(n_freq))  # slope per freq
        self.b = nn.Parameter(torch.zeros(n_freq))  # bias  per freq

    def forward(self, energy):  # [B, F]
        if self.use_log:
            energy = torch.log1p(energy)
        gate = torch.sigmoid(self.a[None, :] * energy + self.b[None, :])  # [B, F]
        return gate

class MultiBandDiagComplexScale(nn.Module):
    def __init__(self, n_freq, n_bands=2, init=0.02):
        super().__init__()
        self.n_bands = n_bands
        # 每个 band 一个复数缩放向量（按频点）
        self.scales = nn.ParameterList()
        splits = torch.tensor_split(torch.arange(n_freq), n_bands)
        for sp in splits:
            scale = torch.randn(len(sp), dtype=torch.cfloat) * init
            self.scales.append(nn.Parameter(scale))

    def forward(self, freq):  # [B, N, F] complex
        bands = torch.tensor_split(freq, self.n_bands, dim=2)
        outs = []
        for b, h in zip(bands, self.scales):
            # b: [B,N,Fb], s: [Fb]
            outs.append(b * h[None, None, :])
        return torch.cat(outs, dim=2)

class FrequencyModuleLite(nn.Module):
    def __init__(self, seq_len, pred_len, n_bands=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        n_freq = seq_len // 2 + 1

        self.gate = FrequencyGuidanceLite(n_freq)
        self.band_scale = MultiBandDiagComplexScale(n_freq, n_bands=n_bands)

        # 关键：轻量预测头（沿时间）
        self.time_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x):  # x: [B, T, N]
        B, T, N = x.shape
        assert T == self.seq_len

        xn = x.permute(0, 2, 1)  # [B, N, T]
        freq = torch.fft.rfft(xn, dim=2)  # [B, N, F] complex

        energy = freq.abs().mean(dim=1)  # [B, F]  (全局能量也够用，想更细可改成 [B,N,F])
        gate = self.gate(energy).unsqueeze(1)  # [B,1,F]
        freq = freq * gate

        freq = self.band_scale(freq)
        x_filt = torch.fft.irfft(freq, n=self.seq_len, dim=2)  # [B, N, T]

        # 轻量预测：T -> pred_len
        y_pred = self.time_proj(x_filt)  # [B, N, pred_len]
        return y_pred.permute(0, 2, 1)  # [B, pred_len, N]

class FiLM(nn.Module):
    def __init__(self, n_steps, hidden):
        super().__init__()
        self.emb = nn.Embedding(n_steps, hidden * 2)
        nn.init.uniform_(self.emb.weight, -0.02, 0.02)

    def forward(self, h, t):  # h: [B, N, H], t: [B]
        gb = self.emb(t)  # [B, 2H]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma[:, None, :]  # [B,1,H]
        beta  = beta[:, None, :]   # [B,1,H]
        return (1 + gamma) * h + beta

class ConditionalGuidedModelLite(nn.Module):
    def __init__(self, diff_steps, enc_in, pred_len, hidden=256):
        super().__init__()
        n_steps = diff_steps + 1
        self.pred_len = pred_len
        self.enc_in = enc_in

        # guidance 投影（可选：做个 bottleneck 更轻）
        self.freq_proj = nn.Sequential(
            nn.Linear(enc_in, max(8, enc_in // 4)),
            nn.SiLU(),
            nn.Linear(max(8, enc_in // 4), enc_in),
            nn.Sigmoid()
        )

        self.in1 = nn.Linear(pred_len, hidden)
        self.in2 = nn.Linear(pred_len, hidden)

        self.film1 = FiLM(n_steps, hidden)
        self.film2 = FiLM(n_steps, hidden)

        self.mid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, pred_len)

        self.act = nn.SiLU()

    def forward(self, x, x_noise_state, y_t, t):
        # x, x_noise_state: [B, T, N]
        # y_t: [B, L, N] where L=pred_len
        B, L, N = y_t.shape

        fft_x = torch.fft.rfft(x, dim=1)             # [B, F, N]
        fft_n = torch.fft.rfft(x_noise_state, dim=1) # [B, F, N]
        ratio = (fft_n.abs() - fft_x.abs()) / (fft_x.abs() + 1e-4)
        ratio = ratio.clamp(-10.0, 10.0)
        freq_damage = torch.tanh(ratio).mean(dim=1)

        g = self.freq_proj(freq_damage)  # [B, N] in (0,1)
        g = g.unsqueeze(-1)              # [B, N, 1]

        # 以“整段 horizon 向量”为单位做 MLP（跟你原来一致，但更轻更稳）
        y = y_t.permute(0, 2, 1)         # [B, N, L]
        y_guided = y * g                 # [B, N, L]

        h = self.in1(y) + self.in2(y_guided)         # [B, N, H]
        h = self.act(self.film1(h, t))
        h = self.act(self.film2(self.mid(h), t))
        y0_pred = self.out(h).permute(0, 2, 1)           # [B, L, N]
        return y0_pred

class FGD(nn.Module):
    def __init__(self, configs, dim, num_steps, windows, pred_len):
        super().__init__()
        self.num_steps = num_steps
        self.fre_pred = FrequencyModuleLite(windows, pred_len, n_bands=2)
        self.diffussion_model = ConditionalGuidedModelLite(
            configs["diffusion"]["num_steps"], dim, pred_len, hidden=256
        )

        self.lambd_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, noise_x, y, x_mark, y_mark, t):
        y_pred = self.fre_pred(x)                  # [B, L, N]
        y0_pred = self.diffussion_model(x, noise_x, y, t)  # [B, L, N]

        lambd = torch.sigmoid(self.lambd_raw)
        out = lambd * y0_pred + (1 - lambd) * y_pred
        return out

def extract(a, t, x_shape):
    if isinstance(t, int):
        t = torch.tensor([t], device=a.device, dtype=torch.long)
    else:
        t = t.to(a.device).long()
    b = t.shape[0]
    out = a[torch.arange(b, device=a.device), t]
    return out

def inv_softplus(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # softplus^{-1}(x) = log(exp(x) - 1)
    x = x.clamp_min(eps)
    return torch.log(torch.expm1(x))

def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    cosine_s: float = 0.008,
    device=None,
    dtype=torch.float32,
    k: float=1
) -> torch.Tensor:
    """
    Return betas with shape [T].
    For non-const schedules we rescale to exactly hit [beta_start, beta_end].
    """
    T = timesteps
    schedule = schedule.lower()
    device = device or torch.device("cpu")

    if schedule in ["linear", "lin"]:
        betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=dtype)

    elif schedule in ["quad", "quadratic"]:
        # sqrt-linear then square (classic)
        betas = torch.linspace(math.sqrt(beta_start), math.sqrt(beta_end), T, device=device, dtype=dtype) ** 2

    elif schedule in ["cosine", "cos"]:
        # cosine schedule (Nichol & Dhariwal style)
        steps = T + 1
        x = torch.linspace(0, T, steps, device=device, dtype=dtype)
        alphas_cumprod = torch.cos(((x / T) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas_raw = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_raw = betas_raw.clamp(0, 0.999)

        # rescale to [beta_start, beta_end]
        b0, b1 = betas_raw[0], betas_raw[-1]
        betas = (betas_raw - b0) / (b1 - b0 + 1e-12)
        betas = betas * (beta_end - beta_start) + beta_start

    elif schedule in ["jsd"]:
        # 1/T, 1/(T-1), ..., 1
        betas_raw = 1.0 / torch.linspace(T, 1, T, device=device, dtype=dtype)
        # rescale to [beta_start, beta_end]
        b0, b1 = betas_raw[0], betas_raw[-1]
        betas = (betas_raw - b0) / (b1 - b0 + 1e-12)
        betas = betas * (beta_end - beta_start) + beta_start

    elif schedule in ["const", "constant"]:
        
        t = torch.linspace(0, 1, T, device=device, dtype=dtype)
        y = 1.0 - torch.exp(-k * t)
        y = (y - y[0]) / (y[-1] - y[0] + 1e-12)
        betas = y * (beta_end - beta_start) + beta_start

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return betas

class BetaScheduler(nn.Module):
    def __init__(
        self,
        num_steps: int,
        feature_dim: int,
        dataset_name: str,
        d_tau: int = 512,
        windows: int = 192,
        init_schedule: str = "Linear",
        cosine_s: float = 0.008,
        learnable: bool = True,
        eps: float = 1e-6,
        use_beta_range: bool = False
    ):
        super().__init__()
        self.num_steps = num_steps

        self.beta_min = 1e-5
        self.beta_max = 0.1

        self.dataset_name = dataset_name
        self.save_interval = 200
        self.forward_count = 0

        self.init_schedule = init_schedule
        self.cosine_s = cosine_s
        self.learnable = learnable
        self.eps = eps
        self.use_beta_range = use_beta_range

        # [T]
        self.step_embed = nn.Parameter(torch.empty(num_steps))

        # ======== 新增：EMA buffer（存一条稳定的 beta 曲线）========
        self.beta_ema = torch.zeros(num_steps, dtype=self.step_embed.dtype, device=self.step_embed.device)
        self.beta_ema_inited = torch.tensor(False, device=self.step_embed.device)
        self.ema_momentum = 0.995  # 可按需要调：0.99~0.999
        self._init_step_embed()

    @torch.no_grad()
    def _init_step_embed(self):
        device = self.step_embed.device
        dtype = self.step_embed.dtype

        beta_target = make_beta_schedule(
            schedule=self.init_schedule,
            timesteps=self.num_steps,
            beta_start=self.beta_min,
            beta_end=self.beta_max,
            cosine_s=self.cosine_s,
            device=device,
            dtype=dtype,
        )
        self.step_embed.copy_(beta_target)
        

        # if self.use_beta_range:
        #     y = (beta_target - self.beta_min) / (self.beta_max - self.beta_min + 1e-12)
        # else:
        #     y = beta_target

        # y = y.clamp(self.eps, 1.0 - self.eps)
        # self.step_embed.copy_(torch.log(y) - torch.log1p(-y))

    def forward(self, x_enc: torch.Tensor, use_ema: bool = False):
        B = x_enc.shape[0]

        # 不学习：固定 schedule
        if not self.learnable:
            beta = make_beta_schedule(
                schedule=self.init_schedule,
                timesteps=self.num_steps,
                beta_start=self.beta_min,
                beta_end=self.beta_max,
                cosine_s=self.cosine_s,
                device=x_enc.device,
                dtype=x_enc.dtype,
            ).unsqueeze(0).expand(B, -1)
            return beta

        # 学习：sigmoid 保证 [0,1]
        beta01 = self.step_embed.clamp(self.eps, 1.0 - self.eps)#torch.sigmoid(self.step_embed).clamp(self.eps, 1.0 - self.eps)  # [T]
        return beta01.unsqueeze(0).expand(B, -1)

def softplus_inv(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    softplus^{-1}(y) = log(exp(y) - 1)
    数值稳定：用 log(expm1(y))
    """
    y = y.clamp_min(eps)
    return torch.log(torch.expm1(y))


class StaTS_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.windows = config["model"]["windows"]
        self.pred_len = config["model"]["pred_len"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1 

        config_diff = config["diffusion"]

        self.num_steps = config["diffusion"]["num_steps"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = FGD(config, target_dim, self.num_steps, self.windows, self.pred_len)
        
        self.beta_scheduler = BetaScheduler(
            num_steps=self.num_steps,
            feature_dim=self.target_dim,
            dataset_name=config["model"]["dataset"],
            init_schedule=config_diff["schedule"],   
        )

        self.delta_update_interval = 200   
        self.delta_t_stride = 1             
        self.delta_max_batch = 8        
        self.lambda_sched = 5e-4           
        self.lambda_end = 5e-4           
        self._global_step = 0               
        self._target_cum = None          

        self.original_beta = torch.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps, device=device, dtype=torch.float32)

    def get_dynamic_beta(self, observed_data):
        self.beta = self.beta_scheduler(observed_data) 
        self.alpha_hat = 1 - self.beta
        self.alpha = torch.cumprod(self.alpha_hat, dim=-1).to(self.device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alpha)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alpha - 1)
        self.alpha_torch = self.alpha.float()
        return self.beta, self.alpha
    

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2, device=self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask
     
    
    def calc_flatness(self, x):
        # x: [B, L, C]
        fft_vals = torch.fft.rfft(x, dim=1)            # [B, F, C]
        power = (fft_vals.abs() ** 2)                  # [B, F, C]
        power_mean = power.mean(dim=(0, 2))            # [F]
        eps = 1e-8
        power_mean = power_mean + eps

        log_mean = power_mean.log().mean()             # (1/F) Σ log P
        mean_log = power_mean.mean().log()             # log( (1/F) Σ P )

        sfm = torch.exp(log_mean - mean_log)           # exp( mean(logP) - log(meanP) )
        return sfm    # 标量，越接近1越“白”

    def calc_loss(self, x, y, x_mark, y_mark, cond_mask, current_epoch, is_train=True, set_t=-1):
        
        device = self.device
        B = y.size(0)
        observed_data = y 

        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        beta, alpha = self.get_dynamic_beta(x_enc)  
        alpha_torch = alpha  
        
        t = torch.randint(0, self.num_steps, (B,), device=device)
        
        current_beta = beta.gather(1, t.view(B, 1)).view(B, 1, 1)
        current_alpha = alpha_torch.gather(1, t.view(B, 1)).view(B, 1, 1)
        
        y_enc = y - means
        y_enc /= stdev

        noise = torch.randn_like(y_enc)
        noisy_data = torch.sqrt(current_alpha) * y_enc + torch.sqrt(1 - current_alpha) * noise
        history_noise = torch.randn_like(x_enc)
        noise_x = torch.sqrt(current_alpha) * x_enc + torch.sqrt(1 - current_alpha) * history_noise
        
        predicted = self.diffmodel(x_enc, noise_x, noisy_data, x_mark, y_mark, t)
        _, L, _ = predicted.shape
        predicted = predicted * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        predicted = predicted + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        
        residual = predicted - observed_data
        
        denoise_loss = residual.pow(2)
        idx_T = torch.full(
            (B, 1),
            self.num_steps - 1,          
            dtype=torch.long,
            device=alpha_torch.device,
        )
        
        eps = 1e-8
        alpha_bar_T = alpha_torch[:, -1].view(B, 1, 1)  

        eps_T = torch.randn_like(y_enc)
        x_T = torch.sqrt(alpha_bar_T) * y_enc + torch.sqrt(1.0 - alpha_bar_T) * eps_T   # [B, L, N]

        fft_T = torch.fft.rfft(x_T, dim=1)          # [B, F, N]
        power = (fft_T.abs() ** 2).mean(dim=2) + eps  # [B, F] 

        power = power[:, 1:]                    # [B, F-1]

        p = power / (power.sum(dim=1, keepdim=True) + eps)  # [B, K]
        K = p.size(1)

        loss_flat = (p * (torch.log(p.clamp_min(eps)) + math.log(K))).sum(dim=1).mean()  # scalar
        end_loss = getattr(self, "lambda_flat", 0.5) * loss_flat

        # beta_loss = curvature
        beta_mean = beta.mean(0)
        diff_sq = (beta_mean[1:] - beta_mean[:-1]).pow(2)   # [T-1]

        beta_mean = beta.mean(0)                 # [T]

        barrier = -torch.log(beta_mean[1:]).mean()

        idx_T = torch.full((B, 1), self.num_steps - 1, dtype=torch.long, device=device)
        T_alpha = alpha_torch.gather(1, idx_T).view(B, 1, 1).clamp(eps, 1.0 - eps)  # [B,1,1]

        end = torch.sqrt(T_alpha) * y_enc + torch.sqrt(1.0 - T_alpha) * noise  # [B,L,N]

        t_norm = (t.float() / (self.num_steps - 1)).view(B, 1, 1)  # [B,1,1]
        lambda_t = t_norm.clamp(0.0, 1.0)

        noisy_flat  = self.calc_flatness(noisy_data)
        end_flat  = self.calc_flatness(end)
        start_flat  = self.calc_flatness(y_enc)
        target_flat = (1.0 - lambda_t) * start_flat + lambda_t * end_flat  # [B,F,1]

        # energy_loss = (torch.log(noisy_psd + eps) - torch.log(target_psd + eps)).pow(2).mean()
        flat_loss = (noisy_flat - target_flat).pow(2).mean()
        beta_loss = diff_sq.mean() * 10 + beta_mean[0].pow(2) * 1 + end_loss + 1e-3 * barrier + flat_loss

        total_loss = denoise_loss.mean() 
        return predicted, noise, total_loss, beta_loss


    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            return noisy_data.unsqueeze(1)
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return torch.cat([cond_obs, noisy_target], dim=1)  

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def impute(self, x, y, x_mark, y_mark, n_samples=1, observed_mask=None):
        device = self.device
        B, K, L = y.shape
        x = x.to(device); y = y.to(device)
        x_mark = x_mark.to(device); y_mark = y_mark.to(device)

        means = x.mean(1, keepdim=True).detach()
        x_enc = x - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        beta, alpha = self.get_dynamic_beta(x_enc)
        if alpha.dim() == 1:  
            alpha = alpha.unsqueeze(0).expand(B, -1)
        alpha_bar = torch.cumprod(alpha, dim=1)  
        T = alpha.shape[1]

        imputed_samples = torch.zeros(B, n_samples, K, L, device=device)

        
        all_pred_spectra = []  

        for i in range(n_samples):  
            current = torch.randn_like(y)
            pred_spectra_over_t = []  

            for t in reversed(range(T)):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                
                history_noise = torch.randn_like(x_enc)
                current_alpha = alpha[:, t].view(B, 1, 1)
                noise_x = torch.sqrt(current_alpha) * x_enc + torch.sqrt(1 - current_alpha) * history_noise
                x_start = self.diffmodel(x_enc, noise_x, current, x_mark, y_mark, t_tensor)
                t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                pred_noise = self.predict_noise_from_start(current, t_tensor, x_start)

                if t == 0:
                    current = x_start
                else:
                    eta = 0
                    alpha_t = self.alpha[:, t]
                    alpha_t_prev = self.alpha[:, t - 1]
                    sigma = eta * torch.sqrt(
                        ((1 - alpha_t_prev / alpha_t) * (1 - alpha_t) / (1 - alpha_t_prev)).clamp_min(1e-8)
                    )
                    c = torch.sqrt((1 - alpha_t_prev - sigma ** 2).clamp_min(1e-8))

                    # reshape for broadcast
                    alpha_t_prev = alpha_t_prev.view(B, 1, 1)
                    c = c.view(B, 1, 1)
                    sigma = sigma.view(B, 1, 1)

                    pred_mean = x_start * torch.sqrt(alpha_t_prev) + c * pred_noise
                    noise = torch.randn_like(current)
                    current = pred_mean + sigma * noise

            _, L, _ = current.shape
            current = current * (stdev[:, 0, :].unsqueeze(1).repeat(1, K, 1))
            current = current + (means[:, 0, :].unsqueeze(1).repeat(1, K, 1))
            final_out = current
            imputed_samples[:, i] = final_out.detach()
        
        return imputed_samples


class StaTS_Forecasting(StaTS_base):
    def __init__(self, config, device, target_dim):
        super(StaTS_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        x = batch["x"].to(self.device).float()
        y = batch["y"].to(self.device).float()
        x_mark = batch["x_mark"].to(self.device).float()
        y_mark = batch["y_mark"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        return (
            x,
            y,
            x_mark,
            y_mark,
            observed_mask,
            gt_mask,
            for_pattern_mask,
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask

    
    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def forward(self, batch, current_epoch, is_train=1):

        (
            x,
            y,
            x_mark,
            y_mark,
            observed_mask,
            gt_mask,
            for_pattern_mask
        ) = self.process_data(batch)
        
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)
        if is_train:
            noise, predicted, loss, beta_loss = self.train_forward(x, y, x_mark, y_mark, observed_mask, cond_mask, current_epoch)
            return noise, predicted, loss, beta_loss
        else:
            NotImplementedError("we do not caculate loss during validation phase!!!")
        
        
    def train_forward(self, x, y, x_mark, y_mark, observed_mask, cond_mask, current_epoch):
        B, _, _ = x.shape

        noise, predicted, loss, beta_loss = self.calc_loss(x, y, x_mark, y_mark, cond_mask, current_epoch)
        
        return noise, predicted, loss, beta_loss


    def evaluate(self, batch, n_samples, minib_n_sample):
        (
            x,
            y,
            x_mark,
            y_mark,
            observed_mask,
            gt_mask,
            for_pattern_mask,
        ) = self.process_data(batch)

        with torch.no_grad():
            samples = []

            for i in range(0, n_samples, minib_n_sample):
                sub_samples = self.impute(
                    x, y, x_mark, y_mark, minib_n_sample
                )
                samples.append(sub_samples)

            samples = torch.cat(samples, dim=1)  # (B, n_samples, K, L)
        return samples
