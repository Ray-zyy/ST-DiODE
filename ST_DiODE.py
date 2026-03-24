import math
import torch
import torch.fft
from torch import nn


# ==============================================================================
# Section 1: Basic Convolution & Spatial Feature Extraction
# ==============================================================================

class BasicConv2d(nn.Module):
    """Standard 2D convolution block with optional activation and normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation, transpose=False, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if not transpose:
            padding = dilation * (kernel_size - 1) // 2
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=(kernel_size - 1) // 2,
                output_padding=stride // 2
            )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    """Single-layer convolution wrapper with stride control."""

    def __init__(self, C_in, C_out, stride, dilation=1,
                 transpose=False, act_norm=True):
        super().__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(
            C_in, C_out, kernel_size=3, stride=stride,
            dilation=dilation, transpose=transpose, act_norm=act_norm
        )

    def forward(self, x):
        return self.conv(x)


class GroupConv2d(nn.Module):
    """Group convolution with optional activation and normalization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, groups, act_norm=False):
        super().__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    """Multi-scale receptive field module for capturing spatial derivatives."""

    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker // 2, groups=groups, act_norm=True
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


def stride_generator(N, reverse=False):
    """Generate an alternating [1, 2, 1, 2, ...] stride pattern."""
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    return strides[:N]


# ==============================================================================
# Section 2: Spatial Encoder & Decoder
# ==============================================================================

class StateNet(nn.Module):
    """Encode raw observation frames into latent spatial states."""

    def __init__(self, C_in, C_hid, N_S):
        super().__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0], dilation=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s, dilation=s) for s in strides[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class FrameNet(nn.Module):
    """Decode latent states back into high-resolution video frames."""

    def __init__(self, C_hid, C_out, N_S):
        super().__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, kernel_size=1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


# ==============================================================================
# Section 3: SpecDyn — Spectral Dynamics Branch
# ==============================================================================

class RevIN_Spatial(nn.Module):
    """Reversible Instance Normalization for spatiotemporal sequences."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features, 1, 1))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=(1, 3, 4), keepdim=True)
            self.var = x.var(dim=(1, 3, 4), keepdim=True, unbiased=False)
            x = (x - self.mean) / torch.sqrt(self.var + self.eps)
            if self.affine:
                x = x * self.weight + self.bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias) / self.weight
            x = x * torch.sqrt(self.var + self.eps) + self.mean
        return x


class SpecDyn(nn.Module):
    """Spectral dynamics block: captures global periodic patterns via FFT."""

    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.valid_freq_points = int((self.seq_len + 1) / 2 + 0.5)

        self.freq_linear = nn.Sequential(
            nn.Linear(self.valid_freq_points * 2, self.valid_freq_points * 2),
            nn.GELU(),
            nn.Linear(self.valid_freq_points * 2, self.valid_freq_points * 2)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B, T, -1).transpose(1, 2)  # [B, C*H*W, T]

        # Forward FFT along the temporal axis
        x_freq = torch.fft.rfft(x_flat, dim=-1, norm='ortho')
        x_cat = torch.cat([x_freq.real, x_freq.imag], dim=-1)

        # Learnable frequency-domain transform
        x_cat = self.freq_linear(x_cat)

        y_real = x_cat[..., :self.valid_freq_points]
        y_imag = x_cat[..., self.valid_freq_points:]
        y_complex = torch.complex(y_real, y_imag)

        # Inverse FFT back to temporal domain
        x_ifft = torch.fft.irfft(y_complex, n=T, dim=-1, norm='ortho')
        x_out = x_ifft.transpose(1, 2).view(B, T, C, H, W)

        return x + x_out


# ==============================================================================
# Section 4: ContDyn — Continuous Dynamics Branch (Neural ODE)
# ==============================================================================

class ContDynFunc(nn.Module):
    """Defines the continuous state derivative: dz/dt = f(z, t)."""

    def __init__(self, channels, incep_ker=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.net = nn.Sequential(
            Inception(channels, channels // 2, channels,
                      incep_ker=incep_ker, groups=groups),
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, t, z):
        return self.net(z)


class RK4Solver(nn.Module):
    """4th-order Runge-Kutta integrator for the ContDyn ODE."""

    def __init__(self, odefunc, steps_per_interval=2):
        super().__init__()
        self.odefunc = odefunc
        self.steps = steps_per_interval

    def forward(self, z0, time_steps):
        trajectory = []
        z = z0
        dt = 1.0 / self.steps

        for _ in range(time_steps):
            for _ in range(self.steps):
                k1 = self.odefunc(0, z)
                k2 = self.odefunc(0, z + dt * k1 / 2)
                k3 = self.odefunc(0, z + dt * k2 / 2)
                k4 = self.odefunc(0, z + dt * k3)
                z = z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            trajectory.append(z)

        return torch.stack(trajectory, dim=1)  # [B, T_out, C, H, W]


# ==============================================================================
# Section 5: Dual-Stream Evolution Engine (ContDyn + SpecDyn)
# ==============================================================================

class DualStreamTranslator(nn.Module):
    """Fuses the ContDyn (ODE) branch and the SpecDyn (spectral) branch."""

    def __init__(self, channel_in, channel_hid, T_in, T_out,
                 incep_ker=[3, 5, 7, 11], groups=8):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.base_C = channel_in // T_in

        # Initial state extractor for z0
        self.z0_extractor = nn.Sequential(
            nn.Conv2d(self.base_C * T_in, self.base_C, kernel_size=3, padding=1),
            nn.GroupNorm(2, self.base_C),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Branch 1: ContDyn — continuous ODE evolution
        self.contdyn_func = ContDynFunc(self.base_C, incep_ker, groups)
        self.contdyn_solver = RK4Solver(self.contdyn_func, steps_per_interval=2)

        # Branch 2: SpecDyn — global spectral dynamics
        self.time_projector = nn.Linear(T_in, T_out)
        self.specdyn = SpecDyn(seq_len=T_out)
        self.revin = RevIN_Spatial(num_features=self.base_C)

        # Cross-domain fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.base_C, self.base_C, kernel_size=3, padding=1),
            nn.GroupNorm(2, self.base_C),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        B, T_in, C, H, W = x.shape

        # --- ContDyn branch ---
        x_flat = x.view(B, T_in * C, H, W)
        z0 = self.z0_extractor(x_flat)
        y_contdyn = self.contdyn_solver(z0, time_steps=self.T_out)

        # --- SpecDyn branch ---
        y_spec = self.revin(x, mode='norm')
        y_spec_flat = y_spec.view(B, T_in, -1).transpose(1, 2)
        y_spec_proj = self.time_projector(y_spec_flat) \
            .transpose(1, 2).view(B, self.T_out, C, H, W)
        y_spec = self.specdyn(y_spec_proj)
        y_spec = self.revin(y_spec, mode='denorm')

        # --- Fusion ---
        y_fused = torch.cat([y_contdyn, y_spec], dim=2)
        y_fused = y_fused.view(B * self.T_out, 2 * C, H, W)
        y_out = self.fusion(y_fused)
        return y_out.view(B, self.T_out, C, H, W)


# ==============================================================================
# Section 6: AdaLN-ConvNeXt Denoising Network & AGD Loss
# ==============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timestep conditioning."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AdaLN_ConvBlock(nn.Module):
    """ConvNeXt block with Adaptive Layer Norm modulation (scale, shift, gate)."""

    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.GroupNorm(1, channels)
        self.pwconv1 = nn.Conv2d(channels, 4 * channels, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * channels, channels, 1)

        # AdaLN modulation: produces scale, shift, gate from timestep embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 3 * channels)
        )

    def forward(self, x, t_emb):
        scale, shift, gate = self.adaLN_modulation(t_emb) \
            .unsqueeze(-1).unsqueeze(-1).chunk(3, dim=1)

        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x * (1 + scale) + shift  # AdaLN modulation
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return shortcut + x * gate  # Gated residual connection


class ConditionalDenoisingNet(nn.Module):
    """Conditional denoising network built from stacked AdaLN-ConvBlocks."""

    def __init__(self, channels, time_dim=256, num_blocks=3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.proj_in = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            AdaLN_ConvBlock(channels, time_dim) for _ in range(num_blocks)
        ])
        self.norm_out = nn.GroupNorm(1, channels)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Zero-initialize the final projection for stable training onset
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x_t, t, cond):
        t_emb = self.time_mlp(t)
        x = torch.cat([x_t, cond], dim=1)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.norm_out(x)
        x = nn.functional.silu(x)
        return self.proj_out(x)


class AGD(nn.Module):
    """
    Anchor-Guided Diffusion (AGD).

    Combines a standard diffusion training loss with a deterministic anchor
    loss at the critical timestep k* where alpha_cumprod is closest to 0.5.
    """

    def __init__(self, channels, timesteps=1000, ddim_steps=10, lambda_weight=0.99):
        super().__init__()
        self.timesteps = timesteps
        self.ddim_steps = ddim_steps
        self.lambda_weight = lambda_weight

        self.denoise_net = ConditionalDenoisingNet(channels, num_blocks=3)

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))

        # Anchor timestep k*: where alpha_cumprod is closest to 0.5
        self.k_star = torch.argmin(torch.abs(alphas_cumprod - 0.5)).item()
        alpha_k_star = alphas_cumprod[self.k_star]
        self.register_buffer(
            'anchor_scale',
            torch.sqrt(alpha_k_star) / torch.sqrt(1.0 - alpha_k_star)
        )

    def forward_train(self, z_true, z_cond):
        """Compute the combined AGD training loss."""
        B = z_true.shape[0]
        device = z_true.device

        # 1) Standard diffusion loss
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise = torch.randn_like(z_true)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
        z_t = sqrt_alpha_t * z_true + sqrt_one_minus_alpha_t * noise

        pred_noise = self.denoise_net(z_t, t, z_cond)
        loss_diff = nn.MSELoss()(pred_noise, noise)

        # 2) Deterministic anchor loss at k*
        t_anchor = torch.full((B,), self.k_star, device=device, dtype=torch.long)
        z_zero = torch.zeros_like(z_true)
        pred_anchor = self.denoise_net(z_zero, t_anchor, z_cond)
        target_anchor = -self.anchor_scale * z_true
        loss_anchor = nn.MSELoss()(pred_anchor, target_anchor)

        # 3) Combined AGD loss
        return self.lambda_weight * loss_diff + (1 - self.lambda_weight) * loss_anchor

    @torch.no_grad()
    def sample_ddim(self, z_cond):
        """DDIM accelerated sampling at inference time."""
        B, C, H, W = z_cond.shape
        device = z_cond.device

        z_t = torch.randn((B, C, H, W), device=device)
        step_size = self.timesteps // self.ddim_steps
        time_steps = torch.arange(
            self.timesteps - 1, -1, -step_size, device=device
        ).long()

        for t in time_steps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.denoise_net(z_t, t_batch, z_cond)

            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (self.alphas_cumprod[t - step_size]
                          if t - step_size >= 0
                          else torch.tensor(1.0, device=device))

            pred_x0 = (z_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            dir_xt = torch.sqrt(1 - alpha_prev) * pred_noise
            z_t = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

        return z_t


# ==============================================================================
# Section 7: ST-DiODE — Full Model Assembly
# ==============================================================================

class ST_DiODE(nn.Module):
    """
    Spatiotemporal Diffusion-ODE model (ST-DiODE).

    Architecture:
        StateNet  -->  DualStreamTranslator (ContDyn + SpecDyn)  -->  AGD  -->  FrameNet
    """

    def __init__(self, shape_in, T_out=None, hid_S=16, hid_T=256, N_S=4, N_T=8,
                 incep_ker=[3, 5, 7, 11], groups=8):
        super().__init__()
        T_in, C, H, W = shape_in
        self.T_in = T_in
        self.T_out = T_out if T_out is not None else T_in

        self.enc = StateNet(C, hid_S, N_S)
        self.translator = DualStreamTranslator(
            T_in * hid_S, hid_T, T_in, self.T_out, incep_ker, groups
        )
        self.agd = AGD(
            channels=self.T_out * hid_S,
            timesteps=1000, ddim_steps=10, lambda_weight=0.99
        )
        self.dec = FrameNet(hid_S, C, N_S)

    def forward(self, observations, target_y=None):
        """
        Args:
            observations: Input sequence [B, T_in, C, H, W].
            target_y:     Ground-truth future frames [B, T_out, C, H, W] (training only).

        Returns:
            Training:  (deterministic predictions, AGD loss)
            Inference: diffusion-sampled predictions
        """
        B, T_in, C, H, W = observations.shape
        obs_flat = observations.view(B * self.T_in, C, H, W)

        # Encode observations into latent states
        latent, shallow_mem = self.enc(obs_flat)
        _, C_hid, H_hid, W_hid = latent.shape

        # Dual-stream temporal evolution
        state_seq = latent.view(B, self.T_in, C_hid, H_hid, W_hid)
        evolved_seq = self.translator(state_seq)
        z_cond = evolved_seq.reshape(B, self.T_out * C_hid, H_hid, W_hid)

        # Broadcast shallow skip connection to match T_out
        if self.T_out != self.T_in:
            s_shape = shallow_mem.shape[1:]
            shallow_seq = shallow_mem.view(B, self.T_in, *s_shape)
            shallow_mem = shallow_seq[:, -1:].expand(B, self.T_out, *s_shape) \
                .reshape(B * self.T_out, *s_shape)
        else:
            shallow_mem = shallow_mem.view(B * self.T_out, *shallow_mem.shape[1:])

        # --- Training ---
        if self.training and target_y is not None:
            target_flat = target_y.view(B * self.T_out, C, H, W)
            z_true, _ = self.enc(target_flat)
            z_true = z_true.view(B, self.T_out * C_hid, H_hid, W_hid)

            agd_loss = self.agd.forward_train(z_true=z_true, z_cond=z_cond)

            # Deterministic branch output for backbone stabilization
            y_det = self.dec(
                z_cond.view(B * self.T_out, C_hid, H_hid, W_hid), shallow_mem
            )
            y_det = y_det.view(B, self.T_out, C, H, W)
            return y_det, agd_loss

        # --- Inference (DDIM sampling) ---
        z_sampled = self.agd.sample_ddim(z_cond)
        z_sampled = z_sampled.view(B * self.T_out, C_hid, H_hid, W_hid)
        predictions = self.dec(z_sampled, shallow_mem)
        return predictions.view(B, self.T_out, C, H, W)
