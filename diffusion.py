import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# schedules
def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(max=0.999).float()

def sigmoid_schedule(T, beta_start=1e-4, beta_end=0.02, tau=6.0):
    x = torch.linspace(-tau, tau, T)
    s = torch.sigmoid(x)
    s = (s - s.min()) / (s.max() - s.min())
    return beta_start + (beta_end - beta_start) * s


def make_schedule(name, T):
    if name == "linear":  return linear_schedule(T)
    if name == "cosine":  return cosine_schedule(T)
    if name == "sigmoid": return sigmoid_schedule(T)
    raise ValueError(f"Unknown schedule: {name}")


class Diffusion:
    """DDPM forward+reverse. see Ho et al 2020"""
    def __init__(self, T=1000, schedule="linear", prediction="epsilon", device="cuda"):
        self.T = T
        self.prediction = prediction
        self.device = device

        betas = make_schedule(schedule, T).to(device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_bar = alphas_bar

        # q(x_t | x_0) coefs
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)

        # reverse process coefs
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
        self.beta_over_sqrt_one_minus_alphas_bar = betas / torch.sqrt(1 - alphas_bar)

        # true posterior variance (DDPM eq. 7), matters for cosine
        alphas_bar_prev = torch.cat([torch.ones(1, device=device), alphas_bar[:-1]])
        posterior_var = (1.0 - alphas_bar_prev) / (1.0 - alphas_bar) * betas
        self.posterior_sigma = torch.sqrt(posterior_var.clamp(min=1e-20))

        # posterior mean coefficients for the x0-parameterized reverse step:
        # mean = coef1 * x0_hat + coef2 * x_t
        # used with a clamped x0_hat to prevent cosine first-step drift.
        self.posterior_mean_coef1 = torch.sqrt(alphas_bar_prev) * betas / (1.0 - alphas_bar)
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)

    def training_step(self, model, x0):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        eps = torch.randn_like(x0)

        # x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
        a = self.sqrt_alphas_bar[t][:, None, None, None]
        b = self.sqrt_one_minus_alphas_bar[t][:, None, None, None]
        x_t = a * x0 + b * eps

        pred = model(x_t, t)

        if self.prediction == "epsilon":
            return F.mse_loss(pred, eps)
        elif self.prediction == "x0":
            return F.mse_loss(pred, x0)
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction}")

    @torch.no_grad()
    def sample(self, model, shape):
        x_t = torch.randn(shape, device=self.device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            pred = model(x_t, t_batch)

            if self.prediction == "x0":
                x0_hat = pred
            else:
                x0_hat = (x_t - self.sqrt_one_minus_alphas_bar[t] * pred) / self.sqrt_alphas_bar[t]

            # clamp x0 to data range
            x0_hat = x0_hat.clamp(-1.0, 1.0)

            # posterior mean from clamped x0_hat (DDPM eq. 7)
            mean = self.posterior_mean_coef1[t] * x0_hat + self.posterior_mean_coef2[t] * x_t

            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + self.posterior_sigma[t] * noise
            else:
                x_t = mean
        return x_t
