import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import OrpheusConfig

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.full((hidden_size,), 0.4))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class ProjectionLayer(nn.Module):
    def __init__(self, stack_factor: int = 8):
        super().__init__()
        self.stack_factor = stack_factor

    def _pad_and_stack(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_embeds.shape
        audio_embeds = F.pad(
            audio_embeds, (0, 0, 0, self.stack_factor - T % self.stack_factor)
        )
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(
            B, T // self.stack_factor, C * self.stack_factor
        )
        return audio_embeds
    

class OrpheusProjector(ProjectionLayer):
    def __init__(self, config: OrpheusConfig):
        self.hidden_dim = config.hidden_size
        super().__init__(config.stack_factor)
        self.ln_pre = RMSNorm(config.audio_hidden_size * self.stack_factor)
        self.linear_1 = nn.Linear(
            config.audio_hidden_size * self.stack_factor,
            self.hidden_dim,
            bias=False,
        )
        self.act = SwiGLU()
        self.linear_2 = nn.Linear(
            self.hidden_dim // 2, self.hidden_dim, bias=False
        )
        self.ln_post = RMSNorm(self.hidden_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states
