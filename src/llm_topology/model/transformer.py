from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ActivationKind, InitScheme, ModelSpec, NormPlacement, PositionEncoding


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    hidden_states: list[torch.Tensor]
    final_hidden: torch.Tensor
    attentions: list[torch.Tensor] | None = None
    ffn_activations: list[torch.Tensor] | None = None


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


def _activation(kind: ActivationKind) -> nn.Module:
    if kind is ActivationKind.GELU:
        return nn.GELU()
    if kind is ActivationKind.SILU:
        return nn.SiLU()
    return nn.ReLU()


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        frequencies = torch.outer(positions, inv_freq)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, time, head_dim]
        time = x.shape[-2]
        cos = self.cos[:time][None, None, :, :]
        sin = self.sin[:time][None, None, :, :]
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.n_heads = spec.n_heads
        self.head_dim = spec.head_dim
        self.dropout = spec.dropout
        self.qkv = nn.Linear(spec.d_model, 3 * spec.d_model, bias=spec.use_bias)
        self.proj = nn.Linear(spec.d_model, spec.d_model, bias=spec.use_bias)
        self.rope = (
            RotaryEmbedding(self.head_dim, spec.max_seq_len)
            if spec.position_encoding is PositionEncoding.ROPE
            else None
        )
        mask = torch.tril(torch.ones(spec.max_seq_len, spec.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, time, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:time, :time]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, dim)
        out = self.proj(out)
        return out, attention if return_attention else None


class MLP(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.fc1 = nn.Linear(spec.d_model, spec.d_ff, bias=spec.use_bias)
        self.fc2 = nn.Linear(spec.d_ff, spec.d_model, bias=spec.use_bias)
        self.activation = _activation(spec.activation)
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, x: torch.Tensor, *, return_activation: bool = False):
        hidden = self.activation(self.fc1(x))
        output = self.fc2(self.dropout(hidden))
        return output, (hidden if return_activation else None)


class Block(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.attn = CausalSelfAttention(spec)
        self.mlp = MLP(spec)
        self.norm1 = LayerNorm(spec.d_model)
        self.norm2 = LayerNorm(spec.d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
        return_ffn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.spec.norm_placement is NormPlacement.PRE:
            attn_out, attention = self.attn(self.norm1(x), return_attention=return_attention)
            x = x + attn_out
            mlp_out, ffn = self.mlp(self.norm2(x), return_activation=return_ffn)
            x = x + mlp_out
        else:
            attn_out, attention = self.attn(x, return_attention=return_attention)
            x = self.norm1(x + attn_out)
            mlp_out, ffn = self.mlp(x, return_activation=return_ffn)
            x = self.norm2(x + mlp_out)
        return x, attention, ffn


class TopologyTransformer(nn.Module):
    """A small decoder-only transformer built specifically for this project."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        vocab_size: int,
        n_structures: int,
        max_dim: int = 2,
    ) -> None:
        super().__init__()
        spec.validate_heads()
        self.spec = spec
        self.vocab_size = vocab_size
        self.n_structures = n_structures
        self.max_dim = max_dim
        self.token_embedding = nn.Embedding(vocab_size, spec.d_model)
        self.position_embedding = (
            nn.Embedding(spec.max_seq_len, spec.d_model)
            if spec.position_encoding is PositionEncoding.LEARNED
            else None
        )
        self.embedding_dropout = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList([Block(spec) for _ in range(spec.n_layers)])
        self.final_norm = LayerNorm(spec.d_model)
        self.lm_head = nn.Linear(spec.d_model, vocab_size, bias=False)
        if spec.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.structure_head = nn.Linear(spec.d_model, n_structures)
        self.genus_head = nn.Linear(spec.d_model, 1)
        self.components_head = nn.Linear(spec.d_model, 1)
        self.betti_head = nn.Linear(spec.d_model, max_dim + 1)
        self.ph_head = nn.Linear(spec.d_model, max_dim + 1)
        self.apply_init()

    def apply_init(self) -> None:
        init = self.spec.init
        with torch.random.fork_rng():
            torch.manual_seed(init.seed)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    self._init_tensor(module.weight, init.scheme, init.std, init.mean)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    self._init_tensor(module.weight, init.scheme, init.std, init.mean)
                elif isinstance(module, LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            # Norm parameters and zero-ish heads need deterministic overrides.
            for module in self.modules():
                if isinstance(module, LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _init_tensor(
        tensor: torch.Tensor,
        scheme: InitScheme,
        std: float,
        mean: float,
    ) -> None:
        if scheme is InitScheme.NORMAL:
            nn.init.normal_(tensor, mean=mean, std=std)
        elif scheme is InitScheme.UNIFORM:
            nn.init.uniform_(tensor, -std, std)
        elif scheme is InitScheme.XAVIER_UNIFORM:
            nn.init.xavier_uniform_(tensor)
        elif scheme is InitScheme.XAVIER_NORMAL:
            nn.init.xavier_normal_(tensor)
        elif scheme is InitScheme.ORTHOGONAL:
            nn.init.orthogonal_(tensor)
        elif scheme is InitScheme.ZERO:
            nn.init.zeros_(tensor)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def aux_forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "structure_logits": self.structure_head(hidden),
            "genus": self.genus_head(hidden).squeeze(-1),
            "n_components": self.components_head(hidden).squeeze(-1),
            "betti": self.betti_head(hidden),
            "ph": self.ph_head(hidden),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_attentions: bool = False,
        return_ffn: bool = False,
    ) -> TransformerOutput:
        batch, time = input_ids.shape
        if time > self.spec.max_seq_len:
            raise ValueError(
                f"sequence length {time} exceeds model context {self.spec.max_seq_len}"
            )
        positions = torch.arange(time, device=input_ids.device)
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(positions)[None, :, :]
        hidden = self.embedding_dropout(hidden)
        hidden_states = [hidden]
        attentions: list[torch.Tensor] = []
        ffn_activations: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, attention, ffn = block(
                hidden,
                return_attention=return_attentions,
                return_ffn=return_ffn,
            )
            hidden_states.append(hidden)
            if attention is not None:
                attentions.append(attention)
            if ffn is not None:
                ffn_activations.append(ffn)
        final_hidden = self.final_norm(hidden)
        logits = self.lm_head(final_hidden)
        return TransformerOutput(
            logits=logits,
            hidden_states=hidden_states,
            final_hidden=final_hidden,
            attentions=attentions if return_attentions else None,
            ffn_activations=ffn_activations if return_ffn else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 80,
        eos_id: int | None = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        output = input_ids
        for _ in range(max_new_tokens):
            if output.shape[1] >= self.spec.max_seq_len:
                break
            logits = self.forward(output).logits[:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            output = torch.cat([output, next_token], dim=1)
            if eos_id is not None and int(next_token[0, 0]) == eos_id:
                break
        return output
