#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KuaiRand-1K UniFormer-style multi-target ranking model."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import NUM_TIME_BUCKETS


LOGGER = logging.getLogger("kuairand")


@dataclass(frozen=True)
class TokenFeatureSpec:
    name: str
    start: int
    end: int
    feat_len: int
    num_embeddings: int
    padding_idx: int | None
    is_identifier: bool


def build_token_feature_specs(side_schema: dict[str, Any]) -> list[TokenFeatureSpec]:
    specs = []
    for name, spec in side_schema["features"].items():
        specs.append(
            TokenFeatureSpec(
                name=name,
                start=int(spec["start"]),
                end=int(spec["end"]),
                feat_len=int(spec["feat_len"]),
                num_embeddings=int(spec["num_embeddings"]),
                padding_idx=spec["padding_idx"],
                is_identifier=bool(spec["is_identifier"]),
            )
        )
    return specs


class FeatureEmbeddingBank(nn.Module):
    """One embedding table per flattened static feature.

    A bank can be shared by several tokenizers. The target item tokenizer and
    all sequence-domain tokenizers share the same item bank.
    """

    def __init__(self, specs: list[TokenFeatureSpec], emb_dim: int) -> None:
        super().__init__()
        self.specs = specs
        self.emb_dim = int(emb_dim)
        self.embeddings = nn.ModuleList()
        LOGGER.info(
            "Create FeatureEmbeddingBank: tables=%d emb_dim=%d rows=%s",
            len(specs),
            self.emb_dim,
            [(spec.name, spec.num_embeddings) for spec in specs],
        )
        for spec in specs:
            emb = nn.Embedding(
                spec.num_embeddings,
                self.emb_dim,
                padding_idx=spec.padding_idx,
                sparse=True,
            )
            emb._kuairand_num_embeddings = spec.num_embeddings
            emb._kuairand_padding_idx = spec.padding_idx
            emb._kuairand_feature_name = spec.name
            self.embeddings.append(emb)
        self.reset_parameters()
        LOGGER.info("FeatureEmbeddingBank initialized: tables=%d", len(specs))

    def reset_parameters(self) -> None:
        for emb in self.embeddings:
            nn.init.xavier_normal_(emb.weight.data)
            padding_idx = getattr(emb, "_kuairand_padding_idx", None)
            if padding_idx is not None:
                emb.weight.data[int(padding_idx)].zero_()

    @staticmethod
    def _pool_feature_embedding(values: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        values = values.long().clamp(min=0, max=emb.num_embeddings - 1)
        emb_all = emb(values)
        if values.shape[-1] == 1:
            return emb_all.squeeze(-2)
        mask = (values != 0).unsqueeze(-1).to(emb_all.dtype)
        count = mask.sum(dim=-2).clamp(min=1.0)
        return (emb_all * mask).sum(dim=-2) / count

    def embed_flat(self, features: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for spec, emb in zip(self.specs, self.embeddings):
            vals = features[:, spec.start:spec.end]
            out.append(self._pool_feature_embedding(vals, emb))
        return out

    def embed_flat_concat(self, features: torch.Tensor) -> torch.Tensor:
        emb_list = self.embed_flat(features)
        if emb_list:
            return torch.cat(emb_list, dim=-1)
        return features.new_zeros(features.shape[0], 1, dtype=torch.float)

    def embed_sequence(self, features: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for spec, emb in zip(self.specs, self.embeddings):
            vals = features[:, :, spec.start:spec.end]
            out.append(self._pool_feature_embedding(vals, emb))
        return out


class RankMixerTokenizer(nn.Module):
    """RankMixer tokenizer with optional external embedding bank."""

    def __init__(
        self,
        specs: list[TokenFeatureSpec],
        emb_dim: int,
        d_model: int,
        num_tokens: int,
        embedding_bank: FeatureEmbeddingBank | None = None,
        shuffle_seed: int = -1,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        self.specs = specs
        self.emb_dim = int(emb_dim)
        self.d_model = int(d_model)
        self.num_tokens = int(num_tokens)
        if embedding_bank is None:
            self.embedding_bank = FeatureEmbeddingBank(specs, emb_dim)
        else:
            object.__setattr__(self, "embedding_bank", embedding_bank)
        self.feature_order = list(range(len(specs)))
        self.shuffle_seed = int(shuffle_seed)
        if self.shuffle_seed >= 0:
            random.Random(self.shuffle_seed).shuffle(self.feature_order)

        total_emb_dim = max(1, len(specs) * self.emb_dim)
        self.chunk_dim = math.ceil(total_emb_dim / self.num_tokens)
        self.padded_total_dim = self.chunk_dim * self.num_tokens
        self.pad_size = self.padded_total_dim - total_emb_dim
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.SiLU(),
            )
            for _ in range(self.num_tokens)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        emb_list = self.embedding_bank.embed_flat(features)
        emb_list = [emb_list[idx] for idx in self.feature_order]
        if emb_list:
            cat = torch.cat(emb_list, dim=-1)
        else:
            cat = features.new_zeros(features.shape[0], 1, dtype=torch.float)
        if self.pad_size > 0:
            cat = F.pad(cat, (0, self.pad_size))
        chunks = cat.split(self.chunk_dim, dim=-1)
        tokens = [proj(chunk).unsqueeze(1) for proj, chunk in zip(self.token_projs, chunks)]
        return torch.cat(tokens, dim=1)


class SequenceDomainTokenizer(nn.Module):
    """Tokenize one positive-history domain with shared item embeddings."""

    def __init__(
        self,
        item_embedding_bank: FeatureEmbeddingBank,
        emb_dim: int,
        d_model: int,
        num_time_buckets: int = NUM_TIME_BUCKETS,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "item_embedding_bank", item_embedding_bank)
        self.proj = nn.Sequential(
            nn.Linear(len(item_embedding_bank.specs) * emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.time_emb = nn.Embedding(num_time_buckets, d_model, padding_idx=0)
        nn.init.xavier_normal_(self.time_emb.weight.data)
        self.time_emb.weight.data[0].zero_()

    def forward(
        self,
        item_features: torch.Tensor,
        time_bucket: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb_list = self.item_embedding_bank.embed_sequence(item_features)
        tokens = self.project_concat_emb(torch.cat(emb_list, dim=-1), time_bucket, padding_mask)
        return tokens

    def project_concat_emb(
        self,
        concat_emb: torch.Tensor,
        time_bucket: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.proj(concat_emb) + self.time_emb(time_bucket.long())
        return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(d_model) * int(hidden_mult)
        self.fc = nn.Linear(d_model, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.out(self.dropout(x1 * F.silu(x2)))


class FeedForwardResidual(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class PerTokenFFN(nn.Module):
    def __init__(self, num_tokens: int, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(d_model) * int(hidden_mult)
        self.in_ffn = nn.Parameter(torch.empty(num_tokens, d_model, hidden_dim))
        self.out_ffn = nn.Parameter(torch.empty(num_tokens, hidden_dim, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.in_ffn, -1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_model))
        nn.init.uniform_(self.out_ffn, -1.0 / math.sqrt(hidden_dim), 1.0 / math.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("btd,tdh->bth", x, self.in_ffn)
        x = self.dropout(F.gelu(x))
        return torch.einsum("bth,thd->btd", x, self.out_ffn)


class PerTokenSwiGLU(nn.Module):
    """SwiGLU FFN with independent parameters for every token position."""

    def __init__(self, num_tokens: int, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("Per-token FFN hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.in_ffn = nn.Parameter(torch.empty(num_tokens, d_model, hidden_dim * 2))
        self.out_ffn = nn.Parameter(torch.empty(num_tokens, hidden_dim, d_model))
        self.dropout = nn.Dropout(dropout)
        self._coactivation_collector: Any | None = None
        nn.init.uniform_(self.in_ffn, -1.0 / math.sqrt(d_model), 1.0 / math.sqrt(d_model))
        nn.init.uniform_(self.out_ffn, -1.0 / math.sqrt(hidden_dim), 1.0 / math.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.einsum("btd,tdh->bth", x, self.in_ffn).chunk(2, dim=-1)
        hidden = value * F.silu(gate)
        if self._coactivation_collector is not None:
            self._coactivation_collector.update(hidden.detach(), self.out_ffn.detach())
        x = self.dropout(hidden)
        return torch.einsum("bth,thd->btd", x, self.out_ffn)

    def set_coactivation_collector(self, collector: Any | None) -> None:
        self._coactivation_collector = collector


def _permute_packed_input_channels_(
    tensor: torch.Tensor,
    source_channel_indices: torch.Tensor,
    hidden_dim: int,
) -> None:
    """Reorder packed value/gate channels with bounded temporary memory."""

    with torch.no_grad():
        for token_idx in range(int(tensor.shape[0])):
            order = source_channel_indices[token_idx].reshape(-1).to(tensor.device)
            token_tensor = tensor[token_idx]
            value = token_tensor[:, :hidden_dim].index_select(1, order)
            gate = token_tensor[:, hidden_dim:].index_select(1, order)
            token_tensor[:, :hidden_dim].copy_(value)
            token_tensor[:, hidden_dim:].copy_(gate)


def _permute_output_channels_(
    tensor: torch.Tensor,
    source_channel_indices: torch.Tensor,
) -> None:
    """Reorder output-projection rows with bounded temporary memory."""

    with torch.no_grad():
        for token_idx in range(int(tensor.shape[0])):
            order = source_channel_indices[token_idx].reshape(-1).to(tensor.device)
            reordered = tensor[token_idx].index_select(0, order)
            tensor[token_idx].copy_(reordered)


def _permute_parameter_and_adam_state_(
    parameter: nn.Parameter,
    source_channel_indices: torch.Tensor,
    hidden_dim: int,
    packed_input: bool,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    """Apply the expert permutation to a Parameter and all shape-matched slots."""

    tensors = [parameter.data]
    if optimizer is not None:
        state = optimizer.state.get(parameter, {})
        tensors.extend(
            value
            for value in state.values()
            if torch.is_tensor(value) and tuple(value.shape) == tuple(parameter.shape)
        )
    for tensor in tensors:
        if packed_input:
            _permute_packed_input_channels_(tensor, source_channel_indices, hidden_dim)
        else:
            _permute_output_channels_(tensor, source_channel_indices)


class PerTokenSparseMoE(nn.Module):
    """Per-position 1-always + top-1-of-N sparse SwiGLU.

    Expert weights stay packed in the original two Parameters.  Conversion
    permutes the Dense hidden channels into contiguous expert buckets, which
    preserves the Parameter identities and lets AdamW state migrate exactly.
    Routed samples are dispatched before their expert projections, so only the
    selected routed expert is evaluated.
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
        num_routed_experts: int,
        router_init_std: float = 0.01,
        in_ffn: nn.Parameter | None = None,
        out_ffn: nn.Parameter | None = None,
        source_channel_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.num_routed_experts = int(num_routed_experts)
        self.num_total_experts = self.num_routed_experts + 1
        if self.num_routed_experts <= 0:
            raise ValueError("num_routed_experts must be positive")
        if self.hidden_dim % self.num_total_experts != 0:
            raise ValueError(
                "hidden_dim must be divisible by 1 + num_routed_experts: "
                f"{self.hidden_dim} vs {self.num_total_experts}"
            )
        self.expert_hidden_dim = self.hidden_dim // self.num_total_experts
        if in_ffn is None:
            in_ffn = nn.Parameter(
                torch.empty(self.num_tokens, self.d_model, self.hidden_dim * 2)
            )
            nn.init.uniform_(
                in_ffn,
                -1.0 / math.sqrt(self.d_model),
                1.0 / math.sqrt(self.d_model),
            )
        if out_ffn is None:
            out_ffn = nn.Parameter(
                torch.empty(self.num_tokens, self.hidden_dim, self.d_model)
            )
            nn.init.uniform_(
                out_ffn,
                -1.0 / math.sqrt(self.hidden_dim),
                1.0 / math.sqrt(self.hidden_dim),
            )
        self.in_ffn = in_ffn
        self.out_ffn = out_ffn
        self.router = nn.Parameter(
            self.in_ffn.new_empty(
                self.num_tokens,
                self.d_model,
                self.num_routed_experts,
            )
        )
        nn.init.trunc_normal_(self.router, mean=0.0, std=float(router_init_std))
        self.dropout = nn.Dropout(dropout)

        if source_channel_indices is None:
            source_channel_indices = torch.arange(self.hidden_dim, dtype=torch.long)
            source_channel_indices = source_channel_indices.view(
                1, self.num_total_experts, self.expert_hidden_dim
            ).expand(self.num_tokens, -1, -1).clone()
        expected_shape = (
            self.num_tokens,
            self.num_total_experts,
            self.expert_hidden_dim,
        )
        if tuple(source_channel_indices.shape) != expected_shape:
            raise ValueError(
                "source_channel_indices shape mismatch: "
                f"{tuple(source_channel_indices.shape)} vs {expected_shape}"
            )
        self.register_buffer(
            "source_channel_indices",
            source_channel_indices.detach().to(dtype=torch.long, device="cpu").clone(),
            persistent=True,
        )
        self._last_aux_loss: torch.Tensor | None = None
        self._last_router_metrics: dict[str, torch.Tensor] = {}

    @classmethod
    def from_dense(
        cls,
        dense: PerTokenSwiGLU,
        source_channel_indices: torch.Tensor,
        num_routed_experts: int,
        optimizer: torch.optim.Optimizer | None,
        router_init_std: float = 0.01,
    ) -> "PerTokenSparseMoE":
        hidden_dim = int(dense.hidden_dim)
        num_tokens, d_model, packed_hidden = dense.in_ffn.shape
        if int(packed_hidden) != hidden_dim * 2:
            raise ValueError("Dense packed input projection has an invalid shape")

        flat = source_channel_indices.reshape(int(num_tokens), -1)
        expected = torch.arange(hidden_dim, dtype=torch.long).expand(int(num_tokens), -1)
        if not torch.equal(torch.sort(flat.cpu(), dim=1).values, expected):
            raise ValueError("Each MoE partition must be an exact hidden-channel permutation")

        _permute_parameter_and_adam_state_(
            dense.in_ffn,
            source_channel_indices,
            hidden_dim,
            packed_input=True,
            optimizer=optimizer,
        )
        _permute_parameter_and_adam_state_(
            dense.out_ffn,
            source_channel_indices,
            hidden_dim,
            packed_input=False,
            optimizer=optimizer,
        )
        return cls(
            num_tokens=int(num_tokens),
            d_model=int(d_model),
            hidden_dim=hidden_dim,
            dropout=float(dense.dropout.p),
            num_routed_experts=num_routed_experts,
            router_init_std=router_init_std,
            in_ffn=dense.in_ffn,
            out_ffn=dense.out_ffn,
            source_channel_indices=source_channel_indices,
        )

    def dense_equivalent_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all packed channels; useful for conversion audits."""

        value, gate = torch.einsum("btd,tdh->bth", x, self.in_ffn).chunk(2, dim=-1)
        hidden = self.dropout(value * F.silu(gate))
        return torch.einsum("bth,thd->btd", hidden, self.out_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or int(x.shape[1]) != self.num_tokens:
            raise ValueError(
                f"Sparse MoE expects [B, {self.num_tokens}, D], got {tuple(x.shape)}"
            )

        expert_width = self.expert_hidden_dim
        hidden_dim = self.hidden_dim

        always_value = torch.einsum(
            "btd,tdh->bth", x, self.in_ffn[:, :, :expert_width]
        )
        always_gate = torch.einsum(
            "btd,tdh->bth",
            x,
            self.in_ffn[:, :, hidden_dim : hidden_dim + expert_width],
        )
        always_hidden = self.dropout(always_value * F.silu(always_gate))
        always_output = torch.einsum(
            "bth,thd->btd", always_hidden, self.out_ffn[:, :expert_width, :]
        )

        router_logits = torch.einsum("btd,tde->bte", x, self.router)
        # Keep routing probabilities in FP32 for stable routing statistics.  Under
        # autocast, x itself remains FP32 while expert matmuls return BF16, so the
        # selected gate must be cast at the point where expert outputs are merged.
        router_probs = F.softmax(router_logits.float(), dim=-1)
        top1_indices = router_probs.argmax(dim=-1)
        routed_outputs = []
        for token_idx in range(self.num_tokens):
            token_output = torch.zeros_like(always_output[:, token_idx, :])
            token_routes = top1_indices[:, token_idx]
            token_input = x[:, token_idx, :]
            for routed_expert in range(self.num_routed_experts):
                sample_indices = torch.nonzero(
                    token_routes == routed_expert,
                    as_tuple=False,
                ).flatten()
                if sample_indices.numel() == 0:
                    continue
                selected_input = token_input.index_select(0, sample_indices)
                start = (routed_expert + 1) * expert_width
                end = start + expert_width
                selected_value = selected_input @ self.in_ffn[
                    token_idx, :, start:end
                ]
                selected_gate = selected_input @ self.in_ffn[
                    token_idx, :, hidden_dim + start : hidden_dim + end
                ]
                selected_hidden = self.dropout(
                    selected_value * F.silu(selected_gate)
                )
                selected_output = selected_hidden @ self.out_ffn[
                    token_idx, start:end, :
                ]
                selected_prob = (
                    router_probs[sample_indices, token_idx, routed_expert]
                    .unsqueeze(-1)
                    .to(dtype=selected_output.dtype)
                )
                token_output = token_output.index_add(
                    0,
                    sample_indices,
                    selected_output * selected_prob,
                )
            routed_outputs.append(token_output)
        routed_output = torch.stack(routed_outputs, dim=1)

        hard_usage = F.one_hot(
            top1_indices,
            num_classes=self.num_routed_experts,
        ).to(router_probs.dtype).mean(dim=0)
        mean_probability = router_probs.mean(dim=0)
        balance_per_token = self.num_routed_experts * (
            hard_usage * mean_probability
        ).sum(dim=-1)
        self._last_aux_loss = balance_per_token.mean()
        with torch.no_grad():
            entropy = -(router_probs.float() * torch.log(router_probs.float() + 1e-10)).sum(-1)
            self._last_router_metrics = {
                "balance_loss": self._last_aux_loss.detach(),
                "entropy": entropy.mean().detach(),
                "max_probability": router_probs.float().amax(dim=-1).mean().detach(),
                "usage_min": hard_usage.amin().detach(),
                "usage_max": hard_usage.amax().detach(),
            }
        return always_output + routed_output

    def auxiliary_loss(self) -> torch.Tensor | None:
        return self._last_aux_loss

    def router_metrics(self) -> dict[str, torch.Tensor]:
        return dict(self._last_router_metrics)


class SelfAttentionResidual(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return x + out


class CrossAttentionResidual(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.q_norm = RMSNorm(d_model)
        self.kv_norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        safe_mask = key_padding_mask
        all_pad = None
        if key_padding_mask is not None:
            all_pad = key_padding_mask.all(dim=1)
            if bool(all_pad.any()):
                safe_mask = key_padding_mask.clone()
                safe_mask[all_pad, 0] = False

        out, _ = self.attn(
            self.q_norm(query),
            self.kv_norm(key_value),
            self.kv_norm(key_value),
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        if all_pad is not None and bool(all_pad.any()):
            out = out.masked_fill(all_pad[:, None, None], 0.0)
        return query + out


class FIMLayer(nn.Module):
    """Feature Interaction Module layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_seq_query_tokens: int,
        num_nonseq_tokens: int,
        seq_domains: list[str],
        s_hidden_multi: int,
        ns_hidden_dim: int,
        dropout: float,
        moe_num_routed_experts: int = 0,
        moe_router_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_seq_query_tokens = int(num_seq_query_tokens)
        self.seq_domains = list(seq_domains)
        self.domain_ca = nn.ModuleDict({
            domain: CrossAttentionResidual(d_model, num_heads, dropout)
            for domain in self.seq_domains
        })
        self.domain_ffn = nn.ModuleDict({
            domain: FeedForwardResidual(d_model, s_hidden_multi, dropout)
            for domain in self.seq_domains
        })
        total_tokens = num_seq_query_tokens + num_nonseq_tokens
        self.self_attn = SelfAttentionResidual(d_model, num_heads, dropout)
        self.pt_norm = RMSNorm(d_model)
        if int(moe_num_routed_experts) > 0:
            self.per_token_ffn = PerTokenSparseMoE(
                total_tokens,
                d_model,
                ns_hidden_dim,
                dropout,
                num_routed_experts=int(moe_num_routed_experts),
                router_init_std=float(moe_router_init_std),
            )
        else:
            self.per_token_ffn = PerTokenSwiGLU(
                total_tokens,
                d_model,
                ns_hidden_dim,
                dropout,
            )

    def forward(
        self,
        seq_query_tokens: torch.Tensor,
        nonseq_tokens: torch.Tensor,
        seq_tokens: dict[str, torch.Tensor],
        seq_padding_masks: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoded = []
        for domain in self.seq_domains:
            q = self.domain_ca[domain](
                seq_query_tokens,
                seq_tokens[domain],
                seq_padding_masks[domain],
            )
            decoded.append(self.domain_ffn[domain](q))
        fused_seq_tokens = sum(decoded) / len(decoded) if decoded else seq_query_tokens

        combined = torch.cat([fused_seq_tokens, nonseq_tokens], dim=1)
        combined = self.self_attn(combined)
        combined = combined + self.per_token_ffn(self.pt_norm(combined))
        return (
            combined[:, :self.num_seq_query_tokens],
            combined[:, self.num_seq_query_tokens:],
        )


class TaskTokenInitializer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tasks: int,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.task_emb = nn.Embedding(num_tasks, d_model)
        self.task_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        pooled = context_tokens.mean(dim=1)
        task_tokens = []
        for task_idx, proj in enumerate(self.task_projs):
            emb = self.task_emb.weight[task_idx].expand(pooled.shape[0], -1)
            task_tokens.append(proj(torch.cat([pooled, emb], dim=-1)).unsqueeze(1))
        return torch.cat(task_tokens, dim=1)


class TIMLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_tasks: int,
        t_hidden_multi: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_attn = CrossAttentionResidual(d_model, num_heads, dropout)
        self.self_attn = SelfAttentionResidual(d_model, num_heads, dropout)
        self.pt_norm = RMSNorm(d_model)
        self.per_token_ffn = PerTokenFFN(
            num_tasks,
            d_model,
            t_hidden_multi,
            dropout,
        )

    def forward(
        self,
        task_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        task_tokens = self.cross_attn(task_tokens, context_tokens)
        task_tokens = self.self_attn(task_tokens)
        task_tokens = task_tokens + self.per_token_ffn(self.pt_norm(task_tokens))
        return task_tokens


class UniFormer(nn.Module):
    def __init__(
        self,
        feature_schema: dict[str, Any],
        task_names: list[str],
        seq_domains: list[str],
        d_model: int = 400,
        emb_dim: int = 128,
        user_tokens: int = 25,
        item_tokens: int = 8,
        num_fim_layers: int = 2,
        num_tim_layers: int = 1,
        num_heads: int = 5,
        s_hidden_multi: int = 2,
        ns_hidden_dim: int | None = None,
        ns_hidden_multi: float | None = None,
        t_hidden_multi: int = 1,
        dropout: float = 0.01,
        ns_shuffle_seed: int = -1,
        item_feature_values: Any | None = None,
        moe_num_routed_experts: int = 0,
        moe_router_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_tim_layers <= 0:
            raise ValueError("num_tim_layers must be positive")
        if ns_hidden_dim is not None and int(ns_hidden_dim) <= 0:
            raise ValueError("ns_hidden_dim must be positive")
        if ns_hidden_multi is not None and float(ns_hidden_multi) <= 0:
            raise ValueError("ns_hidden_multi must be positive")
        multiplier_hidden_dim = None
        if ns_hidden_multi is not None:
            raw_hidden_dim = int(d_model) * float(ns_hidden_multi)
            multiplier_hidden_dim = int(round(raw_hidden_dim))
            if not math.isclose(raw_hidden_dim, multiplier_hidden_dim):
                raise ValueError(
                    "d_model * ns_hidden_multi must produce an integer hidden dimension"
                )
        if ns_hidden_dim is None:
            # This directory is the cold-start Upper (0.3B) experiment. Keep
            # multiplier support only as a compatibility path for older calls.
            ns_hidden_dim = multiplier_hidden_dim or 1800
        elif multiplier_hidden_dim is not None and int(ns_hidden_dim) != multiplier_hidden_dim:
            raise ValueError(
                "ns_hidden_dim and ns_hidden_multi describe different hidden dimensions: "
                f"{ns_hidden_dim} vs {multiplier_hidden_dim}"
            )
        self.task_names = list(task_names)
        self.seq_domains = list(seq_domains)
        self.d_model = int(d_model)
        self.ns_hidden_dim = int(ns_hidden_dim)
        self.moe_num_routed_experts = int(moe_num_routed_experts)
        self.moe_router_init_std = float(moe_router_init_std)
        if self.moe_num_routed_experts < 0:
            raise ValueError("moe_num_routed_experts must be non-negative")
        if (
            self.moe_num_routed_experts > 0
            and self.ns_hidden_dim % (self.moe_num_routed_experts + 1) != 0
        ):
            raise ValueError(
                "ns_hidden_dim must be divisible by 1 + moe_num_routed_experts: "
                f"{self.ns_hidden_dim} vs {self.moe_num_routed_experts + 1}"
            )
        self.user_specs = build_token_feature_specs(feature_schema["user"])
        self.item_specs = build_token_feature_specs(feature_schema["item"])
        if item_feature_values is None:
            raise ValueError("item_feature_values is required for sequence item-feature lookup")
        item_feature_tensor = torch.as_tensor(item_feature_values, dtype=torch.int32)
        if item_feature_tensor.ndim != 2:
            raise ValueError(
                f"item_feature_values must be 2D, got shape={tuple(item_feature_tensor.shape)}"
            )
        expected_item_dim = int(feature_schema["item"]["dim"])
        if int(item_feature_tensor.shape[1]) != expected_item_dim:
            raise ValueError(
                "item_feature_values width does not match item schema dim: "
                f"{item_feature_tensor.shape[1]} vs {expected_item_dim}"
            )
        self.register_buffer(
            "item_feature_values",
            item_feature_tensor.contiguous(),
            persistent=False,
        )
        LOGGER.info(
            "Registered item feature lookup buffer: rows=%d dim=%d size=%.2f MB int32",
            int(item_feature_tensor.shape[0]),
            int(item_feature_tensor.shape[1]),
            item_feature_tensor.numel() * item_feature_tensor.element_size() / 1024 / 1024,
        )

        self.user_tokenizer = RankMixerTokenizer(
            self.user_specs,
            emb_dim=emb_dim,
            d_model=d_model,
            num_tokens=user_tokens,
            shuffle_seed=ns_shuffle_seed,
        )
        self.item_embedding_bank = FeatureEmbeddingBank(self.item_specs, emb_dim)
        self.item_tokenizer = RankMixerTokenizer(
            self.item_specs,
            emb_dim=emb_dim,
            d_model=d_model,
            num_tokens=item_tokens,
            embedding_bank=self.item_embedding_bank,
            shuffle_seed=ns_shuffle_seed,
        )
        self.seq_tokenizers = nn.ModuleDict({
            domain: SequenceDomainTokenizer(
                item_embedding_bank=self.item_embedding_bank,
                emb_dim=emb_dim,
                d_model=d_model,
            )
            for domain in self.seq_domains
        })

        self.num_user_tokens = int(user_tokens)
        self.num_item_tokens = int(item_tokens)
        self.num_nonseq_tokens = self.num_user_tokens + self.num_item_tokens
        self.num_seq_query_tokens = self.num_nonseq_tokens
        self.fim_layers = nn.ModuleList([
            FIMLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_seq_query_tokens=self.num_seq_query_tokens,
                num_nonseq_tokens=self.num_nonseq_tokens,
                seq_domains=self.seq_domains,
                s_hidden_multi=s_hidden_multi,
                ns_hidden_dim=self.ns_hidden_dim,
                dropout=dropout,
                moe_num_routed_experts=self.moe_num_routed_experts,
                moe_router_init_std=self.moe_router_init_std,
            )
            for _ in range(num_fim_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.task_token_initializer = TaskTokenInitializer(
            d_model=d_model,
            num_tasks=len(self.task_names),
        )
        self.tim_layers = nn.ModuleList([
            TIMLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_tasks=len(self.task_names),
                t_hidden_multi=t_hidden_multi,
                dropout=dropout,
            )
            for _ in range(num_tim_layers)
        ])
        self.task_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(len(self.task_names))
        ])

        LOGGER.info(
            "UniFormer: user_tokens=%d item_tokens=%d fim_layers=%d tim_layers=%d "
            "s_hidden_multi=%d ns_hidden_dim=%d effective_ns_hidden_multi=%g "
            "t_hidden_multi=%d moe_routed_experts=%d tasks=%s",
            self.num_user_tokens,
            self.num_item_tokens,
            num_fim_layers,
            num_tim_layers,
            s_hidden_multi,
            self.ns_hidden_dim,
            self.ns_hidden_dim / self.d_model,
            t_hidden_multi,
            self.moe_num_routed_experts,
            self.task_names,
        )

    def _build_sequence_concat_embeddings(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Build sequence feature embeddings via one batch-level unique lookup.

        The three sequence domains are concatenated first, then unique video ids
        are embedded once. Each domain gathers back its [B, L, total_emb_dim]
        tensor through the inverse indices returned by torch.unique.
        """

        num_items = int(self.item_feature_values.shape[0])
        if num_items <= 0:
            raise RuntimeError("item_feature_values is empty")

        flat_ids = []
        domain_shapes: dict[str, torch.Size] = {}
        domain_numel: dict[str, int] = {}
        domain_masks: dict[str, torch.Tensor] = {}

        for domain in self.seq_domains:
            video_ids = batch[f"{domain}_video_id"].long()
            padding_mask = batch[f"{domain}_padding_mask"].bool()
            safe_ids = video_ids.clamp(min=0, max=num_items - 1)
            domain_shapes[domain] = video_ids.shape
            domain_numel[domain] = int(video_ids.numel())
            domain_masks[domain] = padding_mask
            flat_ids.append(safe_ids.reshape(-1))

        all_ids = torch.cat(flat_ids, dim=0)
        unique_ids, inverse = torch.unique(
            all_ids,
            sorted=False,
            return_inverse=True,
        )
        unique_item_features = self.item_feature_values[unique_ids]
        unique_concat_emb = self.item_embedding_bank.embed_flat_concat(unique_item_features)

        seq_concat_emb: dict[str, torch.Tensor] = {}
        offset = 0
        for domain in self.seq_domains:
            numel = domain_numel[domain]
            domain_inverse = inverse[offset : offset + numel].view(domain_shapes[domain])
            seq_concat_emb[domain] = unique_concat_emb[domain_inverse]
            offset += numel

        return seq_concat_emb, domain_masks

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_tokens = self.user_tokenizer(batch["user_feat"])
        item_tokens = self.item_tokenizer(batch["item_feat"])
        nonseq_tokens = torch.cat([user_tokens, item_tokens], dim=1)
        seq_query_tokens = nonseq_tokens.clone()

        seq_concat_emb, seq_masks = self._build_sequence_concat_embeddings(batch)
        seq_tokens = {}
        for domain, tokenizer in self.seq_tokenizers.items():
            padding_mask = seq_masks[domain]
            seq_tokens[domain] = tokenizer.project_concat_emb(
                seq_concat_emb[domain],
                batch[f"{domain}_time_bucket"],
                padding_mask,
            )

        for layer in self.fim_layers:
            seq_query_tokens, nonseq_tokens = layer(
                seq_query_tokens,
                nonseq_tokens,
                seq_tokens,
                seq_masks,
            )

        context = torch.cat([seq_query_tokens, nonseq_tokens], dim=1)
        context = self.final_norm(context)
        task_tokens = self.task_token_initializer(context)
        for layer in self.tim_layers:
            task_tokens = layer(task_tokens, context)
        logits = [
            head(task_tokens[:, idx, :])
            for idx, head in enumerate(self.task_heads)
        ]
        return torch.cat(logits, dim=-1)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(batch)

    def get_sparse_params(self) -> list[nn.Parameter]:
        sparse_ids = set()
        sparse_params = []
        for module in self.modules():
            if isinstance(module, nn.Embedding) and getattr(module, "sparse", False):
                param_id = id(module.weight)
                if param_id not in sparse_ids:
                    sparse_ids.add(param_id)
                    sparse_params.append(module.weight)
        return sparse_params

    def get_dense_params(self) -> list[nn.Parameter]:
        sparse_ids = {id(p) for p in self.get_sparse_params()}
        return [p for p in self.parameters() if id(p) not in sparse_ids]

    def get_router_params(self) -> list[nn.Parameter]:
        return [
            layer.per_token_ffn.router
            for layer in self.fim_layers
            if isinstance(layer.per_token_ffn, PerTokenSparseMoE)
        ]

    def get_base_dense_params(self) -> list[nn.Parameter]:
        sparse_ids = {id(param) for param in self.get_sparse_params()}
        router_ids = {id(param) for param in self.get_router_params()}
        return [
            param
            for param in self.parameters()
            if id(param) not in sparse_ids and id(param) not in router_ids
        ]

    @property
    def is_sparse_moe(self) -> bool:
        return bool(self.fim_layers) and all(
            isinstance(layer.per_token_ffn, PerTokenSparseMoE)
            for layer in self.fim_layers
        )

    def convert_fim_ffns_to_sparse_moe(
        self,
        source_channel_indices: list[torch.Tensor],
        dense_optimizer: torch.optim.Optimizer | None,
        num_routed_experts: int = 5,
        router_init_std: float = 0.01,
    ) -> None:
        if self.is_sparse_moe:
            raise RuntimeError("FIM per-token FFNs are already sparse MoE modules")
        if len(source_channel_indices) != len(self.fim_layers):
            raise ValueError(
                "Partition count must equal FIM layer count: "
                f"{len(source_channel_indices)} vs {len(self.fim_layers)}"
            )
        old_parameter_ids = {
            id(param) for param in self.get_base_dense_params()
        }
        for layer, partition in zip(self.fim_layers, source_channel_indices):
            dense_ffn = layer.per_token_ffn
            if not isinstance(dense_ffn, PerTokenSwiGLU):
                raise TypeError(
                    "Dense-to-MoE conversion requires PerTokenSwiGLU sources"
                )
            layer.per_token_ffn = PerTokenSparseMoE.from_dense(
                dense_ffn,
                source_channel_indices=partition,
                num_routed_experts=int(num_routed_experts),
                optimizer=dense_optimizer,
                router_init_std=float(router_init_std),
            )
        self.moe_num_routed_experts = int(num_routed_experts)
        self.moe_router_init_std = float(router_init_std)
        new_parameter_ids = {
            id(param) for param in self.get_base_dense_params()
        }
        if new_parameter_ids != old_parameter_ids:
            raise RuntimeError(
                "Dense-to-MoE conversion replaced a pre-existing dense Parameter; "
                "lossless AdamW continuation is impossible"
            )

    def moe_auxiliary_loss(self) -> torch.Tensor | None:
        losses = []
        for layer in self.fim_layers:
            ffn = layer.per_token_ffn
            if isinstance(ffn, PerTokenSparseMoE):
                value = ffn.auxiliary_loss()
                if value is not None:
                    losses.append(value)
        return torch.stack(losses).mean() if losses else None

    def moe_router_metrics(self) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for layer in self.fim_layers:
            ffn = layer.per_token_ffn
            if not isinstance(ffn, PerTokenSparseMoE):
                continue
            for name, value in ffn.router_metrics().items():
                values.setdefault(name, []).append(float(value))
        return {
            name: sum(items) / len(items)
            for name, items in values.items()
            if items
        }

    def parameter_counts(self) -> dict[str, int]:
        """Return disjoint dense/sparse counts plus the growth-target FIM block."""

        sparse = sum(param.numel() for param in self.get_sparse_params())
        dense = sum(param.numel() for param in self.get_dense_params())
        fim_per_token = sum(
            param.numel()
            for layer in self.fim_layers
            for param in layer.per_token_ffn.parameters()
        )
        router = sum(param.numel() for param in self.get_router_params())
        return {
            "dense": dense,
            "sparse": sparse,
            "total": dense + sparse,
            "fim_per_token": fim_per_token,
            "moe_router": router,
        }

    def reinit_sparse_params(self) -> set[int]:
        """Reinitialize every sparse embedding while leaving dense weights untouched."""

        reinit_ptrs: set[int] = set()
        for module in self.modules():
            if not (
                isinstance(module, nn.Embedding)
                and getattr(module, "sparse", False)
            ):
                continue
            ptr = module.weight.data_ptr()
            if ptr in reinit_ptrs:
                continue
            nn.init.xavier_normal_(module.weight.data)
            padding_idx = getattr(
                module,
                "_kuairand_padding_idx",
                module.padding_idx,
            )
            if padding_idx is not None:
                module.weight.data[int(padding_idx)].zero_()
            reinit_ptrs.add(ptr)
        LOGGER.info("Re-initialized %d sparse embedding tables", len(reinit_ptrs))
        return reinit_ptrs

    def reinit_high_cardinality_params(self, cardinality_threshold: int) -> set[int]:
        reinit_ptrs: set[int] = set()
        seen: set[int] = set()
        for module in self.modules():
            if not isinstance(module, nn.Embedding):
                continue
            ptr = module.weight.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            num_embeddings = int(getattr(module, "_kuairand_num_embeddings", module.num_embeddings))
            if num_embeddings <= int(cardinality_threshold):
                continue
            nn.init.xavier_normal_(module.weight.data)
            padding_idx = getattr(module, "_kuairand_padding_idx", module.padding_idx)
            if padding_idx is not None:
                module.weight.data[int(padding_idx)].zero_()
            reinit_ptrs.add(ptr)
        LOGGER.info(
            "Re-initialized %d high-cardinality embedding tables with threshold=%d",
            len(reinit_ptrs),
            int(cardinality_threshold),
        )
        return reinit_ptrs 
