#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inherit4Rec-D2S calibration and graph partitioning (Appendix A.1)."""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


LOGGER = logging.getLogger("kuairand")
_EPS = 1e-12


@dataclass
class CoActivationStatistics:
    """Store sufficient statistics without a full [token, hidden, hidden] matrix.

    The calibration set is small (512 examples by default). Its top-K channel
    indices need only O(N * token * K) space. Exact C_ij and G_ij are constructed
    one token position at a time during partitioning.
    """

    num_tokens: int
    hidden_dim: int
    top_k: int
    contribution_cutoff: float = 0.0
    importance_energy: torch.Tensor = field(init=False)
    sample_count: int = field(default=0, init=False)
    _top_indices_by_batch: list[torch.Tensor] = field(default_factory=list, init=False)
    _output_channel_norm: torch.Tensor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.num_tokens = int(self.num_tokens)
        self.hidden_dim = int(self.hidden_dim)
        self.top_k = int(self.top_k)
        self.contribution_cutoff = float(self.contribution_cutoff)
        if self.num_tokens <= 0 or self.hidden_dim <= 0:
            raise ValueError("Calibration dimensions must be positive")
        if not 1 <= self.top_k <= self.hidden_dim:
            raise ValueError("top_k must lie in [1, hidden_dim]")
        if not math.isfinite(self.contribution_cutoff) or self.contribution_cutoff < 0:
            raise ValueError("contribution_cutoff must be finite and nonnegative")
        self.importance_energy = torch.zeros(
            self.num_tokens, self.hidden_dim, dtype=torch.float64
        )

    @torch.no_grad()
    def update(self, hidden: torch.Tensor, out_weight: torch.Tensor) -> None:
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != (
            self.num_tokens, self.hidden_dim
        ):
            raise ValueError(f"Unexpected hidden shape: {tuple(hidden.shape)}")
        if tuple(out_weight.shape[:2]) != (self.num_tokens, self.hidden_dim):
            raise ValueError(f"Unexpected output weight shape: {tuple(out_weight.shape)}")
        if hidden.shape[0] == 0:
            return
        if self._output_channel_norm is None:
            self._output_channel_norm = torch.linalg.vector_norm(
                out_weight.float(), dim=-1
            )
        # a_j(x) = |h_j(x)| ||W_down,j:||_2; I_j = mean_x a_j(x)^2.
        contribution = hidden.float().abs() * self._output_channel_norm.unsqueeze(0)
        self.importance_energy.add_(
            contribution.square().sum(dim=0).to(device="cpu", dtype=torch.float64)
        )
        top = contribution.topk(self.top_k, dim=-1, sorted=False)
        # Appendix A.1: b_j(x) = 1[j in TopK(a(x)) and a_j(x) > gamma].
        indices = top.indices.masked_fill(
            top.values <= self.contribution_cutoff, -1
        )
        self._top_indices_by_batch.append(indices.to(device="cpu", dtype=torch.int32))
        self.sample_count += int(hidden.shape[0])

    def top_indices(self) -> torch.Tensor:
        if not self._top_indices_by_batch:
            raise ValueError("Calibration has no samples")
        return torch.cat(self._top_indices_by_batch, dim=0)


def _slice_batch(batch: dict[str, Any], max_samples: int) -> dict[str, Any]:
    labels = batch.get("labels")
    if not torch.is_tensor(labels) or labels.ndim == 0:
        raise ValueError("Calibration batch must contain batched labels")
    batch_size = int(labels.shape[0])
    used = batch_size if max_samples <= 0 else min(batch_size, int(max_samples))
    return {
        name: (
            value[:used]
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size
            else value
        )
        for name, value in batch.items()
    }


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


@torch.no_grad()
def collect_coactivation_statistics(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    max_batches: int,
    samples_per_batch: int,
    top_k: int,
    contribution_cutoff: float,
    use_bf16: bool,
) -> list[CoActivationStatistics]:
    """Collect contribution energy and exact top-K indicators from Dense FFNs."""

    if max_batches <= 0 or samples_per_batch <= 0:
        raise ValueError("Calibration batches and samples per batch must be positive")
    collectors: list[CoActivationStatistics] = []
    modules = []
    for layer_idx, layer in enumerate(model.fim_layers):
        module = layer.per_token_ffn
        if not hasattr(module, "set_coactivation_collector"):
            raise TypeError(f"FIM layer {layer_idx} is not a Dense per-token SwiGLU")
        collector = CoActivationStatistics(
            num_tokens=int(module.in_ffn.shape[0]),
            hidden_dim=int(module.hidden_dim),
            top_k=top_k,
            contribution_cutoff=contribution_cutoff,
        )
        module.set_coactivation_collector(collector)
        collectors.append(collector)
        modules.append(module)

    was_training = model.training
    model.eval()
    try:
        for batch_idx, raw_batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            batch = _batch_to_device(
                _slice_batch(raw_batch, samples_per_batch), device
            )
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=bool(use_bf16 and device.type == "cuda"),
            ):
                model(batch)
            LOGGER.info(
                "MoE calibration | batch=%d/%d samples_per_layer=%d",
                batch_idx + 1,
                max_batches,
                collectors[0].sample_count,
            )
    finally:
        for module in modules:
            module.set_coactivation_collector(None)
        model.train(was_training)
    expected_samples = max_batches * samples_per_batch
    if not collectors or any(c.sample_count != expected_samples for c in collectors):
        raise RuntimeError(
            "Incomplete MoE calibration: expected "
            f"{expected_samples} samples per layer, got "
            f"{[c.sample_count for c in collectors]}"
        )
    return collectors


def _support_aware_similarity(
    token_top_indices: np.ndarray,
    routed_indices: np.ndarray,
    hidden_dim: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Appendix A.1 equations (17)-(18) exactly for one token."""

    num_channels = len(routed_indices)
    index_map = np.full(hidden_dim, -1, dtype=np.int32)
    index_map[routed_indices] = np.arange(num_channels, dtype=np.int32)
    local_top = index_map[np.maximum(token_top_indices, 0)]
    local_top[token_top_indices < 0] = -1
    valid = local_top >= 0
    support = np.bincount(local_top[valid], minlength=num_channels).astype(np.int32)

    # C_ij = sum_n b_i(x_n)b_j(x_n). Only Top-K pairs are visited. Building
    # this matrix for one token bounds peak memory independently of token count.
    rows = np.broadcast_to(local_top[:, :, None], (*local_top.shape, local_top.shape[1]))
    cols = np.broadcast_to(local_top[:, None, :], (*local_top.shape, local_top.shape[1]))
    pair_valid = valid[:, :, None] & valid[:, None, :]
    counts = np.zeros((num_channels, num_channels), dtype=np.int32)
    np.add.at(counts, (rows[pair_valid], cols[pair_valid]), 1)
    if not np.array_equal(np.diag(counts), support):
        raise RuntimeError("Pair counts and marginal supports disagree")

    count_float = counts.astype(np.float64)
    denominator = np.sqrt(
        support.astype(np.float64)[:, None]
        * support.astype(np.float64)[None, :]
    )
    similarity = np.divide(
        count_float,
        denominator,
        out=np.zeros_like(count_float),
        where=denominator > 0,
    )
    similarity *= count_float / (count_float + tau)
    # Keep G_ii for the weighted degree in Eq. (19). The partition objective
    # and swap gains below exclude self-edges as required by Eq. (14).
    return similarity, support


def _within_expert_weight(similarity: np.ndarray, experts: list[list[int]]) -> float:
    return float(
        sum(
            np.triu(similarity[np.ix_(members, members)], k=1).sum()
            for members in experts
        )
    )


def _best_swaps(
    similarity: np.ndarray, experts: list[list[int]], rounds: int
) -> int:
    """Apply one best positive-gain swap per expert pair in each round."""

    swap_count = 0
    for _ in range(rounds):
        improved = False
        for left_expert in range(len(experts)):
            for right_expert in range(left_expert + 1, len(experts)):
                left = np.asarray(experts[left_expert], dtype=np.int64)
                right = np.asarray(experts[right_expert], dtype=np.int64)
                cross = similarity[np.ix_(left, right)]
                left_inside = (
                    similarity[np.ix_(left, left)].sum(axis=1)
                    - similarity[left, left]
                )
                right_inside = (
                    similarity[np.ix_(right, right)].sum(axis=1)
                    - similarity[right, right]
                )
                # Replacing i in A by j in B changes the within-expert sum by
                # (sum_A G[j,a] - G[j,i] - sum_A G[i,a]) plus its B counterpart.
                gain = (
                    cross.sum(axis=0)[None, :]
                    - left_inside[:, None]
                    + cross.sum(axis=1)[:, None]
                    - right_inside[None, :]
                    - 2.0 * cross
                )
                best_flat = int(np.argmax(gain))
                best_gain = float(gain.flat[best_flat])
                if best_gain <= 0.0:
                    continue
                left_pos, right_pos = np.unravel_index(best_flat, gain.shape)
                experts[left_expert][left_pos], experts[right_expert][right_pos] = (
                    experts[right_expert][right_pos],
                    experts[left_expert][left_pos],
                )
                swap_count += 1
                improved = True
        if not improved:
            break
    return swap_count


def _greedy_graph_partition(
    similarity: np.ndarray,
    num_experts: int,
    expert_width: int,
    capacity_bonus: float,
    swap_rounds: int,
    seed: int,
) -> tuple[list[list[int]], float, int]:
    """Appendix A.1 weighted-degree seeding, assignment, and swap search."""

    num_channels = similarity.shape[0]
    if similarity.shape != (num_channels, num_channels):
        raise ValueError("Similarity must be square")
    if num_channels != num_experts * expert_width:
        raise ValueError("Expert capacity does not cover all routed channels")
    generator = np.random.default_rng(seed)
    priority = np.empty(num_channels, dtype=np.int64)
    priority[generator.permutation(num_channels)] = np.arange(num_channels)
    degree = similarity.sum(axis=1)
    ordered = np.lexsort((priority, -degree))
    seeds = [int(ordered[0])]
    candidate_bound = max(256, 64 * num_experts)
    while len(seeds) < num_experts:
        remaining = ordered[~np.isin(ordered, seeds, assume_unique=True)]
        candidates = remaining[: min(len(remaining), candidate_bound)]
        d_max = float(degree[candidates].max())
        representativeness = (
            degree[candidates] / d_max if d_max > 0 else np.zeros(len(candidates))
        )
        selected = np.asarray(seeds, dtype=np.int64)
        denominator = np.sqrt(
            (degree[candidates, None] + _EPS) * (degree[selected][None, :] + _EPS)
        )
        redundancy = np.max(
            similarity[np.ix_(candidates, selected)] / denominator, axis=1
        )
        score = representativeness - redundancy
        choice = np.lexsort((priority[candidates], -score))[0]
        seeds.append(int(candidates[choice]))

    experts = [[seed] for seed in seeds]
    seed_set = set(seeds)
    for channel in ordered:
        channel = int(channel)
        if channel in seed_set:
            continue
        scores = [
            (
                float(similarity[channel, members].sum()) / math.sqrt(len(members))
                + capacity_bonus * (expert_width - len(members))
            )
            if len(members) < expert_width
            else -float("inf")
            for members in experts
        ]
        experts[int(np.argmax(scores))].append(channel)
    if any(len(members) != expert_width for members in experts):
        raise RuntimeError("Greedy partition violated expert capacity")
    swaps = _best_swaps(similarity, experts, swap_rounds)
    return experts, _within_expert_weight(similarity, experts), swaps


def build_coactivation_partitions(
    collectors: list[CoActivationStatistics],
    num_routed_experts: int,
    num_starts: int,
    refine_rounds: int,
    seed: int,
    coactivation_tau: float = 10.0,
    capacity_bonus: float = 1e-7,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Select top-contribution shared channels and graph-partition the rest."""

    num_routed_experts = int(num_routed_experts)
    num_starts = int(num_starts)
    refine_rounds = int(refine_rounds)
    coactivation_tau = float(coactivation_tau)
    capacity_bonus = float(capacity_bonus)
    if num_routed_experts <= 0 or num_starts <= 0 or refine_rounds < 0:
        raise ValueError("Invalid expert count, candidate count, or swap rounds")
    if not math.isfinite(coactivation_tau) or coactivation_tau <= 0:
        raise ValueError("coactivation_tau must be finite and positive")
    if not math.isfinite(capacity_bonus) or capacity_bonus < 0:
        raise ValueError("capacity_bonus must be finite and nonnegative")
    total_experts = num_routed_experts + 1
    partitions = []
    layer_reports = []
    digest = hashlib.sha256()

    for layer_idx, collector in enumerate(collectors):
        if collector.hidden_dim % total_experts != 0:
            raise ValueError("Dense hidden width must be divisible by total experts")
        expert_width = collector.hidden_dim // total_experts
        top_indices = collector.top_indices().numpy()
        channel_ids = np.arange(collector.hidden_dim, dtype=np.int64)
        token_partitions = []
        token_ratios = []
        token_swaps = []
        inactive_channels = 0
        for token_idx in range(collector.num_tokens):
            importance = (
                collector.importance_energy[token_idx].numpy() / collector.sample_count
            )
            # Stable order resolves equal contribution energies by channel index.
            importance_order = np.argsort(-importance, kind="stable")
            shared = np.sort(importance_order[:expert_width])
            routed = channel_ids[~np.isin(channel_ids, shared, assume_unique=True)]
            similarity, support = _support_aware_similarity(
                top_indices[:, token_idx, :],
                routed,
                collector.hidden_dim,
                coactivation_tau,
            )
            inactive_channels += int(np.count_nonzero(support == 0))
            total_weight = float(np.triu(similarity, k=1).sum())
            best_experts = None
            best_weight = -float("inf")
            best_swaps = 0
            for candidate_idx in range(num_starts):
                experts, weight, swaps = _greedy_graph_partition(
                    similarity,
                    num_experts=num_routed_experts,
                    expert_width=expert_width,
                    capacity_bonus=capacity_bonus,
                    swap_rounds=refine_rounds,
                    seed=int(seed) + layer_idx * 1_000_033 + token_idx * 10_007
                    + candidate_idx * 1_000_003,
                )
                # Equation (22) has a constant denominator across candidates.
                if weight > best_weight:
                    best_experts, best_weight, best_swaps = experts, weight, swaps
            if best_experts is None:
                raise RuntimeError("No graph partition candidate was produced")
            routed_buckets = [np.sort(routed[members]) for members in best_experts]
            routed_buckets.sort(key=lambda values: int(values[0]))
            partition = np.stack([shared, *routed_buckets])
            if not np.array_equal(np.sort(partition.reshape(-1)), channel_ids):
                raise RuntimeError(
                    f"Layer {layer_idx} token {token_idx} partition is not exact"
                )
            token_partitions.append(torch.from_numpy(partition.copy()))
            token_ratios.append(best_weight / total_weight if total_weight > 0 else 0.0)
            token_swaps.append(best_swaps)

        layer_partition = torch.stack(token_partitions)
        digest.update(layer_partition.numpy().tobytes())
        partitions.append(layer_partition)
        layer_reports.append({
            "layer": layer_idx,
            "tokens": collector.num_tokens,
            "hidden_dim": collector.hidden_dim,
            "expert_width": expert_width,
            "samples": collector.sample_count,
            "mean_within_expert_edge_ratio": float(np.mean(token_ratios)),
            "total_positive_swaps": int(sum(token_swaps)),
            "inactive_routed_channels": inactive_channels,
        })

    report = {
        "method": "paper_support_aware_graph_partition_v1",
        "coactivation_tau": coactivation_tau,
        "capacity_bonus": capacity_bonus,
        "candidate_partitions": num_starts,
        "max_swap_rounds": refine_rounds,
        "num_routed_experts": num_routed_experts,
        "num_total_experts": total_experts,
        "partition_fingerprint": digest.hexdigest()[:16],
        "layers": layer_reports,
    }
    LOGGER.info("Dense-to-MoE partition complete | report=%s", report)
    return partitions, report


__all__ = [
    "CoActivationStatistics",
    "build_coactivation_partitions",
    "collect_coactivation_statistics",
]
