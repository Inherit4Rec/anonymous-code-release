#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-place Dense-to-Dense widening for the FIM per-token SwiGLU.

The source ``PerTokenSwiGLU`` stores its two input projections in one tensor::

    in_ffn = concat([value (W3), gate (W1)], dim=-1)

This module keeps the source ``in_ffn`` and ``out_ffn`` Parameters unchanged
(including their Python identities) and registers only the newly grown groups.
Keeping the old Parameters intact lets a caller retain the old AdamW state and
attach a fresh optimizer to :func:`get_growth_parameters`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_GROWTH_SEED = 20_260_716
_MONITOR_EPS = 1e-8
_POST_CAST_REL_TOLERANCE = {
    torch.float64: 1e-10,
    torch.float32: 1e-5,
    torch.float16: 5e-3,
    torch.bfloat16: 1e-2,
}
_WEIGHT_SEED_OFFSET = {"w2": 2}


def _shape_list(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _weight_seed(base_seed: int, layer_idx: int, token_idx: int, suffix: str) -> int:
    """Match the per-layer/per-token seed schedule in the reference code."""

    if suffix not in _WEIGHT_SEED_OFFSET:
        raise ValueError(f"Unsupported SwiGLU weight suffix: {suffix!r}")
    if int(layer_idx) < 0 or int(token_idx) < 0:
        raise ValueError("layer_idx and token_idx must be non-negative")
    derived_seed = (
        int(base_seed)
        + int(layer_idx) * 100_000
        + int(token_idx) * 100
        + _WEIGHT_SEED_OFFSET[suffix]
    )
    if not 0 <= derived_seed <= np.iinfo(np.uint32).max:
        raise ValueError(f"Derived NumPy RandomState seed is out of range: {derived_seed}")
    return derived_seed


def _as_numpy_float64(tensor: torch.Tensor) -> np.ndarray:
    """Copy a tensor to a finite, writable float64 NumPy array."""

    value = tensor.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
    if not np.isfinite(value).all():
        raise ValueError("Source per-token FFN contains NaN or Inf")
    return value


def _exact_copies(
    source: np.ndarray,
    copy_count: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Repeat one W1/W3 token matrix without perturbation."""

    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D token weight, got shape={source.shape}")
    if copy_count <= 0:
        raise ValueError("copy_count must be positive")

    source_std = float(np.std(source))
    copies = np.repeat(source[None, :, :], copy_count, axis=0)
    if not np.isfinite(copies).all():
        raise RuntimeError("Generated input-projection growth weights are non-finite")
    copy_max_abs_error = float(np.max(np.abs(copies - source[None, :, :])))
    if copy_max_abs_error != 0.0:
        raise RuntimeError("W1/W3 growth copy is not bitwise exact in float64")
    return copies, {
        "source_mean": float(np.mean(source)),
        "source_std": source_std,
        "source_variance": float(np.var(source)),
        "copy_max_abs_error": copy_max_abs_error,
    }


def _fresh_centered_output_groups_from_source(
    source: np.ndarray,
    copy_count: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Sample W2 from the old W2 statistics, then center across groups.

    For each layer/token, the raw new groups are drawn from
    ``Normal(mean(W2_old), std(W2_old))``.  Subtracting their elementwise mean
    enforces an exact zero-sum growth path before the Parameter dtype cast.
    """

    source = np.asarray(source, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D output weight, got shape={source.shape}")
    if copy_count <= 0:
        raise ValueError("copy_count must be positive")
    if not np.isfinite(source).all():
        raise ValueError("Source W2 contains NaN or Inf")

    source_mean = float(np.mean(source))
    source_variance = float(np.var(source))
    source_std = math.sqrt(max(source_variance, 0.0))
    rng = np.random.RandomState(int(random_seed))
    raw_groups = rng.normal(
        loc=source_mean,
        scale=source_std,
        size=(copy_count,) + source.shape,
    ).astype(np.float64, copy=False)
    raw_sample_mean = float(np.mean(raw_groups))
    raw_sample_variance = float(np.var(raw_groups))
    if copy_count == 2:
        # The 600 -> 1800 experiment adds exactly two copied branches.  Build
        # them as A/-A so cancellation also remains exact after a float32 or
        # bfloat16 cast (negation is exact for finite IEEE floating values).
        paired = 0.5 * (raw_groups[0] - raw_groups[1])
        groups = np.stack([paired, -paired], axis=0)
    else:
        groups = raw_groups - np.mean(raw_groups, axis=0, keepdims=True)

    centered_mean = float(np.mean(groups))
    centered_variance = float(np.var(groups))
    expected_centered_variance = source_variance * (copy_count - 1) / copy_count
    sum_rms = float(np.sqrt(np.mean(np.square(np.sum(groups, axis=0)))))
    if not np.isfinite(groups).all():
        raise RuntimeError("Generated output-projection growth weights are non-finite")
    if sum_rms > 1e-12:
        raise RuntimeError(
            "Output-projection growth groups are not zero-sum: "
            f"RMS={sum_rms:.3e}"
        )
    return groups, {
        "source_mean": source_mean,
        "source_std": source_std,
        "source_variance": source_variance,
        "raw_sample_mean": raw_sample_mean,
        "raw_sample_variance": raw_sample_variance,
        "centered_mean": centered_mean,
        "centered_variance": centered_variance,
        "expected_centered_variance": expected_centered_variance,
        "group_sum_rms": sum_rms,
    }


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


class GrowingPerTokenSwiGLU(nn.Module):
    """A widened per-token SwiGLU with separate old and growth branches.

    ``in_ffn`` and ``out_ffn`` are the exact Parameter objects owned by the
    source module. ``growth_in_ffn`` and ``growth_out_ffn`` contain only the
    newly added hidden groups, so they can be optimized with fresh state and a
    growth-specific learning rate.
    """

    def __init__(
        self,
        source_module: nn.Module,
        target_hidden_dim: int,
        *,
        layer_idx: int,
        seed: int = DEFAULT_GROWTH_SEED,
    ) -> None:
        super().__init__()
        if isinstance(source_module, GrowingPerTokenSwiGLU):
            raise ValueError("The source per-token FFN has already been grown")
        if not hasattr(source_module, "in_ffn") or not hasattr(source_module, "out_ffn"):
            raise TypeError("Source module must expose in_ffn and out_ffn Parameters")
        if not hasattr(source_module, "dropout"):
            raise TypeError("Source module must expose its dropout module")

        source_in = source_module.in_ffn
        source_out = source_module.out_ffn
        if not isinstance(source_in, nn.Parameter) or not isinstance(source_out, nn.Parameter):
            raise TypeError("Source in_ffn and out_ffn must be nn.Parameter instances")
        if source_in.ndim != 3 or source_out.ndim != 3:
            raise ValueError(
                "Per-token FFN parameters must be rank 3: "
                f"in={tuple(source_in.shape)} out={tuple(source_out.shape)}"
            )
        if source_in.shape[-1] % 2 != 0:
            raise ValueError(f"in_ffn final dimension must be even, got {source_in.shape[-1]}")
        if not source_in.dtype.is_floating_point or not source_out.dtype.is_floating_point:
            raise TypeError("Per-token FFN parameters must use floating-point dtypes")
        if source_in.device != source_out.device or source_in.dtype != source_out.dtype:
            raise ValueError("Source in_ffn/out_ffn must share device and dtype")

        num_tokens = int(source_in.shape[0])
        d_model = int(source_in.shape[1])
        source_hidden_dim = int(source_in.shape[2] // 2)
        if num_tokens <= 0 or d_model <= 0 or source_hidden_dim <= 0:
            raise ValueError(
                "Per-token FFN dimensions must be positive: "
                f"tokens={num_tokens}, d_model={d_model}, hidden={source_hidden_dim}"
            )
        expected_out_shape = (num_tokens, source_hidden_dim, d_model)
        if tuple(source_out.shape) != expected_out_shape:
            raise ValueError(
                f"out_ffn shape mismatch: expected {expected_out_shape}, "
                f"got {tuple(source_out.shape)}"
            )

        target_hidden_dim = int(target_hidden_dim)
        if target_hidden_dim <= source_hidden_dim:
            raise ValueError(
                "target_hidden_dim must be wider than the source: "
                f"source={source_hidden_dim}, target={target_hidden_dim}"
            )
        if target_hidden_dim % source_hidden_dim != 0:
            raise ValueError(
                "target_hidden_dim must be an integer multiple of source_hidden_dim: "
                f"source={source_hidden_dim}, target={target_hidden_dim}"
            )
        width_groups = target_hidden_dim // source_hidden_dim
        copy_count = width_groups - 1
        if width_groups <= 2:
            raise ValueError(
                "Dense-to-Dense centered W2 initialization requires more than two total "
                f"width groups, got {width_groups}"
            )
        growth_hidden_dim = target_hidden_dim - source_hidden_dim

        # These assignments deliberately register the original Parameter
        # objects under their original names. Do not clone or wrap them.
        self.in_ffn = source_in
        self.out_ffn = source_out
        self.dropout = source_module.dropout
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.source_hidden_dim = source_hidden_dim
        self.growth_hidden_dim = growth_hidden_dim
        self.target_hidden_dim = target_hidden_dim
        self.hidden_dim = target_hidden_dim
        self.width_groups = width_groups
        self.copy_count = copy_count
        self.layer_idx = int(layer_idx)
        self.growth_seed = int(seed)

        growth_in = torch.empty(
            num_tokens,
            d_model,
            growth_hidden_dim * 2,
            dtype=source_in.dtype,
            device=source_in.device,
        )
        growth_out = torch.empty(
            num_tokens,
            growth_hidden_dim,
            d_model,
            dtype=source_out.dtype,
            device=source_out.device,
        )

        value_copy_max_abs_error: list[float] = []
        gate_copy_max_abs_error: list[float] = []
        out_source_mean: list[float] = []
        out_source_std: list[float] = []
        out_source_variance: list[float] = []
        out_raw_sample_mean: list[float] = []
        out_raw_sample_variance: list[float] = []
        out_centered_mean: list[float] = []
        out_centered_variance: list[float] = []
        out_expected_centered_variance: list[float] = []
        out_sum_rms: list[float] = []

        with torch.no_grad():
            for token_idx in range(num_tokens):
                source_token_in = _as_numpy_float64(source_in[token_idx])
                source_token_out = _as_numpy_float64(source_out[token_idx])
                source_value = source_token_in[:, :source_hidden_dim]  # W3
                source_gate = source_token_in[:, source_hidden_dim:]  # W1

                value_copies, value_diag = _exact_copies(source_value, copy_count)
                gate_copies, gate_diag = _exact_copies(source_gate, copy_count)
                output_groups, output_diag = _fresh_centered_output_groups_from_source(
                    source_token_out,
                    copy_count,
                    _weight_seed(self.growth_seed, self.layer_idx, token_idx, "w2"),
                )

                # Layout is all new value groups followed by all new gate groups;
                # forward().chunk(2) therefore retains the source SwiGLU meaning.
                value_flat = np.concatenate(list(value_copies), axis=1)
                gate_flat = np.concatenate(list(gate_copies), axis=1)
                growth_in_token = np.concatenate([value_flat, gate_flat], axis=1)
                growth_out_token = np.concatenate(list(output_groups), axis=0)
                if growth_in_token.shape != (d_model, growth_hidden_dim * 2):
                    raise RuntimeError(
                        "Generated growth_in_ffn token has unexpected shape: "
                        f"{growth_in_token.shape}"
                    )
                if growth_out_token.shape != (growth_hidden_dim, d_model):
                    raise RuntimeError(
                        "Generated growth_out_ffn token has unexpected shape: "
                        f"{growth_out_token.shape}"
                    )

                growth_in[token_idx].copy_(
                    torch.from_numpy(growth_in_token).to(
                        device=source_in.device,
                        dtype=source_in.dtype,
                    )
                )
                growth_out[token_idx].copy_(
                    torch.from_numpy(growth_out_token).to(
                        device=source_out.device,
                        dtype=source_out.dtype,
                    )
                )

                value_copy_max_abs_error.append(value_diag["copy_max_abs_error"])
                gate_copy_max_abs_error.append(gate_diag["copy_max_abs_error"])
                out_source_mean.append(output_diag["source_mean"])
                out_source_std.append(output_diag["source_std"])
                out_source_variance.append(output_diag["source_variance"])
                out_raw_sample_mean.append(output_diag["raw_sample_mean"])
                out_raw_sample_variance.append(output_diag["raw_sample_variance"])
                out_centered_mean.append(output_diag["centered_mean"])
                out_centered_variance.append(output_diag["centered_variance"])
                out_expected_centered_variance.append(
                    output_diag["expected_centered_variance"]
                )
                out_sum_rms.append(output_diag["group_sum_rms"])

        if not bool(torch.isfinite(growth_in).all().item()):
            raise RuntimeError("growth_in_ffn contains NaN or Inf after dtype conversion")
        if not bool(torch.isfinite(growth_out).all().item()):
            raise RuntimeError("growth_out_ffn contains NaN or Inf after dtype conversion")

        # Re-check the cancellation invariants after conversion to the actual
        # Parameter dtype.  The float64 construction diagnostics alone are not
        # enough because independently rounded groups do not generally sum to
        # bitwise zero in float32/bfloat16.
        # Work token-by-token on CPU to avoid allocating full float64 copies of
        # the 0.3B model on GPU solely for diagnostics.
        post_cast_squares = {
            "value_num": 0.0,
            "value_ref": 0.0,
            "gate_num": 0.0,
            "gate_ref": 0.0,
            "out_num": 0.0,
            "out_ref": 0.0,
        }
        for token_idx in range(num_tokens):
            source_token_in = _as_numpy_float64(source_in[token_idx])
            source_token_out = _as_numpy_float64(source_out[token_idx])
            growth_token_in = _as_numpy_float64(growth_in[token_idx])
            growth_token_out = _as_numpy_float64(growth_out[token_idx])
            source_value = source_token_in[:, :source_hidden_dim]
            source_gate = source_token_in[:, source_hidden_dim:]
            growth_value = growth_token_in[:, :growth_hidden_dim].reshape(
                d_model,
                copy_count,
                source_hidden_dim,
            )
            growth_gate = growth_token_in[:, growth_hidden_dim:].reshape(
                d_model,
                copy_count,
                source_hidden_dim,
            )
            output_groups = growth_token_out.reshape(
                copy_count,
                source_hidden_dim,
                d_model,
            )
            value_residual = growth_value - source_value[:, None, :]
            gate_residual = growth_gate - source_gate[:, None, :]
            output_residual = np.sum(output_groups, axis=0)
            post_cast_squares["value_num"] += float(np.sum(np.square(value_residual)))
            post_cast_squares["value_ref"] += float(
                copy_count * np.sum(np.square(source_value))
            )
            post_cast_squares["gate_num"] += float(np.sum(np.square(gate_residual)))
            post_cast_squares["gate_ref"] += float(
                copy_count * np.sum(np.square(source_gate))
            )
            post_cast_squares["out_num"] += float(np.sum(np.square(output_residual)))
            post_cast_squares["out_ref"] += float(np.sum(np.square(source_token_out)))

        def relative_rms_from_squares(numerator: float, reference: float) -> float:
            return math.sqrt(max(numerator, 0.0)) / max(
                math.sqrt(max(reference, 0.0)),
                _MONITOR_EPS,
            )

        post_cast_value_copy_rel = relative_rms_from_squares(
            post_cast_squares["value_num"], post_cast_squares["value_ref"]
        )
        post_cast_gate_copy_rel = relative_rms_from_squares(
            post_cast_squares["gate_num"], post_cast_squares["gate_ref"]
        )
        post_cast_out_group_sum_rel = relative_rms_from_squares(
            post_cast_squares["out_num"], post_cast_squares["out_ref"]
        )
        post_cast_tolerance = _POST_CAST_REL_TOLERANCE.get(source_in.dtype, 1e-2)
        for diagnostic_name, diagnostic_value in (
            ("value/W3 copy error", post_cast_value_copy_rel),
            ("gate/W1 copy error", post_cast_gate_copy_rel),
            ("output/W2 group sum", post_cast_out_group_sum_rel),
        ):
            if diagnostic_value > post_cast_tolerance:
                raise RuntimeError(
                    f"Post-cast {diagnostic_name} relative RMS {diagnostic_value:.3e} "
                    f"exceeds tolerance {post_cast_tolerance:.3e} for {source_in.dtype}"
                )

        self.growth_in_ffn = nn.Parameter(growth_in)
        self.growth_out_ffn = nn.Parameter(growth_out)
        self.train(source_module.training)

        if self.in_ffn is not source_in or self.out_ffn is not source_out:
            raise RuntimeError("Source Parameter identity was not preserved")

        self._initialization_report: dict[str, Any] = {
            "layer_index": self.layer_idx,
            "num_tokens": self.num_tokens,
            "d_model": self.d_model,
            "source_hidden_dim": self.source_hidden_dim,
            "target_hidden_dim": self.target_hidden_dim,
            "growth_hidden_dim": self.growth_hidden_dim,
            "width_groups": self.width_groups,
            "new_groups": self.copy_count,
            "source_in_shape": _shape_list(self.in_ffn),
            "source_out_shape": _shape_list(self.out_ffn),
            "growth_in_shape": _shape_list(self.growth_in_ffn),
            "growth_out_shape": _shape_list(self.growth_out_ffn),
            "input_growth_initialization": "exact_copy",
            "value_w3_copy_max_abs_error": _summarize(value_copy_max_abs_error),
            "gate_w1_copy_max_abs_error": _summarize(gate_copy_max_abs_error),
            "out_w2_growth_initialization": (
                "normal_from_old_stats_then_zero_sum_projection"
            ),
            "out_w2_zero_sum_method": (
                "paired_antithetic" if self.copy_count == 2 else "mean_projection"
            ),
            "out_w2_source_mean": _summarize(out_source_mean),
            "out_w2_source_std": _summarize(out_source_std),
            "out_w2_source_variance": _summarize(out_source_variance),
            "out_w2_raw_sample_mean": _summarize(out_raw_sample_mean),
            "out_w2_raw_sample_variance": _summarize(out_raw_sample_variance),
            "out_w2_centered_mean": _summarize(out_centered_mean),
            "out_w2_centered_variance": _summarize(out_centered_variance),
            "out_w2_expected_centered_variance": _summarize(
                out_expected_centered_variance
            ),
            "out_w2_group_sum_rms": _summarize(out_sum_rms),
            "post_cast_dtype": str(source_in.dtype),
            "post_cast_rel_tolerance": float(post_cast_tolerance),
            "post_cast_value_w3_copy_rel_rms": post_cast_value_copy_rel,
            "post_cast_gate_w1_copy_rel_rms": post_cast_gate_copy_rel,
            "post_cast_out_w2_group_sum_rel_rms": post_cast_out_group_sum_rel,
            "source_parameter_identity_preserved": True,
            "growth_parameters": int(
                self.growth_in_ffn.numel() + self.growth_out_ffn.numel()
            ),
        }

    @property
    def initialization_report(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the initialization diagnostics."""

        # Values are nested only one level below dictionaries of scalar values.
        return {
            key: (dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value)
            for key, value in self._initialization_report.items()
        }

    @property
    def is_grown(self) -> bool:
        return True

    def get_growth_parameters(self) -> list[nn.Parameter]:
        return [self.growth_in_ffn, self.growth_out_ffn]

    def merged_in_ffn(self) -> torch.Tensor:
        """Return the conventional ``[T, D, 2*target_H]`` fused input weight."""

        old_value, old_gate = self.in_ffn.chunk(2, dim=-1)
        growth_value, growth_gate = self.growth_in_ffn.chunk(2, dim=-1)
        merged = torch.cat(
            [old_value, growth_value, old_gate, growth_gate],
            dim=-1,
        )
        expected_shape = (self.num_tokens, self.d_model, self.target_hidden_dim * 2)
        if tuple(merged.shape) != expected_shape:
            raise RuntimeError(
                f"Merged input weight shape mismatch: expected {expected_shape}, got {tuple(merged.shape)}"
            )
        return merged

    def merged_out_ffn(self) -> torch.Tensor:
        """Return the conventional ``[T, target_H, D]`` output weight."""

        merged = torch.cat([self.out_ffn, self.growth_out_ffn], dim=1)
        expected_shape = (self.num_tokens, self.target_hidden_dim, self.d_model)
        if tuple(merged.shape) != expected_shape:
            raise RuntimeError(
                f"Merged output weight shape mismatch: expected {expected_shape}, got {tuple(merged.shape)}"
            )
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_value, old_gate = torch.einsum("btd,tdh->bth", x, self.in_ffn).chunk(
            2,
            dim=-1,
        )
        growth_value, growth_gate = torch.einsum(
            "btd,tdh->bth",
            x,
            self.growth_in_ffn,
        ).chunk(2, dim=-1)
        old_hidden = old_value * F.silu(old_gate)
        growth_hidden = growth_value * F.silu(growth_gate)

        # The source module sampled one mask for [B, T, source_H].  Reuse that
        # mask for the corresponding coordinates of every logical width group.
        # Independent masks would destroy the W2 zero-sum cancellation despite
        # exact W1/W3 copies, causing a function jump on the first train batch.
        growth_hidden = growth_hidden.reshape(
            *growth_hidden.shape[:-1],
            self.copy_count,
            self.source_hidden_dim,
        )
        if self.training and float(self.dropout.p) > 0.0:
            dropout_scale = self.dropout(torch.ones_like(old_hidden))
            old_hidden = old_hidden * dropout_scale
            growth_hidden = growth_hidden * dropout_scale.unsqueeze(-2)
        growth_hidden = growth_hidden.flatten(start_dim=-2)
        old_out = torch.einsum("bth,thd->btd", old_hidden, self.out_ffn)
        growth_out = torch.einsum(
            "bth,thd->btd",
            growth_hidden,
            self.growth_out_ffn,
        )
        return old_out + growth_out

    def extra_repr(self) -> str:
        return (
            f"tokens={self.num_tokens}, d_model={self.d_model}, "
            f"source_hidden={self.source_hidden_dim}, "
            f"growth_hidden={self.growth_hidden_dim}, "
            f"target_hidden={self.target_hidden_dim}, new_groups={self.copy_count}"
        )


def merge_growing_per_token_weights(
    module: GrowingPerTokenSwiGLU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return conventional monolithic input/output weights for a grown module."""

    if not isinstance(module, GrowingPerTokenSwiGLU):
        raise TypeError("module must be a GrowingPerTokenSwiGLU")
    return module.merged_in_ffn(), module.merged_out_ffn()


def is_grown(module: nn.Module) -> bool:
    """Return whether a pFFN, or every FIM pFFN in a UniFormer, is grown."""

    if isinstance(module, GrowingPerTokenSwiGLU):
        return True
    fim_layers = getattr(module, "fim_layers", None)
    if fim_layers is None or len(fim_layers) == 0:
        return False
    return all(
        isinstance(getattr(layer, "per_token_ffn", None), GrowingPerTokenSwiGLU)
        for layer in fim_layers
    )


def get_growth_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return each newly registered growth Parameter exactly once."""

    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for child in module.modules():
        if not isinstance(child, GrowingPerTokenSwiGLU):
            continue
        for parameter in (child.growth_in_ffn, child.growth_out_ffn):
            parameter_id = id(parameter)
            if parameter_id in seen:
                raise RuntimeError("A growth Parameter is registered by more than one module")
            seen.add(parameter_id)
            parameters.append(parameter)
    return parameters


def grow_uniformer_in_place(
    model: nn.Module,
    target_hidden_dim: int,
    seed: int = DEFAULT_GROWTH_SEED,
) -> dict[str, Any]:
    """Widen every FIM per-token SwiGLU while preserving all source Parameters.

    The operation is intentionally strict and non-idempotent: attempting to
    grow an already or partially grown model raises instead of silently
    changing optimizer coverage.
    """

    fim_layers = getattr(model, "fim_layers", None)
    if fim_layers is None or len(fim_layers) == 0:
        raise ValueError("Model has no FIM layers to grow")

    source_modules: list[nn.Module] = []
    source_parameter_ids: list[tuple[int, int]] = []
    for layer_idx, layer in enumerate(fim_layers):
        pffn = getattr(layer, "per_token_ffn", None)
        if pffn is None:
            raise ValueError(f"FIM layer {layer_idx} has no per_token_ffn")
        if isinstance(pffn, GrowingPerTokenSwiGLU):
            raise ValueError(f"FIM layer {layer_idx} has already been grown")
        if not hasattr(pffn, "in_ffn") or not hasattr(pffn, "out_ffn"):
            raise TypeError(f"FIM layer {layer_idx} pFFN has no in_ffn/out_ffn")
        source_modules.append(pffn)
        source_parameter_ids.append((id(pffn.in_ffn), id(pffn.out_ffn)))

    all_parameter_ids_before = {id(parameter) for parameter in model.parameters()}
    parameter_numel_before = sum(parameter.numel() for parameter in model.parameters())

    # Build all replacements before mutating the model. A failed construction
    # therefore cannot leave only a prefix of the FIM layers grown.
    replacements = [
        GrowingPerTokenSwiGLU(
            source_module,
            target_hidden_dim,
            layer_idx=layer_idx,
            seed=seed,
        )
        for layer_idx, source_module in enumerate(source_modules)
    ]

    source_hidden_dims = {module.source_hidden_dim for module in replacements}
    target_hidden_dims = {module.target_hidden_dim for module in replacements}
    if len(source_hidden_dims) != 1 or len(target_hidden_dims) != 1:
        raise RuntimeError(
            "All FIM layers must have identical source and target hidden dimensions"
        )

    for layer, replacement in zip(fim_layers, replacements):
        layer.per_token_ffn = replacement
    if hasattr(model, "ns_hidden_dim"):
        model.ns_hidden_dim = int(target_hidden_dim)

    for layer_idx, (layer, expected_ids) in enumerate(zip(fim_layers, source_parameter_ids)):
        replacement = layer.per_token_ffn
        actual_ids = (id(replacement.in_ffn), id(replacement.out_ffn))
        if actual_ids != expected_ids:
            raise RuntimeError(
                f"FIM layer {layer_idx} did not preserve source Parameter identities"
            )

    all_parameter_ids_after = {id(parameter) for parameter in model.parameters()}
    if not all_parameter_ids_before.issubset(all_parameter_ids_after):
        missing_count = len(all_parameter_ids_before - all_parameter_ids_after)
        raise RuntimeError(f"Growth dropped {missing_count} source Parameters")

    growth_parameters = get_growth_parameters(model)
    expected_growth_parameter_objects = 2 * len(fim_layers)
    if len(growth_parameters) != expected_growth_parameter_objects:
        raise RuntimeError(
            "Growth Parameter coverage mismatch: "
            f"expected {expected_growth_parameter_objects}, got {len(growth_parameters)}"
        )
    new_parameter_ids = all_parameter_ids_after - all_parameter_ids_before
    if new_parameter_ids != {id(parameter) for parameter in growth_parameters}:
        raise RuntimeError("New model Parameters do not exactly match the growth Parameter set")

    parameter_numel_after = sum(parameter.numel() for parameter in model.parameters())
    growth_parameter_numel = sum(parameter.numel() for parameter in growth_parameters)
    added_parameter_numel = parameter_numel_after - parameter_numel_before
    if added_parameter_numel != growth_parameter_numel:
        raise RuntimeError(
            "Parameter-count coverage mismatch: "
            f"model added {added_parameter_numel}, growth set has {growth_parameter_numel}"
        )
    if not is_grown(model):
        raise RuntimeError("Not every FIM per-token FFN was grown")

    source_hidden_dim = next(iter(source_hidden_dims))
    width_groups = int(target_hidden_dim) // int(source_hidden_dim)
    report: dict[str, Any] = {
        "status": "grown",
        "fim_layers": int(len(fim_layers)),
        "source_hidden_dim": int(source_hidden_dim),
        "target_hidden_dim": int(target_hidden_dim),
        "growth_hidden_dim": int(target_hidden_dim) - int(source_hidden_dim),
        "width_groups": width_groups,
        "new_groups": width_groups - 1,
        "input_growth_initialization": "exact_copy",
        "output_growth_initialization": (
            "normal_from_old_stats_then_zero_sum_projection"
        ),
        "output_zero_sum_method": (
            "paired_antithetic" if width_groups - 1 == 2 else "mean_projection"
        ),
        "seed": int(seed),
        "source_parameter_objects_preserved": int(len(all_parameter_ids_before)),
        "growth_parameter_objects": int(len(growth_parameters)),
        "parameters_before": int(parameter_numel_before),
        "parameters_after": int(parameter_numel_after),
        "parameters_added": int(added_parameter_numel),
        "coverage_validated": True,
        "layers": [replacement.initialization_report for replacement in replacements],
    }
    return report


__all__ = [
    "DEFAULT_GROWTH_SEED",
    "GrowingPerTokenSwiGLU",
    "get_growth_parameters",
    "grow_uniformer_in_place",
    "is_grown",
    "merge_growing_per_token_weights",
]
