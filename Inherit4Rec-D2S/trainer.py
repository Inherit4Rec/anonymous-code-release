#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-machine trainer for KuaiRand-1K multi-target prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset


LOGGER = logging.getLogger("kuairand")


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def binary_logloss_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = logits.astype(np.float64, copy=False)
    labels = labels.astype(np.float64, copy=False)
    loss = np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    return float(loss.mean())


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    n = labels.shape[0]
    pos = int(labels.sum())
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            ranks[i:j] = 0.5 * (ranks[i] + ranks[j - 1])
        i = j

    rank_sum_pos = ranks[sorted_labels == 1].sum()
    return float((rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute threshold-based average precision without sklearn.

    Samples with equal scores are evaluated as one threshold, so the result is
    deterministic and does not depend on the original order of tied samples.
    """

    labels = labels.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels, dtype=np.int64)

    threshold_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        sorted_scores.shape[0] - 1,
    ]
    threshold_tp = true_positives[threshold_ends]
    predicted_positives = threshold_ends + 1
    precision = threshold_tp / predicted_positives
    recall = threshold_tp / pos
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def grouped_auc(labels: np.ndarray, scores: np.ndarray, group_ids: np.ndarray) -> float:
    order = np.argsort(group_ids, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    group_ids = group_ids[order]

    weighted_auc = 0.0
    total_weight = 0
    start = 0
    n = group_ids.shape[0]
    while start < n:
        end = start + 1
        while end < n and group_ids[end] == group_ids[start]:
            end += 1
        group_labels = labels[start:end]
        if group_labels.min() != group_labels.max():
            weight = end - start
            weighted_auc += binary_auc(group_labels, scores[start:end]) * weight
            total_weight += weight
        start = end

    if total_weight == 0:
        return float("nan")
    return float(weighted_auc / total_weight)


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    task_names: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"tasks": {}}
    loglosses = []
    aucs = []
    auprcs = []
    gaucs = []
    probs = sigmoid_np(logits)

    for idx, task in enumerate(task_names):
        y = labels[:, idx]
        task_logits = logits[:, idx]
        task_probs = probs[:, idx]
        logloss = binary_logloss_from_logits(task_logits, y)
        auc = binary_auc(y, task_probs)
        auprc = binary_auprc(y, task_probs)
        gauc = grouped_auc(y, task_probs, user_ids)
        metrics["tasks"][task] = {
            "logloss": logloss,
            "auc": auc,
            "auprc": auprc,
            "gauc": gauc,
        }
        loglosses.append(logloss)
        aucs.append(auc)
        auprcs.append(auprc)
        gaucs.append(gauc)

    metrics["logloss"] = float(np.nanmean(np.asarray(loglosses, dtype=np.float64)))
    metrics["auc"] = float(np.nanmean(np.asarray(aucs, dtype=np.float64)))
    metrics["auprc"] = float(np.nanmean(np.asarray(auprcs, dtype=np.float64)))
    metrics["gauc"] = float(np.nanmean(np.asarray(gaucs, dtype=np.float64)))
    return metrics


def build_dataloaders(bundle: Any, *, batch_size: int, eval_batch_size: int,
                      grad_accum_steps: int, num_workers: int,
                      prefetch_factor: int, pin_memory: bool) -> dict[str, DataLoader]:
    """Give every epoch the same ordered, optimizer-step-aligned examples."""
    effective_batch = batch_size * grad_accum_steps
    if effective_batch <= 0 or eval_batch_size <= 0:
        raise ValueError("Batch sizes and accumulation must be positive")
    used = len(bundle.train_dataset) // effective_batch * effective_batch
    if used == 0:
        raise ValueError("Training data contains no complete effective batch")
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers:
        kwargs["prefetch_factor"] = prefetch_factor
    LOGGER.info("Train samples: raw=%d used=%d dropped=%d", len(bundle.train_dataset),
                used, len(bundle.train_dataset) - used)
    return {
        "train": DataLoader(Subset(bundle.train_dataset, range(used)),
                            batch_size=batch_size, shuffle=False, drop_last=True, **kwargs),
        "valid": DataLoader(bundle.valid_dataset, batch_size=eval_batch_size,
                            shuffle=False, **kwargs),
        "test": DataLoader(bundle.test_dataset, batch_size=eval_batch_size,
                           shuffle=False, **kwargs),
    }


def train_epoch(model: nn.Module, loader: DataLoader, *, device: torch.device,
                dense_optimizer: torch.optim.Optimizer,
                sparse_optimizer: torch.optim.Optimizer | None,
                extra_optimizer: torch.optim.Optimizer | None = None,
                extra_before_step: Any | None = None,
                grad_accum_steps: int = 1, grad_clip: float = 1.0,
                use_bf16: bool = True, moe_aux_coeff: float = 0.0) -> dict[str, float]:
    model.train()
    optimizers = [opt for opt in (dense_optimizer, sparse_optimizer, extra_optimizer)
                  if opt is not None]
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_samples = 0
    updates = 0
    for step, raw_batch in enumerate(loader, 1):
        batch = batch_to_device(raw_batch, device)
        labels = batch["labels"].float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_bf16 and device.type == "cuda"):
            logits = model(batch)
            task_loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none").mean(dim=0).mean()
            aux_loss = model.moe_auxiliary_loss() if moe_aux_coeff else None
            loss = task_loss + moe_aux_coeff * aux_loss if aux_loss is not None else task_loss
        (loss / grad_accum_steps).backward()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        if step % grad_accum_steps == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.get_dense_params(), grad_clip)
            if extra_before_step is not None:
                extra_before_step()
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            updates += 1
    if len(loader) % grad_accum_steps:
        raise RuntimeError("DataLoader length must be divisible by grad_accum_steps")
    return {"loss": total_loss / total_samples, "samples": total_samples, "updates": updates}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device,
             task_names: list[str], use_bf16: bool = True) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    outputs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    user_ids: list[torch.Tensor] = []
    for raw_batch in loader:
        batch = batch_to_device(raw_batch, device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_bf16 and device.type == "cuda"):
            logits = model(batch)
        outputs.append(logits.float().cpu())
        labels.append(batch["labels"].cpu())
        user_ids.append(batch["user_id"].cpu())
    if was_training:
        model.train()
    if not outputs:
        raise ValueError("Evaluation data is empty")
    return compute_metrics(torch.cat(outputs).numpy(), torch.cat(labels).numpy(),
                           torch.cat(user_ids).numpy(), task_names)


def save_weights(model: nn.Module, path: str | Path) -> None:
    """Only weights are persisted; training always starts from epoch one."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), target)
