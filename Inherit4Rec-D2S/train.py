#!/usr/bin/env python3
"""Train five Dense epochs, convert to Sparse MoE, then train five more epochs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from dataset import DEFAULT_SEQUENCE_CONFIG, TARGET_COLUMNS, build_dataset_bundle
from model import UniFormer
from moe import build_coactivation_partitions, collect_coactivation_statistics
from trainer import build_dataloaders, evaluate, save_weights, train_epoch
from utils import build_logger, set_seed


TOTAL_EPOCHS = 10
CONVERSION_AFTER_EPOCH = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--valid_data", type=int, default=2)
    parser.add_argument("--test_data", type=int, default=2)
    parser.add_argument("--seq_config", default=DEFAULT_SEQUENCE_CONFIG)
    parser.add_argument("--static_chunksize", type=int, default=500_000)
    parser.add_argument("--max_rows", type=int)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--d_model", type=int, default=400)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--user_tokens", type=int, default=25)
    parser.add_argument("--item_tokens", type=int, default=8)
    parser.add_argument("--num_fim_layers", type=int, default=2)
    parser.add_argument("--num_tim_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--s_hidden_multi", type=int, default=2)
    parser.add_argument("--ns_hidden_dim", type=int, default=1800)
    parser.add_argument("--t_hidden_multi", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--ns_shuffle_seed", type=int, default=42)
    parser.add_argument("--moe_num_routed_experts", type=int, default=5)
    parser.add_argument("--moe_aux_loss_coeff", type=float, default=0.01)
    parser.add_argument("--moe_router_init_std", type=float, default=0.01)
    parser.add_argument("--moe_calibration_batches", type=int, default=8)
    parser.add_argument("--moe_calibration_samples_per_batch", type=int, default=64)
    parser.add_argument("--moe_coactivation_top_k", type=int, default=32)
    parser.add_argument("--moe_contribution_cutoff", type=float, default=0.0)
    parser.add_argument("--moe_coactivation_tau", type=float, default=10.0)
    parser.add_argument("--moe_partition_capacity_bonus", type=float, default=1e-7)
    parser.add_argument("--moe_partition_starts", type=int, default=4)
    parser.add_argument("--moe_partition_refine_rounds", type=int, default=2)
    parser.add_argument("--moe_partition_seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sparse_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--bf16", dest="use_bf16", action="store_true")
    parser.add_argument("--no_bf16", dest="use_bf16", action="store_false")
    parser.add_argument("--reinit_sparse_every_epoch", dest="reset_sparse", action="store_true")
    parser.add_argument("--no_reinit_sparse_every_epoch", dest="reset_sparse", action="store_false")
    parser.set_defaults(use_bf16=True, reset_sparse=True)
    parser.add_argument("--ckpt_dir", default="checkpoints/dense_to_moe_03b")
    parser.add_argument("--log_file", default="logs/dense_to_moe_03b.log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ns_hidden_dim % (args.moe_num_routed_experts + 1):
        raise ValueError("FFN width must divide evenly across the shared and routed experts")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    set_seed(args.seed)
    logger = build_logger(log_file=args.log_file)
    bundle = build_dataset_bundle(
        data_dir=args.data_dir, train_start=args.train_start,
        valid_days=args.valid_data, test_days=args.test_data,
        static_chunksize=args.static_chunksize, max_rows=args.max_rows,
        sequence_config=args.seq_config,
    )
    loaders = build_dataloaders(
        bundle, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=device.type == "cuda" and not args.disable_pin_memory,
    )
    model = UniFormer(
        feature_schema=bundle.feature_schema(), task_names=TARGET_COLUMNS,
        seq_domains=list(bundle.sequence_lengths.keys()), d_model=args.d_model,
        emb_dim=args.emb_dim, user_tokens=args.user_tokens, item_tokens=args.item_tokens,
        num_fim_layers=args.num_fim_layers, num_tim_layers=args.num_tim_layers,
        num_heads=args.num_heads, s_hidden_multi=args.s_hidden_multi,
        ns_hidden_dim=args.ns_hidden_dim, t_hidden_multi=args.t_hidden_multi,
        dropout=args.dropout, ns_shuffle_seed=args.ns_shuffle_seed,
        item_feature_values=bundle.item_table.values,
    ).to(device)
    dense_optimizer = torch.optim.AdamW(model.get_dense_params(), lr=args.lr,
                                         weight_decay=args.weight_decay)
    sparse_params = model.get_sparse_params()
    sparse_optimizer = (torch.optim.SparseAdam(sparse_params, lr=args.sparse_lr)
                        if sparse_params else None)
    router_optimizer: torch.optim.AdamW | None = None
    output = Path(args.ckpt_dir)
    output.mkdir(parents=True, exist_ok=True)
    logger.info("D2S start | device=%s split=%s dense_params=%s",
                device, bundle.split_summary.as_dict(), model.parameter_counts()["dense"])
    history: list[dict] = []
    best_epoch: int | None = None
    best_gauc = -float("inf")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        phase = "dense" if epoch <= CONVERSION_AFTER_EPOCH else "moe"
        train_metrics = train_epoch(
            model, loaders["train"], device=device, dense_optimizer=dense_optimizer,
            sparse_optimizer=sparse_optimizer, extra_optimizer=router_optimizer,
            grad_accum_steps=args.grad_accum_steps, grad_clip=args.grad_clip,
            use_bf16=args.use_bf16,
            moe_aux_coeff=args.moe_aux_loss_coeff if router_optimizer is not None else 0.0,
        )
        valid = evaluate(model, loaders["valid"], device=device,
                         task_names=TARGET_COLUMNS, use_bf16=args.use_bf16)
        logger.info("Epoch %d/%d %s | train_loss=%.6f valid_GAUC=%.6f",
                    epoch, TOTAL_EPOCHS, phase, train_metrics["loss"], valid["gauc"])
        history.append({"epoch": epoch, "phase": phase,
                        "train": train_metrics, "valid": valid})
        save_weights(model, output / "last.pt")
        if epoch == CONVERSION_AFTER_EPOCH:
            save_weights(model, output / "dense_source.pt")
            dense_optimizer.zero_grad(set_to_none=True)
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad(set_to_none=True)
            # Calibration should not move the random stream used to initialize
            # the router or the next epoch's sparse embeddings.
            py_rng, np_rng = random.getstate(), np.random.get_state()
            torch_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            try:
                collectors = collect_coactivation_statistics(
                    model, loaders["train"], device,
                    max_batches=args.moe_calibration_batches,
                    samples_per_batch=args.moe_calibration_samples_per_batch,
                    top_k=args.moe_coactivation_top_k,
                    contribution_cutoff=args.moe_contribution_cutoff,
                    use_bf16=args.use_bf16,
                )
                partitions, _ = build_coactivation_partitions(
                    collectors, num_routed_experts=args.moe_num_routed_experts,
                    num_starts=args.moe_partition_starts,
                    refine_rounds=args.moe_partition_refine_rounds,
                    seed=args.moe_partition_seed,
                    coactivation_tau=args.moe_coactivation_tau,
                    capacity_bonus=args.moe_partition_capacity_bonus,
                )
            finally:
                random.setstate(py_rng)
                np.random.set_state(np_rng)
                torch.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
            model.convert_fim_ffns_to_sparse_moe(
                partitions, dense_optimizer=dense_optimizer,
                num_routed_experts=args.moe_num_routed_experts,
                router_init_std=args.moe_router_init_std,
            )
            router_optimizer = torch.optim.AdamW(model.get_router_params(), lr=args.lr,
                                                  weight_decay=args.weight_decay)
            logger.info("D2S conversion complete | dense_params=%s",
                        model.parameter_counts()["dense"])
        elif epoch > CONVERSION_AFTER_EPOCH and (
            best_epoch is None or (math.isfinite(valid["gauc"]) and valid["gauc"] > best_gauc)
        ):
            best_epoch, best_gauc = epoch, valid["gauc"]
            save_weights(model, output / "best.pt")
        if epoch < TOTAL_EPOCHS and args.reset_sparse and sparse_params:
            model.reinit_sparse_params()
            sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=args.sparse_lr)

    test = evaluate(model, loaders["test"], device=device,
                    task_names=TARGET_COLUMNS, use_bf16=args.use_bf16)
    summary = {"method": "D2S", "dense_epochs": 5, "moe_epochs": 5,
               "best_epoch": best_epoch, "history": history, "test": test}
    (output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("D2S complete | final_test_GAUC=%.6f", test["gauc"])


if __name__ == "__main__":
    main()
