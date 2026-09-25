#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset utilities for KuaiRand-1K multi-target prediction.

This module consumes the preprocessed files directly under ``data_dir``:

  * user_features_mapped.csv 
  * user_statistic_info.json
  * item_features_mapped.csv
  * item_statistic_info.json
  * standard_interactions.csv
  * user_positive_sequences.pkl

Static user/item features are preloaded into numpy arrays. Feature lengths are
read from ``feat_len`` in the statistic JSON. Features with ``feat_len > 1`` are
parsed as comma-delimited strings and padded/truncated to the configured length.

Interactions are sorted by ``time_ms`` and split by UTC+8 day.
Positive user sequences are truncated by each sample's ``time_ms`` and
left-padded to the configured max lengths.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_COLUMNS = [
    "is_click",
    "is_like",
    "is_comment",
    "long_view",
]
INTERACTION_COLUMNS = ["user_id", "video_id", "time_ms", *TARGET_COLUMNS]
DEFAULT_SEQUENCE_CONFIG = "seq_a:512,seq_b:512,seq_c:512"
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1
UTC8_OFFSET_MS = 8 * 60 * 60 * 1000
DAY_MS = 24 * 60 * 60 * 1000
UNIX_EPOCH_DATE = date(1970, 1, 1)


@dataclass(frozen=True)
class FeaturePosition:
    """One flattened static-feature position."""

    feature: str
    slot: int
    vocab_size: int
    is_identifier: bool = False


@dataclass
class FeatureTable:
    """Preloaded static feature table indexed by raw entity id."""

    name: str
    id_column: str
    values: np.ndarray
    positions: list[FeaturePosition]

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])

    @property
    def vocab_sizes(self) -> list[int]:
        return [position.vocab_size for position in self.positions]

    @property
    def identifier_mask(self) -> list[bool]:
        return [position.is_identifier for position in self.positions]

    @property
    def position_names(self) -> list[str]:
        names = []
        for position in self.positions:
            if position.slot == 0:
                names.append(position.feature)
            else:
                names.append(f"{position.feature}[{position.slot}]")
        return names

    @property
    def feature_specs(self) -> dict[str, dict[str, Any]]:
        """Feature-level column slices and embedding-table hints.

        ``start`` and ``end`` are column offsets in ``values`` / batch feature
        tensors. Non-identifier features reserve 0 for missing/padding, so their
        embedding table should usually use ``num_feat_value + 1`` rows.
        Identifier columns are raw 0-based ids and use ``num_feat_value`` rows.
        """

        specs: dict[str, dict[str, Any]] = {}
        for column, position in enumerate(self.positions):
            if position.feature not in specs:
                num_embeddings = (
                    position.vocab_size
                    if position.is_identifier
                    else position.vocab_size + 1
                )
                specs[position.feature] = {
                    "start": column,
                    "end": column,
                    "columns": [],
                    "feat_len": 0,
                    "num_feat_value": position.vocab_size,
                    "num_embeddings": num_embeddings,
                    "padding_idx": None if position.is_identifier else 0,
                    "is_identifier": position.is_identifier,
                }

            spec = specs[position.feature]
            spec["end"] = column + 1
            spec["columns"].append(column)
            spec["feat_len"] += 1
        return specs

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "id_column": self.id_column,
            "dim": self.dim,
            "flat_position_names": self.position_names,
            "features": self.feature_specs,
        }


@dataclass
class SplitSummary:
    min_day: int
    max_day: int
    unique_days: list[int]
    train_days: list[int]
    valid_days: list[int]
    test_days: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_day": self.min_day,
            "max_day": self.max_day,
            "unique_days": self.unique_days,
            "train_days": self.train_days,
            "valid_days": self.valid_days,
            "test_days": self.test_days,
            "min_date_utc8": day_ordinal_to_date(self.min_day),
            "max_date_utc8": day_ordinal_to_date(self.max_day),
            "train_dates_utc8": [day_ordinal_to_date(day) for day in self.train_days],
            "valid_dates_utc8": [day_ordinal_to_date(day) for day in self.valid_days],
            "test_dates_utc8": [day_ordinal_to_date(day) for day in self.test_days],
        }


@dataclass
class DatasetBundle:
    user_table: FeatureTable
    item_table: FeatureTable
    train_dataset: "InteractionDataset"
    valid_dataset: "InteractionDataset"
    test_dataset: "InteractionDataset"
    split_summary: SplitSummary
    sequence_lengths: dict[str, int]

    def feature_schema(self) -> dict[str, Any]:
        return {
            "user": self.user_table.schema(),
            "item": self.item_table.schema(),
            "sequences": {
                name: {
                    "max_len": max_len,
                    "video_id_key": f"{name}_video_id",
                    "time_bucket_key": f"{name}_time_bucket",
                    "padding_mask_key": f"{name}_padding_mask",
                    "padding_side": "left",
                    "padding_mask_true_means": "padding",
                    "num_time_buckets": NUM_TIME_BUCKETS,
                }
                for name, max_len in self.sequence_lengths.items()
            },
        }


def day_ordinal_to_date(day: int) -> str:
    return (UNIX_EPOCH_DATE + timedelta(days=int(day))).isoformat()


def resolve_data_paths(data_dir: str | Path) -> dict[str, Path]:
    base_dir = Path(data_dir)
    paths = {
        "user_features": base_dir / "user_features_mapped.csv",
        "user_info": base_dir / "user_statistic_info.json",
        "item_features": base_dir / "item_features_mapped.csv",
        "item_info": base_dir / "item_statistic_info.json",
        "interactions": base_dir / "standard_interactions.csv",
        "positive_sequences": base_dir / "user_positive_sequences.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required data files are missing under "
            f"{base_dir}: {', '.join(missing)}"
        )
    return paths


def load_feature_info(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    for feature, spec in info.items():
        if "feat_len" not in spec:
            raise ValueError(f"{path} feature {feature!r} misses feat_len")
        if "num_feat_value" not in spec:
            raise ValueError(f"{path} feature {feature!r} misses num_feat_value")
    return info


def parse_sequence_config(config: str) -> dict[str, int]:
    sequence_lengths: dict[str, int] = {}
    for part in config.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition(":")
        if not sep:
            raise ValueError(f"Invalid sequence config item: {part!r}")
        name = name.strip()
        max_len = int(value.strip())
        if not name:
            raise ValueError(f"Invalid empty sequence name in config: {part!r}")
        if max_len <= 0:
            raise ValueError(f"Sequence {name!r} max_len must be positive")
        sequence_lengths[name] = max_len
    if not sequence_lengths:
        raise ValueError("sequence_config must define at least one sequence")
    return sequence_lengths


def load_positive_sequences(
    sequence_path: Path,
    sequence_lengths: dict[str, int],
) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    with sequence_path.open("rb") as f:
        raw_sequences = pickle.load(f)

    sequences: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for raw_user_id, raw_seq_dict in raw_sequences.items():
        user_id = int(raw_user_id)
        parsed_seq_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for seq_name in sequence_lengths:
            if seq_name not in raw_seq_dict:
                continue
            video_list, time_list = raw_seq_dict[seq_name]
            video_ids = np.asarray(video_list, dtype=np.int64)
            times = np.asarray(time_list, dtype=np.int64)
            if video_ids.shape[0] != times.shape[0]:
                raise ValueError(
                    f"{sequence_path} user {user_id} sequence {seq_name!r} has "
                    f"{video_ids.shape[0]} videos but {times.shape[0]} timestamps"
                )
            parsed_seq_dict[seq_name] = (video_ids, times)
        sequences[user_id] = parsed_seq_dict
    return sequences


def parse_int_array_value(value: Any, feat_len: int, delimiter: str = ",") -> list[int]:
    """Parse a comma-delimited multi-value feature into a padded int list."""

    out = [0] * feat_len
    if pd.isna(value):
        return out
    text = str(value).strip()
    if not text:
        return out
    parts = [part.strip() for part in text.split(delimiter) if part.strip()]
    for idx, part in enumerate(parts[:feat_len]):
        try:
            out[idx] = int(float(part))
        except ValueError:
            out[idx] = 0
    return out


def build_feature_positions(
    info: dict[str, dict[str, Any]],
    id_column: str,
) -> list[FeaturePosition]:
    positions: list[FeaturePosition] = []
    for feature, spec in info.items():
        feat_len = int(spec["feat_len"])
        vocab_size = int(spec["num_feat_value"])
        for slot in range(feat_len):
            positions.append(
                FeaturePosition(
                    feature=feature,
                    slot=slot,
                    vocab_size=vocab_size,
                    is_identifier=(feature == id_column),
                )
            )
    return positions


def load_static_feature_table(
    feature_path: Path,
    info_path: Path,
    id_column: str,
    name: str,
    chunksize: int = 500_000,
) -> FeatureTable:
    """Load static feature CSV into a dense numpy array indexed by raw id."""

    info = load_feature_info(info_path)
    feature_names = list(info.keys())
    if id_column not in info:
        raise ValueError(f"{info_path} does not contain id column {id_column!r}")

    table_size = int(info[id_column]["num_feat_value"])
    positions = build_feature_positions(info, id_column=id_column)
    total_dim = len(positions)
    values = np.zeros((table_size, total_dim), dtype=np.int32)

    input_columns = pd.read_csv(feature_path, nrows=0).columns
    missing = sorted(set(feature_names) - set(input_columns))
    if missing:
        raise ValueError(f"{feature_path} misses static features: {missing}")

    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for feature in feature_names:
        feat_len = int(info[feature]["feat_len"])
        offsets[feature] = (offset, feat_len)
        offset += feat_len

    for chunk in pd.read_csv(
        feature_path,
        usecols=feature_names,
        chunksize=chunksize,
    ):
        entity_ids = (
            pd.to_numeric(chunk[id_column], errors="coerce")
            .fillna(-1)
            .astype(np.int64)
            .to_numpy()
        )
        valid_rows = (entity_ids >= 0) & (entity_ids < table_size)
        if not np.any(valid_rows):
            continue
        valid_entity_ids = entity_ids[valid_rows]

        for feature in feature_names:
            start, feat_len = offsets[feature]
            series = chunk.loc[valid_rows, feature]
            if feat_len == 1:
                encoded = (
                    pd.to_numeric(series, errors="coerce")
                    .fillna(0)
                    .astype(np.int64)
                    .to_numpy()
                )
                values[valid_entity_ids, start] = encoded.astype(np.int32)
            else:
                delimiter = str(info[feature].get("array_delimiter", ","))
                encoded_rows = [
                    parse_int_array_value(value, feat_len, delimiter=delimiter)
                    for value in series
                ]
                values[valid_entity_ids, start : start + feat_len] = np.asarray(
                    encoded_rows,
                    dtype=np.int32,
                )

    return FeatureTable(
        name=name,
        id_column=id_column,
        values=values,
        positions=positions,
    )


class InteractionDataset(Dataset):
    """Map-style interaction dataset with static user/item feature lookup."""

    def __init__(
        self,
        name: str,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        time_ms: np.ndarray,
        labels: np.ndarray,
        user_features: np.ndarray,
        item_features: np.ndarray,
        positive_sequences: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
        sequence_lengths: dict[str, int],
    ) -> None:
        self.name = name
        self.user_ids = user_ids.astype(np.int64, copy=False)
        self.item_ids = item_ids.astype(np.int64, copy=False)
        self.time_ms = time_ms.astype(np.int64, copy=False)
        self.labels = labels.astype(np.int8, copy=False)
        self.user_features = user_features
        self.item_features = item_features
        self.positive_sequences = positive_sequences
        self.sequence_lengths = sequence_lengths
        self.user_dim = int(user_features.shape[1])
        self.item_dim = int(item_features.shape[1])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _build_sequence_tensors(
        self,
        user_id: int,
        time_ms: int,
        seq_name: str,
        max_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq_video_ids = np.zeros(max_len, dtype=np.int32)
        time_bucket = np.zeros(max_len, dtype=np.int16)
        padding_mask = np.ones(max_len, dtype=bool)

        user_sequences = self.positive_sequences.get(user_id)
        if not user_sequences or seq_name not in user_sequences:
            return seq_video_ids, time_bucket, padding_mask

        video_ids, times = user_sequences[seq_name]
        cutoff = int(np.searchsorted(times, time_ms, side="left"))
        if cutoff <= 0:
            return seq_video_ids, time_bucket, padding_mask

        start = max(0, cutoff - max_len)
        history_video_ids = video_ids[start:cutoff]
        history_times = times[start:cutoff]
        history_len = int(history_video_ids.shape[0])
        target_start = max_len - history_len

        seq_video_ids[target_start:] = history_video_ids
        padding_mask[target_start:] = False
        diff_seconds = np.maximum((time_ms - history_times) // 1000, 0)
        raw_buckets = np.clip(
            np.searchsorted(BUCKET_BOUNDARIES, diff_seconds),
            0,
            len(BUCKET_BOUNDARIES) - 1,
        )
        time_bucket[target_start:] = raw_buckets.astype(np.int16) + 1
        return seq_video_ids, time_bucket, padding_mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_id = int(self.user_ids[idx])
        item_id = int(self.item_ids[idx])
        time_ms = int(self.time_ms[idx])

        if 0 <= user_id < self.user_features.shape[0]:
            user_feat = self.user_features[user_id]
        else:
            user_feat = np.zeros(self.user_dim, dtype=np.int32)

        if 0 <= item_id < self.item_features.shape[0]:
            item_feat = self.item_features[item_id]
        else:
            item_feat = np.zeros(self.item_dim, dtype=np.int32)

        sample = {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "video_id": torch.tensor(item_id, dtype=torch.long),
            "time_ms": torch.tensor(time_ms, dtype=torch.long),
            "user_feat": torch.from_numpy(user_feat.astype(np.int32, copy=False)),
            "item_feat": torch.from_numpy(item_feat.astype(np.int32, copy=False)),
            "labels": torch.from_numpy(self.labels[idx]),
        }
        for seq_name, max_len in self.sequence_lengths.items():
            seq_video_ids, time_bucket, padding_mask = self._build_sequence_tensors(
                user_id=user_id,
                time_ms=time_ms,
                seq_name=seq_name,
                max_len=max_len,
            )
            sample[f"{seq_name}_video_id"] = torch.from_numpy(seq_video_ids)
            sample[f"{seq_name}_time_bucket"] = torch.from_numpy(time_bucket)
            sample[f"{seq_name}_padding_mask"] = torch.from_numpy(padding_mask)
        return sample


def add_utc8_day(interactions: pd.DataFrame) -> pd.DataFrame:
    local_ms = interactions["time_ms"].to_numpy(dtype=np.int64, copy=False) + UTC8_OFFSET_MS
    interactions = interactions.copy()
    interactions["day_utc8"] = local_ms // DAY_MS
    return interactions


def load_sorted_interactions(
    interaction_path: Path,
    max_rows: int | None = None,
) -> pd.DataFrame:
    input_columns = pd.read_csv(interaction_path, nrows=0).columns
    missing = sorted(set(INTERACTION_COLUMNS) - set(input_columns))
    if missing:
        raise ValueError(f"{interaction_path} misses interaction columns: {missing}")

    dtype = {
        "user_id": "int64",
        "video_id": "int64",
        "time_ms": "int64",
        "is_click": "int8",
        "is_like": "int8",
        "is_comment": "int8",
        "long_view": "int8",
    }
    interactions = pd.read_csv(
        interaction_path,
        usecols=INTERACTION_COLUMNS,
        dtype=dtype,
        nrows=max_rows,
    )
    interactions = add_utc8_day(interactions)
    interactions = interactions.sort_values("time_ms", kind="mergesort").reset_index(drop=True)
    return interactions


def split_interactions_by_day(
    interactions: pd.DataFrame,
    train_start: int = 8,
    valid_days: int = 1,
    test_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitSummary]:
    if valid_days < 1 or test_days < 1:
        raise ValueError("valid_days and test_days must be positive")
    if train_start < 0:
        raise ValueError("train_start must be >= 0")

    unique_days = sorted(int(day) for day in interactions["day_utc8"].unique())
    required_days = train_start + valid_days + test_days + 1
    if len(unique_days) < required_days:
        raise ValueError(
            f"Not enough UTC+8 days for split: got {len(unique_days)}, "
            f"need at least {required_days}. Try a smaller train_start or use more rows."
        )

    test_day_list = unique_days[-test_days:]
    valid_day_list = unique_days[-(test_days + valid_days) : -test_days]
    train_day_list = unique_days[train_start : -(test_days + valid_days)]
    if not train_day_list:
        raise ValueError("Empty train split. Adjust train_start/valid_days/test_days.")

    train_df = interactions[interactions["day_utc8"].isin(train_day_list)]
    valid_df = interactions[interactions["day_utc8"].isin(valid_day_list)]
    test_df = interactions[interactions["day_utc8"].isin(test_day_list)]
    summary = SplitSummary(
        min_day=unique_days[0],
        max_day=unique_days[-1],
        unique_days=unique_days,
        train_days=train_day_list,
        valid_days=valid_day_list,
        test_days=test_day_list,
    )
    return train_df, valid_df, test_df, summary


def dataframe_to_dataset(
    name: str,
    df: pd.DataFrame,
    user_table: FeatureTable,
    item_table: FeatureTable,
    positive_sequences: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    sequence_lengths: dict[str, int],
) -> InteractionDataset:
    labels = df[TARGET_COLUMNS].to_numpy(dtype=np.int8, copy=True)
    return InteractionDataset(
        name=name,
        user_ids=df["user_id"].to_numpy(dtype=np.int64, copy=True),
        item_ids=df["video_id"].to_numpy(dtype=np.int64, copy=True),
        time_ms=df["time_ms"].to_numpy(dtype=np.int64, copy=True),
        labels=labels,
        user_features=user_table.values,
        item_features=item_table.values,
        positive_sequences=positive_sequences,
        sequence_lengths=sequence_lengths,
    )


def build_dataset_bundle(
    data_dir: str | Path,
    train_start: int = 8,
    valid_days: int = 1,
    test_days: int = 1,
    static_chunksize: int = 500_000,
    max_rows: int | None = None,
    sequence_config: str = DEFAULT_SEQUENCE_CONFIG,
) -> DatasetBundle:
    paths = resolve_data_paths(data_dir)
    sequence_lengths = parse_sequence_config(sequence_config)

    user_table = load_static_feature_table(
        paths["user_features"],
        paths["user_info"],
        id_column="user_id",
        name="user",
        chunksize=static_chunksize,
    )
    item_table = load_static_feature_table(
        paths["item_features"],
        paths["item_info"],
        id_column="video_id",
        name="item",
        chunksize=static_chunksize,
    )

    positive_sequences = load_positive_sequences(
        paths["positive_sequences"],
        sequence_lengths=sequence_lengths,
    )
    interactions = load_sorted_interactions(paths["interactions"], max_rows=max_rows)
    train_df, valid_df, test_df, split_summary = split_interactions_by_day(
        interactions,
        train_start=train_start,
        valid_days=valid_days,
        test_days=test_days,
    )

    return DatasetBundle(
        user_table=user_table,
        item_table=item_table,
        train_dataset=dataframe_to_dataset(
            "train",
            train_df,
            user_table,
            item_table,
            positive_sequences,
            sequence_lengths,
        ),
        valid_dataset=dataframe_to_dataset(
            "valid",
            valid_df,
            user_table,
            item_table,
            positive_sequences,
            sequence_lengths,
        ),
        test_dataset=dataframe_to_dataset(
            "test",
            test_df,
            user_table,
            item_table,
            positive_sequences,
            sequence_lengths,
        ),
        split_summary=split_summary,
        sequence_lengths=sequence_lengths,
    )
