# Data format and availability

The experiments use [KuaiRand-1K](https://github.com/chongminggao/KuaiRand), whose original files are available from the dataset authors via [Zenodo](https://zenodo.org/records/10439422). The training scripts in this repository consume **preprocessed** files, not the original KuaiRand archive. No data, user identifiers, or preprocessing artifact is included in this anonymous code release. The exact transformation from the original files to the six inputs below is not provided here; consequently, the full numerical results cannot be reproduced from this repository alone. This limitation should be reflected in any reproducibility statement.

Place these six files in one directory and set `UNIFORMER_DATA_DIR` to it:

| File | Expected contents |
| --- | --- |
| `user_features_mapped.csv` | One row per user; `user_id` plus columns named by `user_statistic_info.json`. |
| `user_statistic_info.json` | JSON mapping each user feature name to `feat_len` and `num_feat_value`, with optional `array_delimiter`. |
| `item_features_mapped.csv` | One row per item; `video_id` plus columns named by `item_statistic_info.json`. |
| `item_statistic_info.json` | Same feature metadata format for item features. |
| `standard_interactions.csv` | `user_id`, `video_id`, `time_ms`, `is_click`, `is_like`, `is_comment`, `long_view`. Times are Unix milliseconds; the four targets are integer labels. |
| `user_positive_sequences.pkl` | Python-pickled mapping: `user_id -> {sequence_name: (video_id_list, time_ms_list)}`. The default sequence names are `seq_a`, `seq_b`, and `seq_c`; each sequence must be ordered by timestamp. Only load a pickle from a trusted source. |

User and item IDs must index the corresponding static feature tables. Non-identifier feature value `0` is reserved for missing or padding. Multi-value static features are comma-separated by default and are padded or truncated to the configured `feat_len`.

The loader sorts interactions by `time_ms`, groups them into UTC+8 calendar days, uses the final two days for test and the preceding two days for validation, and uses the earlier days for training (`train_start=0`). Each sequence is cut off strictly before the interaction timestamp and left-padded to length 256 by the default run scripts. The data loading and split code is in each experiment's `dataset.py`.
