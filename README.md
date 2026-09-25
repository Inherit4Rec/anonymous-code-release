# Anonymous code for Inherit4Rec

This repository contains two standalone PyTorch implementations used to study parameter inheritance during model growth:

- `Inherit4Rec-D2D/`: dense-to-dense growth. Train a narrow model for five epochs, widen the FIM per-token feed-forward layers, then train for five more epochs. The growth step is implemented in `growth.py`.
- `Inherit4Rec-D2S/`: dense-to-sparse growth. Train a dense model for five epochs, convert the FIM feed-forward layers to a sparse mixture of experts, then train for five more epochs. The partition and conversion steps are implemented in `moe.py` and `model.py`.

## Environment

Use Python 3.9 or newer. The pinned dependencies are in `requirements.txt`.

```bash
bash install_dependencies.sh
.venv/bin/python Inherit4Rec-D2D/smoke_test.py
.venv/bin/python Inherit4Rec-D2S/smoke_test.py
```

The smoke tests use synthetic tensors and require no dataset or GPU. Full training was configured for a GPU. `run.sh` accepts `GPU_ID` to select a CUDA device and passes additional arguments through to `train.py`.

## Data and full training

The training code expects six **preprocessed** KuaiRand-1K files. Data files are not included in this repository. See [DATA.md](DATA.md) for the source dataset, exact input schema, split rules, and the limitation on reproducing the original preprocessing. Obtain or prepare data in that format before running either experiment.

```bash
export UNIFORMER_DATA_DIR=/path/to/preprocessed/kuairand_1k
GPU_ID=0 bash Inherit4Rec-D2D/run.sh
GPU_ID=0 bash Inherit4Rec-D2S/run.sh
```

The default output directory for each experiment is its own directory; set `UNIFORMER_OUTPUT_DIR` to move checkpoints and logs elsewhere. The run scripts fix model sizes, seeds, optimizer settings, batch sizes, and conversion settings for the reported configuration. `train.py` exposes their command-line alternatives.

Each experiment starts from random initialization and uses the same chronological split. The D2D and D2S runs are independent. Each run writes `run_summary.json` and model-only checkpoints. Full-result reproduction requires the same preprocessed data and sufficient compute; the data-free smoke tests check the conversion logic and a small training path.
