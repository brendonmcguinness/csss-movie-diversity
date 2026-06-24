# Movie Interaction VRAE

This project forecasts the scene-local character interaction graph of the next movie scene from the preceding five scenes. It is a CPU-only PyTorch adaptation of the variational recurrent autoencoder idea.

## Representation

Each input movie JSON must have the form:

```json
{
  "title": "Up",
  "character_ids": [
    {
      "name": "CARL",
      "id": 0
    }
  ],
  "scenes": [
    {
      "heading": "INT. MOVIE THEATRE - CONTINUOUS",
      "character_ids": [0, 1]
    }
  ]
}
```

Character IDs must be `0, 1, ..., n - 1` in chronological order of first speaking appearance. Every scene's speaking characters form a clique, including diagonal entries. Matrices are scene-local, not cumulative. Because they are symmetric, only the upper triangle including the diagonal is stored, giving 32,896 binary coordinates for a 256-node graph.

The dataset builder performs a complete validation pass before writing output. If any movie has more than 256 characters, it lists every offending file and exits without silently truncating the data.

## Installation

Create a virtual environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No CUDA installation is required. The training, evaluation, and prediction scripts explicitly use the CPU.

## 1. Build the processed dataset

Place movie JSON files in a directory such as `data/json/`, then run:

```bash
python build_dataset.py \
  --input-dir data/json \
  --output-dir data/processed \
  --config config.json
```

Add `--recursive` when JSON files are stored in subdirectories. The command creates compressed per-movie scene arrays, per-movie metadata, a deterministic movie-level split manifest, and the training-set global-frequency baseline.

The default split is approximately 72% training, 8% validation, and 20% testing. Splitting is by movie, so windows from one movie never enter multiple subsets.

## 2. Train

```bash
python train.py \
  --processed-dir data/processed \
  --output-dir runs/default \
  --config config.json
```

The best validation checkpoint is written to `runs/default/best.pt`; the latest checkpoint is written to `runs/default/last.pt`. Resume with:

```bash
python train.py \
  --processed-dir data/processed \
  --output-dir runs/default \
  --config config.json \
  --resume runs/default/last.pt
```

The validation criterion is approximate area under the precision-recall curve on strict upper-triangle interaction entries. KL warm-up, weighted binary cross-entropy, gradient clipping, and early stopping are enabled by default.

## 3. Evaluate

```bash
python evaluate.py \
  --processed-dir data/processed \
  --checkpoint runs/default/best.pt \
  --output runs/default/evaluation.json
```

Evaluation reports separate metrics for:

- all packed coordinates;
- diagonal coordinates, representing character appearance;
- strict upper-triangle coordinates, representing pairwise interactions.

It compares the model against all-zero, repeat-last-scene, recent-window-union, and global-frequency baselines. Thresholds are chosen using validation movies and then held fixed on test movies. Approximate precision-recall statistics use probability histograms rather than storing every dense prediction in memory.

## 4. Predict a next scene

```bash
python predict.py \
  --movie-json data/json/up.json \
  --checkpoint runs/default/best.pt \
  --output-prefix predictions/up_next
```

By default, the final five scenes in the JSON are used to predict the scene after the end of the screenplay. To run a historical forecast within a known movie, specify the zero-based final observed scene:

```bash
python predict.py \
  --movie-json data/json/up.json \
  --checkpoint runs/default/best.pt \
  --last-observed-scene 40 \
  --output-prefix predictions/up_after_scene_40
```

The script writes:

- an NPZ file containing packed and full 256 by 256 probability and binary matrices;
- a JSON file containing the top predicted character appearances and interactions.

## Configuration

`config.json` controls the window size, random seed, model dimensions, CPU thread count, optimization settings, histogram resolution, and top-k metrics. The default model uses a 128-dimensional scene embedding, a 128-unit LSTM, and a 32-dimensional latent variable.

## Tests

```bash
pytest -q
```

## Methods document

The full project methodology, including screenplay parsing, JSON construction, cumulative D3 visualization, scene-local graph construction, VRAE forecasting, and evaluation, is in `methods.tex`. A compiled copy is included as `methods.pdf`.
