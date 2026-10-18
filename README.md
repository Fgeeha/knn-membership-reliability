# knn-membership-reliability

Code, preregistrations and raw results for the study *Input membership and decision
reliability in compressed plant disease diagnosis* (N. S. Kolesnikov, Volgograd State
Technical University; manuscript in preparation).

Код, предрегистрации и сырые результаты к статье «Принадлежность входа и надёжность
решения в сжатом конвейере диагностики болезней растений».

The study asks what the built-in confidence of a compressed nearest-neighbour (1-NN)
plant disease pipeline measures: whether the input belongs to the reference space, or
whether the decision is correct. It combines a 2⁴ factorial design (105 runs), a
preregistered replication on an independent field dataset (240 runs), a gallery-size
bridge experiment, a repeat on a second encoder, per-query bootstrap and an on-device
cost measurement on Raspberry Pi 4B.

## Contents

| Path | What it is |
|---|---|
| `src/reliability_gate/` | library: ten confidence signals over sorted gallery distances (`signals.py`), tie-aware AUROC, risk–coverage, bootstrap CI (`metrics.py`), two-loop rejection gate (`gate.py`), risk-controlled threshold for `class_gap` (`calibration.py`), audit of any classifier's confidence (`audit.py`); numpy only |
| `tests/` | unit tests of the library |
| `scripts/` | experiment pipeline by stage; the step number is in each docstring |
| `preregistration/` | `FACTORIAL_DESIGN.md` (stage S2, fixed 2026-08-16), `S3_PREREGISTRATION.md` (2026-08-16), `S4_PREREGISTRATION.md` (2026-08-17); kept unchanged after the results came in |
| `label_alignment.json`, `label_alignment.yaml` | rule-based alignment of 13 PlantWild classes to the canonical label space; its sha256 is recorded in every result file |
| `results/` | raw result files of every run (JSON with `run_meta`, logs), split indices and label arrays |

Stages: S1 — confidence profiles of the deployed bundle; S2 — factorial design on
PlantVillage → PlantDoc; S3 — replication on PlantWild v2, bridge, two-loop criterion,
OOD; S4 — ResNet-50 repeat, membership operating point, per-query bootstrap.

## What runs out of the box

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_reliability_gate.py
PYTHONPATH=src python -m reliability_gate.gate        # gate self-check
PYTHONPATH=src python -m reliability_gate.audit dump.csv --score conf --correct correct
```

The library needs only numpy; the tests also use scikit-learn to cross-check AUROC.
Every number in the manuscript can be traced to a file in `results/`.

## What needs the data and the pipeline

The experiment scripts and the `verify_*` checks read feature caches extracted from the
images (about 1.3 GB). The caches are not published: they are derived from datasets
whose licences do not allow redistribution of derivatives (PlantWild v2 is
CC BY-NC-ND 4.0). To regenerate them you need the datasets below and the deployed
pipeline `plantdiag-edge` (ONNX runner, iCORAL adaptation and the PolarQuant gallery
codec), which the scripts import from a sibling directory:

```
parent/
├── knn-membership-reliability/
└── plantdiag-edge/
```

Datasets: PlantVillage (github.com/spMohanty/PlantVillage-Dataset), PlantDoc
(github.com/pratikkayal/PlantDoc-Dataset), PlantWild v2 (huggingface.co/datasets/uqtwei2/PlantWild,
revision 527a72eb), FCDD fruits (kaggle.com/datasets/ateebnoone/fruits-dataset-for-fruit-disease-classification),
ibean (github.com/AI-Lab-Makerere/ibean). Images are not redistributed here.

Environment of the reported runs: Python 3.12–3.13, numpy 1.26.4 and 2.5.2 (results
agree on both), scikit-learn 1.9, onnxruntime 1.29, torch 2.7; the device cost was
measured on Raspberry Pi 4B (aarch64, Python 3.11.2).

## Attribution

PolarQuant (3-bit gallery compression after a randomized Walsh–Hadamard rotation) is
not a contribution of this work; it is taken from TurboQuantCV. iCORAL is an
incremental form of CORAL (Sun, Feng, Saenko, AAAI 2016).

## Licence

Code: MIT (see `LICENSE`). Result files and preregistrations: CC BY 4.0.
