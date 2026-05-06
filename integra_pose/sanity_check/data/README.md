# Sanity Check — Data Fixtures

This folder contains **synthetic** test inputs used by
`integra_pose.sanity_check.runner`. None of these files contain real
animal data, ML model weights, or copyrighted media.

## What's here

```
data/
├── synthetic_dataset.yaml         ← class names for the synthetic batch
├── synthetic_labels/              ← 10 YOLO-format .txt label files
│   ├── frame_0000.txt
│   ├── frame_0001.txt
│   └── ...
└── expected/
    └── expected_bouts_summary.txt ← human-readable expected result for the
                                     bout-analyzer end-to-end stage
```

## Why no videos / model

Bundling a YOLO-trained pose model would require shipping AGPL-3.0
weights with every IntegraPose install — putting a license obligation
on every user (including those who later commercialize). Bundling
sample videos has separate IACUC + redistribution concerns.

The synthetic YOLO label files exercise the same parser, the same
preprocessing, the same bout-detection math the wizard uses on real
data. They just don't validate ML inference itself — that's what the
wizard's existing **Model Preflight** button is for, run against the
researcher's own model.

## What the synthetic batch represents

10 frames of a single animal switching between two behaviour classes:

* Frames 0–4: class 0 (**walking**) at the centre-left of the frame
* Frames 5–9: class 1 (**rearing**) at the centre of the frame

With ``max_gap_frames=2`` and ``min_bout_frames=2``, the bout analyser
should produce **exactly 2 bouts** (one per class). The expected
counts live in ``expected/expected_bouts_summary.txt``; the runner
diffs against them.

## Format note

Each label file is one line:

```
class_id  cx  cy  w  h  track_id
```

All coordinates are normalised to [0, 1] (YOLO convention). Track ID
is included so the bout analyser's tracking path is exercised.
