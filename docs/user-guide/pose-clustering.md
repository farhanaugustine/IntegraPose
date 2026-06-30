# Behavior Clustering (Tab 7)

Behavior Clustering - also called sub-behavior discovery - turns pose
data into a per-class, sub-behavior-level view of your animals' movement.
Where Bout Analytics on Tab 6 tells you *"the animal walked for 320
frames"*, Tab 7 splits *walking* into the distinct kinds of walking your
data actually contains - for example, forward locomotion, lateral creep,
and a brief scurry - and gives each one a name, a score, a stability
check, and (optionally) a folder of bout clips ready to drop into a
downstream classifier. The underlying technique is a VAE + HMM pipeline,
which is reflected in the in-app tab title: **Behavior Clustering (VAE +
HMM)**.

## What it is for

Use Behavior Clustering when you want to:

- Find the **sub-behaviors hidden inside a known class** ("how many distinct kinds of *grooming* are in this dataset?").
- Surface **candidate behaviors you didn't pre-annotate** for a YOLO model.
- Decide **which sub-behaviors are real** and which are noise - with a stability check and a signal score, not a single arbitrary threshold.
- Hand off **named, organized clip folders** to a downstream classifier, YOLO clip trainer, or manual review workflow.

It is not a replacement for Tab 6 Bout Analytics, and it is not designed
for detection-only workflows: the clustering depends on pose features.

## At a glance

| Best for | Main input | Typical output |
| --- | --- | --- |
| Discovering sub-behaviors within a YOLO pose class | Pose detections (Tab 6 / batch manifests / raw) | Per-frame sub-cluster labels, bouts CSV, candidate scores, named clip folders |

## How a session flows

```text
Pose data (per-class)
  -> UMAP + HDBSCAN per class           # Run Behavior Clustering
  -> Bout aggregation                    # contiguous frames per sub-cluster
  -> (optional) Stability audit          # is the partition seed-stable?
  -> Signal score                        # which sub-clusters look real?
  -> Review candidates                   # ranked, color-coded list
  -> Name sub-behaviors                  # 9-bout triptych dialog
  -> Export sub-cluster clips (optional) # behavior-name folders + manifest
```

You can stop at any step. A lab that only wants the bouts CSV can stop
right after the run; a lab that wants classifier training data goes
through the full chain.

## Three ways to bring data in

| Entry path | Use it when |
| --- | --- |
| Continue from latest Bout Analytics run | You just finished a Tab 6 run on pose data |
| Import analytics manifest(s) | You want to combine one or more Tab 6 / batch results |
| Add manual sources | You want pose-directory + video pairs without going through Tab 6 first |

Each pose source can be assigned to a group (`Control`, `Treatment`, `WT`,
`KO`, etc.) so downstream summaries respect your study design.

## Quick start

### Option A - Continue from Bout Analytics

1. Finish a pose-based Bout Analytics run on Tab 6.
2. Switch to Tab 7 and click **Continue from Latest Tab 6 Run**.
3. Pick the target group name.
4. Click **Run Behavior Clustering**.

### Option B - Import an analytics manifest

1. On Tab 7, click **Import Analytics Manifest(s)...**.
2. Select one or more `run_manifest.json` files.
3. Pick the target group name and run.

### Option C - Add manual sources

1. On Tab 7, add a group.
2. Add pose-directory + video-source pairs.
3. Fill in keypoint and behavior names.
4. Run.

## What the run does

For each YOLO class with enough frames, Behavior Clustering:

1. Builds a feature vector per detection (keypoints + bbox-derived terms).
2. Reduces dimensionality with UMAP.
3. Clusters with HDBSCAN - frames that don't fit any sub-cluster are
   labelled noise (`-1`) instead of being forced into a group.
4. Aggregates contiguous same-label frames into bouts.
5. Saves results next to your output folder.

Each run is stamped with a unique `run_id`, so saved cluster names from
an earlier run with different parameters won't be reused if you change
clustering settings.

## The four review steps after a run

### 1. State Summary (always shown)

The Sub-Behavior Summary panel on Tab 3 shows, per class:

- frame count
- number of sub-behaviors found
- number of bouts
- noise-frame count
- (when the audit ran) stability ARI and a Stable / Unstable badge

### 2. Run stability audit (optional, opt-in)

When **Run stability audit** is checked, the run re-clusters with N
additional random seeds and reports the mean Adjusted Rand Index for
each class. ARI >= 0.5 is reported as **Stable**; below that, the cluster
boundaries are seed-sensitive and should be treated as soft. The audit
roughly multiplies the discovery runtime by N + 1 - use it for a final
check, not for every parameter sweep.

### 3. Review Candidate Sub-Clusters

Click **Review Candidate Sub-Clusters** to open a ranked, color-coded
table of every sub-cluster in the run. Each row carries a 0-1 candidate
score (size, subject coverage, mean bout duration, and stability ARI),
a verdict (Likely real / Review / Likely noise), and a short note when
something looks off. The table is sorted strongest first. The same data
is saved as `sub_behavior_candidate_scores.csv` next to the run.

### 4. Name Sub-Behaviors...

Click **Name Sub-Behaviors...** to walk the sub-clusters one at a time.
Each step shows a 3 x 3 grid of triptychs - three frames per bout
(start / middle / end) for the nine longest bouts of that sub-cluster.
Type a name (`forward-locomotion`, `lateral-creep`, etc.) and click
**Save & Next**. Names are written to `state_names.json` next to the
run output.

## Export sub-cluster clips (optional, recommended for classifier training)

If you want to feed your sub-behaviors into a downstream classifier
trainer, click **Export Sub-cluster Clips**. The result is a
ready-to-train layout:

```text
<output_folder>/sub_cluster_clips/
  forward_locomotion/
    video1__t0__b0001__f12-87.mp4
    video2__t0__b0023__f120-178.mp4
  lateral_creep/
    video1__t0__b0007__f200-260.mp4
  walking__subcluster_2/         # not yet named, fallback name
    video1__t0__b0011__f600-650.mp4
  clip_manifest.csv
```

- One `.mp4` per bout, contiguous frames.
- One folder per sub-behavior; folder name comes from `state_names.json`.
- A `clip_manifest.csv` row per clip with status, behavior, source video,
  track, subject, frame range, and `run_id`.

The export is **always optional** - it never runs unless you click the
button, and you can stop at the bouts CSV if that's all your project
needs.

## Main parameter areas

### Inputs and grouping

- keypoint names
- behavior names
- YOLO label format assumptions
- normalization reference points
- skeleton connections
- group and source setup

### Clustering parameters

- **Min Bout Duration** (frames) - bouts shorter than this are dropped from the bouts table.
- **UMAP Neighbors** - `n_neighbors` for UMAP. Larger values preserve global structure.
- **UMAP Components** - number of UMAP dimensions HDBSCAN clusters in.
- **HDBSCAN Min Cluster Size** - smallest sub-cluster HDBSCAN will return.
- **Stability seeds (N)** - additional clusterings used for the optional audit.

## Outputs

| File | What it contains |
| --- | --- |
| `sub_behavior_per_frame.csv` | Per-frame namespaced labels (`0:1`, `0:2`, ..., `-1` for noise) |
| `sub_behavior_bouts.csv` | One row per bout: class, sub-cluster, start/end frame, duration, video |
| `sub_behavior_candidate_scores.csv` | Ranked sub-cluster candidates with score, verdict, and notes |
| `sub_behavior_summary.txt` | Per-class summary + run_id |
| `sub_behavior_stability.json` | Per-class ARI matrices when the audit ran |
| `sub_behavior_run_id.txt` | UUID identifying this run |
| `state_names.json` | User-supplied sub-behavior names (when the naming dialog has been used) |
| `sub_cluster_clips/` | Per-bout `.mp4`s and `clip_manifest.csv` (when clip export ran) |

## Best practices

- Use pose outputs, not detection-only outputs.
- Run Bout Analytics first when you want reviewed bouts or ROI-grounded summaries before discovery.
- Run the **Stability audit** before naming or exporting clips - it's the cheapest way to spot a partition that's seed-sensitive.
- Name only the sub-clusters with a Likely real verdict and clear motion in the triptych grid; leave the rest unnamed and they'll fall back to `<class>__subcluster_N` filenames in clip export.
- Re-running with different clustering parameters generates a new `run_id`; old `state_names.json` won't pollute the new run's clip filenames.

## Troubleshooting

- **No bouts found.** Lower `Min Bout Duration`, raise `Max Frame Gap`, or import a Tab 6 / batch manifest from a pose run.
- **Most frames are noise.** Lower `HDBSCAN Min Cluster Size`. Some noise is expected and healthy; aim for a partition where the majority of frames land in named sub-clusters.
- **Sub-clusters keep shifting between runs.** Run the **Stability audit**. If mean ARI is below 0.5, treat the partition as soft and consider raising `Min Cluster Size` or reducing UMAP components.
- **Clip export reports `skipped`.** Check the manifest's `reason` column - common causes are missing source video for a directory or the bout's class landing in noise.
- **Old names appear on a new run.** Names are stamped with a `run_id`; if the file's `run_id` doesn't match the current run, the loader skips it. If you want to reuse names across runs intentionally, name through the naming dialog after the new run finishes.
