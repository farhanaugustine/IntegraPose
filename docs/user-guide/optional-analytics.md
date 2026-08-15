# Optional Analytics Reference

Optional analytics let you match the output to the behavioral question instead of producing every possible table for every experiment.

Choose an assay preset for a sensible starting point, then enable or disable individual metrics as needed. Full Preflight reports when a selected metric cannot run with the current videos, model, or annotations.

## Assay presets

| Preset | Typical focus |
| --- | --- |
| **Custom / mixed** | Select only the metrics needed for a custom experiment |
| **Open field** | Zone occupancy, locomotion, session dynamics, and quality checks |
| **Elevated plus maze** | Open/closed arm occupancy, entries, latency, and behavioral allocation |
| **T-maze** | Choice, latency, revisits, and arm switching |
| **Y-maze** | Arm preference, revisit structure, and temporal switching |
| **Barnes maze** | Search latency, visits, movement, and behavioral transitions |
| **Object preference / NOR** | Object visits, discrimination, first choice, and interaction structure |

A preset selects a starting collection of metrics. It does not prevent you from changing the individual selections.

## Choice and preference

| Analysis | What it answers | What it needs |
| --- | --- | --- |
| **Preference indices** | Does the animal spend more time with one zone or object than another? Which was chosen first? | At least two usable zones, objects, or both |
| **Latency metrics** | How long until the first zone entry, object interaction, exit, or revisit? | Qualified ROI or object visits |
| **Visit structure** | How many visits occurred, and how were visit durations distributed? | Qualified ROI or object dwell events |
| **Object transitions** | In what order did the animal visit objects? Did it alternate or revisit? | Object interaction enabled with at least two objects |
| **Normalized summaries** | What percentage of the session was spent in each zone or object, and what was the rate per minute? | ROI or object results and a valid video duration |

Typical outputs include zone/object preference tables, latency tables, visit summaries, object transition matrices, and normalized per-track summaries.

## Zones and mazes

| Analysis | What it answers | What it needs |
| --- | --- | --- |
| **Zone occupancy heatmap** | How did occupancy of each arena zone change over the session? | Arena ROIs and qualified ROI visits |
| **Zone context windows** | Which zone was occupied immediately before and after a behavior bout? | Arena ROIs and behavior bouts |

Use exclusive ROI results when each moment should belong to one primary zone. Use concurrent ROI results when nested zones should all receive occupancy.

## Behavior structure

| Analysis | What it answers | What it needs |
| --- | --- | --- |
| **Behavior transitions** | Which behaviors tend to follow one another? | A model with more than one behavior class |
| **Temporal trends** | Did behavior frequency or duration change across the session? | Behavior bouts and a valid frame rate |
| **Activity budgets** | How was observed time distributed among behaviors? | Behavior bouts |
| **Inter-bout intervals** | How much time passed between repeated bouts, and when did the first bout occur? | Repeated bouts of a behavior |
| **Event-aligned windows** | Which behavior labels occurred around ROI or object entry and exit events? | ROI or object events plus behavior labels |
| **Bout timeline export** | When did each behavior occur during the recording? | Behavior bouts |

Event-aligned windows summarize behavior labels around ROI or object events.
They do not perform event-aligned pose or movement modeling.

## Motion and quality checks

| Analysis | What it answers | What it needs |
| --- | --- | --- |
| **Movement summaries** | How fast and how far did each animal move? How did movement change across bouts? | Tracked positions; pose information adds richer measurements |
| **Detection quality** | Were detections, boxes, and pose points consistently available? | Detection or pose outputs |
| **Multi-animal proximity** | How close were animals to one another, and when did they co-occur? | Multiple tracked animals with stable IDs |

Pixel-based distances remain in pixels unless the recording has been calibrated outside IntegraPose.

## Common per-video outputs

Selected analyses create their own folders inside the video's analytics folder. Common files include:

- behavior transition matrices and per-track transition tables
- behavior duration and count tables by time bin
- activity budgets overall, per animal, and per ROI
- pairwise proximity and behavior-overlap tables
- ROI occupancy matrices
- bout-level kinematic summaries
- detection-quality summaries
- inter-bout interval tables
- ROI context windows
- bout timeline tables and optional Gantt figures
- zone and object preference summaries
- zone and object latency tables
- zone and object visit-structure tables
- object visit sequences and transition matrices
- event-aligned behavior-window tables
- normalized zone and object summaries

The exact filenames are listed in:

- the video's `run_manifest.json`
- `module_tables/module_file_index.csv`
- the `Module_File_Index` sheet in `batch_results.xlsx`

## Batch-level tables and figures

IntegraPose gathers compatible per-video tables into `module_tables/`. The consolidated tables retain the video's Group, Subject ID, and Time Point so they can be filtered or compared without manually combining files.

When figure export is enabled and the necessary data are available, IntegraPose may create:

- temporal trend plots
- activity-budget plots
- ROI occupancy heatmaps
- behavior-transition heatmaps
- preference and latency comparisons
- visit-structure summaries
- inter-bout interval summaries
- object-transition heatmaps
- event-aligned behavior heatmaps

Use `figures/figure_manifest.csv` to see which figures were actually produced.

## Choosing a manageable set

Start with the smallest set that directly answers the experimental question.

For example:

- An open-field study may need movement summaries, center/periphery dwell, latency, and an activity budget.
- A novel-object study may need object interactions, preference indices, first-interaction latency, visit structure, and object transitions.
- A multi-animal study may need behavior bouts, stable tracking, and proximity measures.

Adding unrelated metrics increases the number of comparisons and the amount of output that must be reviewed.

See [Batch Output Map](batch-output-map.md) for file locations and [Advanced Batch Statistics](advanced-batch-statistics.md) for group-level inference.
