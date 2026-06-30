# EDA Tool Plugin

!!! note "Plugin status - research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

!!! note
    The EDA Tool ships as an optional plugin. Enable it from **Plugins -> Manage Plugins...**, then launch it from **Plugins -> Launch EDA Tool**.

The EDA Tool helps you explore pose and bout outputs before final analysis.

## Requirements

- Install optional analytics dependencies (`pip install ".[plugins]"`).
- Use a project with completed inference and/or bout exports, or provide a compatible CSV file.

## Key features

- **Data ingestion**: Load outputs from the current project or import external CSV files.
- **Interactive PCA and clustering**: Tune feature sets, run PCA + KMeans, and inspect embeddings with configurable coloring.
- **Video synchronization**: Click scatter points to jump the preview to matching frames.
- **Export utilities**: Save updated cluster labels and high-resolution plots.

## Workflow

1. Launch the plugin and choose current project data or a CSV input.
2. Filter by track, behavior, or detection scope, then curate the feature list.
3. Run dimensionality reduction and clustering; iterate until groups are meaningful.
4. Export labels and plots for downstream analysis.

## Tips

- Keep the Log tab visible while running PCA/clustering to monitor status and warnings.
- Save intermediate outputs so comparisons between parameter settings are easy to reproduce.
