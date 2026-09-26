# EDA Tool Plugin

The EDA Tool is an optional workspace for exploring geometric pose features,
comparing labeled behavior profiles, and inspecting clusters alongside a matching
video. Enable it from **Plugins → Manage Plugins...**, then select
**Plugins → Launch EDA Tool**.

## Requirements and inputs

- Install the complete user profile (`pip install ".[dev,plugins]"`).
- Provide frame-indexed YOLO pose-label files from one recording, such as
  `trial_frame_000000.txt`, and the ordered keypoint names used during inference.
- An optional `data.yaml` supplies keypoint configuration and class names.
- For video inspection, select the original recording matching those frame indices.

The loader expects whitespace-separated YOLO pose rows. A `.csv` filename alone
does not make a general CSV table, bout summary, or exported feature table a
compatible input. Select the inference-label folder explicitly; the tool does not
automatically import the current IntegraPose project.

## Workflow

1. In **Load Data & Config**, select the labels and optional YAML file. Check
   keypoint order, visibility threshold, and coordinate normalization, then load
   and preprocess the data.
2. In **Feature Engineering**, define skeleton connections or enable all geometric
   features. Select and calculate the distances, angles, and other features needed.
3. In **Analysis & Clustering**, select features and choose individual detections
   or average behavior profiles. Run optional PCA, followed by hierarchical
   clustering (AHC) or KMeans.
4. For AHC on individual detections, a flat-cluster count of zero produces only
   a dendrogram. Set a positive count to request assignments. AHC on average
   behavior profiles produces a dendrogram of the behavior means.
5. Review plots and status text in **Visualizations & Output**. In **Video & Cluster
   Sync**, load the matching video and use playback or the frame slider to highlight
   the corresponding observations on the feature map.
6. Use **Save Plot** to export a figure and **Export Data → Export All** to save
   data tables, cluster assignments when available, and supporting metadata.

## Interpretation and limits

- PCA and clustering use standardized selected features. Rows missing any selected
  feature are excluded from the fit; their assignments remain missing in exports.
- Clusters describe similarity in those features. They are not automatically
  validated behaviors, phenotypes, or independent experimental replicates.
- Class names come from the label configuration. If the pose model labels animals
  rather than behaviors, its classes must not be interpreted as behavior categories.
- The video map follows playback and slider position; scatter-point clicking does
  not seek the video. Use one recording at a time to keep frame identities unambiguous.
- **Behavioral Analytics** provides descriptive summaries and a separate advanced
  bout-analysis workflow. The basic switch count is global across ordered detections,
  not an animal-specific transition estimate.

Keep the input labels, configuration, selected feature list, and exported results
with the study record so the analysis can be reproduced.
