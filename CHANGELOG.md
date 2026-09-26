# Changelog

## 3.1.0b0 — 2026-09-26

- Added Discovery Explorer with linked video, cluster views, feature plots,
  manual review, and saved analysis history.
- Improved Behavior Clustering project loading, analytics imports, input
  validation, and cluster assignment reporting.
- Added the Qt Assisted Pose Curation workspace.
- Expanded Dataset Augmentor Lab with target image counts, behavior selection,
  and stricter pose-label and export validation.
- Fixed EDA recording, workbook exports, and analysis reruns; refreshed analytics
  plot styling and expanded regression tests and continuous validation.

## 2026-09-22

- Refreshed batch, movement, and bout analytics plots with white panels, light
  gray gridlines, pastel fills, and consistent typography.
- Preserved plot types, heatmap scales, review-status colors, analysis calculations,
  filenames, and tabular exports. Batch export themes now restore caller settings.
- Added regression coverage for plot outputs, unchanged CSV/workbook
  contents, source measurements, figure cleanup, and plot-style isolation.

## 2026-09-18

- Added exact final-image targets and behavior-name selection to Dataset Augmentor Lab.
- Default augmentation to training while retaining explicit selection of other splits.
- Preserve unselected splits, background labels, pose YAML metadata, and recording provenance.
- Correct anatomical flip mapping and synthetic-occlusion visibility; render named keypoint previews.
- Seed transform randomness, reject malformed labels and output collisions, validate exports,
  and leave incomplete runs clearly marked.
- Added regression coverage for pose geometry, visibility, split/behavior selection,
  exact counts, reproducibility, source-group separation, and failed writes.

## 2026-09-10

- Fixed EDA synchronized video exports by retaining the configured encoder dimensions.
- Use the installed OpenPyXL dependency for EDA bout workbooks and report export
  failures instead of silently substituting an empty workbook.

- Fixed EDA hierarchical clustering when no flat assignments are requested,
  including dendrograms of average behavior profiles.
- Clear obsolete PCA coordinates and cluster assignments when rerunning EDA analyses.
- Keep the existing EDA window registered when one of its child widgets is destroyed.
- Clarified EDA input formats, video navigation, and interpretation limits.
- Marked Fura Imaging Lab as incomplete and under development in its guides,
  plugin catalog, and Plugin Manager description.

## 2026-09-06

### Discovery project loading and Qt startup

- Fixed Tab 7 Load Project to restore both main IntegraPose project files and
  older Tab 7-only files. Unrelated JSON files now produce an error instead
  of silently clearing the setup.
- Reported Discovery Explorer child-process failures in the main GUI, with
  the environment and startup log needed for troubleshooting.
- Added a Windows Qt startup safeguard for a Conda ICU DLL conflict affecting
  Discovery Explorer and Bout Reviewer, without changing system search paths.
- Added regression checks for project formats, launch errors and GUI buttons.

## 2026-09-05

### Initial Discovery Explorer

- Added a separate Qt workspace for linked source-video playback, layered
  timelines, 2D/3D cluster views, feature comparisons and manual review.
- Retained clustering coordinates and a keypoint-derived feature catalog.
  Display-only projections are labeled separately from clustering inputs.
- Added reversible interval/point annotation, scoped merges, managed run
  history, background feature-reuse reruns and explicit assignment exports.
- Integrated the embedded Tab 7 setup with main project Save/Open. Save As
  copies the analysis workspace while retaining source-video references.
- Added feature/time/space plots with explicit observed-time denominators,
  graph configurations and underlying-data exports.
- Documented the initial implementation's limits, including cohort-level and
  event-aligned views, subset reclustering and portable-project packaging.

### Tab 7 clustering and documentation

- Fixed Tab 6 and completed-batch handoffs to Tab 7 for analytics manifest
  schema 4. The writer and reader now share a version contract; schemas 1–4
  are supported, while malformed and unknown versions remain rejected.
- Added regression tests for both handoff callbacks, partial batch imports,
  group preservation, missing inputs, and manifest-version validation.
- Kept UMAP/HDBSCAN clustering on CPU and removed the experimental GPU
  adapter and selector. No new GPU dependencies are required.
- Separated minimum samples per class from minimum bout duration and stopped
  silently reducing the requested HDBSCAN minimum cluster size.
- Preserved detection-row identity with duplicate DataFrame indices, rejected
  invalid feature/class inputs, and distinguished skipped samples from noise
  in assignments, summaries, and bout exports.
- Saved CPU execution, dimensionality-reduction, seed, and package-version
  information with clustering diagnostics.
- Fixed namespaced cluster labels in candidate scoring and naming.
- Replaced obsolete clustering GUI titles, help, dependency checks, and baseline-group
  prompts with the current Behavior Clustering workflow.
- Updated the Tab 7 guide, quick start, and landing-page descriptions to explain
  current controls, outputs, and interpretation limits.
- Made the landing-page IntegraPose heading fit its content column without
  splitting the name across lines.

## 2026-08-06

### Flexible Bout Review Workspace

- Made the existing video/review and workspace/timeline dividers wider and
  fully adjustable.
- Added **View** controls for hiding and restoring the video, review/scoring,
  and timeline panels, together with **Reset layout** (`Ctrl+Shift+0`).
- Reduced rigid window and video minimums and fitted the initial reviewer size
  to the available screen. Review and scoring pages now scroll when space is
  limited, while video frames retain their aspect ratio.
- Made review and scoring table columns resizable and reorderable.
- Saved window geometry, splitter positions, panel visibility, and table-column
  layouts between sessions, with automatic recovery from disconnected or
  repositioned displays.
- Removed the timeline's former height ceiling and added regression coverage
  for resizing, persistence, reset, and off-screen recovery.

## 2026-08-05

### Single-animal dashboard and review readiness

- Kept the Bout Analytics **Single Animal Analysis** setting fixed for the
  entire run, including annotated-video rendering.
- Aligned dashboard detections with the canonical Track 0 used by
  single-animal bout, ROI, and object analytics, so behavior names and bout
  counts appear in the validation video.
- Changed dashboard export to publish the final MP4 only after the container
  is closed and can be reopened successfully. In-progress renders no longer
  appear under the final `_annotated.mp4` filename.
- Prevented the main GUI from opening the Bout Review Workspace while Bout
  Analytics is still rendering its video.
- Added regression coverage for tracked single-animal labels, behavior cards,
  failed-render cleanup, and reviewer launch timing.

## 2026-08-04

### Documentation and release freeze

- Harmonized the README and user guide with the controls, defaults, outputs,
  safeguards, and handoffs in the shipped `3.0.1b2` UI.
- Documented all six frame-extraction modes, the current folder-flattening and
  setup QA workflows, inference save defaults, webcam ROI controls, and the
  pose-only scope of the Model Training tab.
- Distinguished the recommended full `.[plugins]` install from the minimal
  core install, and documented the five-stage Sanity Check.
- Added the bundled Fura Imaging Lab to the plugin catalog and clarified that
  specialist plugin outputs are not replaced by a core project bundle.

## 2026-07-29

### Integrated bout review

- Added one video-synchronized review workspace for Class ID behavior bouts,
  concurrent ROI visits, exclusive ROI-X visits, and object interactions.
- Added frame-accurate boundary correction, behavior reclassification, track-ID
  correction, rejection, manual addition, splitting, merging, and explicit
  acknowledgement of legitimate same-track behavior overlaps.
- Added separate mutually exclusive and multi-label behavior-bout construction
  so experiments can either retain one behavior state or preserve meaningful
  concurrent classes.
- Added per-track behavior completion and separate ROI, ROI-X, and
  object-interaction completion states. Incomplete work remains provisional
  and cannot silently replace automatic results.
- Added default tIoU 0.50 model-review scoring, an optional 0.25/0.50/0.75/0.95
  sweep, frame agreement, boundary error, and per-behavior correction metrics.
- Organized review tables and figures into `Behavior_Bouts`, `ROI_Bouts`,
  `Object_Interactions`, and `Shared_Audit` folders.
- Added run-relative review paths and annotated-video, source-video, and manual
  source-folder fallbacks for work moved between computers or external drives.
- Connected both Tab 6 and the Batch Processing Wizard to the same
  IntegraPose-themed reviewer while retaining legacy tools as fallbacks.

### Documentation

- Added a researcher-facing Bout Review Workspace guide with multi-animal,
  multi-label, overlap, completion, tIoU, portability, and output guidance.
- Updated the Bout Analytics, Batch Wizard, output map, metrics reference,
  model workflows, Quick Start, and reproducibility guidance for the current
  review workflow.
- Clarified the difference between authoritative prediction review and the
  separate Manual Bout Scorer sidecar.

## 2026-07-23

### Batch Processing Wizard

- Reorganized the wizard into a clearer step-by-step workflow.
- Added bulk and per-video editing for Group, Subject ID, and Time Point.
- Added conservative metadata discovery from filenames and cohort folders. Existing values are preserved, and ambiguous labels are left for manual review.
- Added guided arena ROI and object-placement queues across included videos.
- Added a visible orange dotted boundary for the object-interaction distance measured from the object ROI edge.
- Clarified that regular arena ROIs support bounding-box or selected-keypoint entry, while object interaction requires pose keypoints.

### Bouts, spatial visits, and preflight

- Behavior bouts no longer bridge across an explicitly observed different behavior.
- Saved bout tables report observed and bridged frame support together with the applied duration and gap settings.
- ROI and object visits use their own minimum dwell and maximum gap controls.
- Full Preflight now reports missing or ambiguous study-design labels, incompatible metric selections, and the analyses affected by each issue.

### Statistics and documentation

- Moved optional statistical controls into a collapsed Advanced Statistics section.
- Added automatic study-design role assignment for Group, Subject ID, and Time Point.
- Separated repeated-subject mixed-effects analysis from the optional KPSS stationarity diagnostic.
- Added end-user guides for batch outputs, optional analytics, advanced statistics, metric units, and bout construction.

## 2026-07-10

### Inference output integrity

- Made `max_det` a verified postcondition before file, batch, and webcam rendering, ROI callbacks, CSV/TXT export, and crop generation; excess backend rows are reduced by confidence and malformed over-returns fail instead of being selected arbitrarily.
- Added a session-owned batch `Max detections/frame` setting and recorded requested versus effective values in inference metadata; Single Animal Analysis enforces an effective cap of one.
- Standardized saved frame labels as zero-based `{source}_frame_000000.txt` files with a manifest, including empty files for frames without detections, and prevented crop and run-directory overwrites.
- Migrated Bout Analytics, validation video, Behavior Clustering, gait, and EDA readers to the shared frame resolver; auxiliary TXT files are ignored, source boundaries are enforced, and duplicate frame aliases now stop analysis.
- Added strict TXT/CSV reconciliation and pose-schema validation so malformed, ambiguous, or mismatched scientific inputs cannot be silently merged.
- Blocked advanced multi-video-folder inference from combining timelines, annotated media, or persistent tracker state; those folders now route users to the Batch Processing Wizard.
- Reset TandemYTC tracking at every source-video boundary, enforced its detection cap before crop consumers, and rejected duplicate sources, video IDs, and sanitized artifact keys before inference.
- Added SHA-256 validation for TandemYTC sources, annotations, converted videos, checkpoints, and result artifacts; stale resume/conversion caches now rebuild through validated atomic MP4 replacement.
- Isolated AutoLabel Forge frames and datasets by run, added collision-resistant source identities and extraction/label manifests, and made unreadable inputs or failed frame writes stop before labeling.
- Made assisted-curation audit images collision-resistant, blocked malformed annotations from assist/save without rewriting them, and fixed active-learning scoring to a recorded `max_det=1` contract.

### Legacy BehaviorScope plugin removal

- Removed the bundled legacy BehaviorScope plugin now that it is maintained as a separate project.
- Removed its implicit-trust and installation references while retaining TandemYTC artifact-compatibility identifiers and historical changelog provenance.
- Generalized active Behavior Clustering clip-export wording for downstream classifier workflows.

### Scientific analytics correctness

- Standardized YOLO pose labels across inference, Tab 7, and gait ingestion with an explicit schema sidecar, fixed suffix ordering, and shared 2D/3D parsing.
- Corrected ROI hysteresis across tracking gaps, disabled-ROI filtering, nested-ROI primary selection, inclusive entry/exit intervals, per-track latency pairing, and object-interaction frame-delta rates.
- Kept Tab 6 and batch ROI/object debounce parameters identical, including explicit zero-valued thresholds and per-video FPS resolution.
- Corrected Behavior Clustering confidence filtering, body-normalized social geometry, finite off-diagonal distances, frame-delta velocity/acceleration, and temporal resets across long gaps.
- Changed batch group tests to use subject-level means instead of repeated-video pseudoreplication; repeated factors now route to mixed models, whose fixed effects and adjusted p-values are exported.
- Corrected gait FPS units, dropped-frame derivatives, stride boundaries, contiguous transition/CCM windows, and per-video group weighting.
- Corrected Fura channel timestamp pairing, invalid ratio denominators, background-subtraction labels, baseline normalization, drift validation, and baseline interpolation.

### Bout review and batch reliability

- Added an ethogram-context bout reviewer with neighboring and overlapping behavior lanes, stable bout IDs, inclusive frame boundaries, and correct/incorrect/relabel decisions.
- Preserved detected bouts as immutable source data, stored append-only review decisions, and materialized reviewed bouts and summaries only after a complete review.
- Made authoritative reviewed bouts the preferred Tab 6 and batch aggregation source while recording raw/reviewed provenance and invalidating stale raw-derived module outputs.
- Rehydrated completed-video artifacts during batch resume so rebuilt cohort workbooks include prior completed videos; missing artifacts now trigger reprocessing.
- Added explicit success, partial, failed, and cancelled batch outcomes with completed/failed counts and runtime-resolved device provenance.

### Safety, runtime, and installation

- Added transactional dataset splitting, source/output alias rejection, and exclusive frame-transfer writes to prevent source deletion and late collision overwrites.
- Stopped cancelled saves, failed subprocesses, failed inference, and partial batches from reporting success.
- Hardened reproducibility bundle import against path traversal, links, duplicate paths, undeclared files, checksum/size mismatches, and decompression bombs; imports now stage into an isolated local directory.
- Redacted machine-local paths, source URLs, host identifiers, and registry paths from exported bundles while retaining scientific parameters and checksums.
- Blocked unsafe TandemYTC pickle fallback unless the user explicitly opts in for a trusted legacy checkpoint.
- Standardized `-1` accelerator auto-selection across training, file inference, batch, and webcam for CUDA/ROCm, MPS, and CPU systems.
- Updated the supported Python range to 3.10-3.11 and removed the unused Roboflow SDK dependency that could install headless OpenCV into the desktop environment.
- Declared the actual AutoLabel Forge GroundingDINO/autodistill dependencies in unified, plugin, and package-extra install paths.
- Added Python 3.10/3.11 package and scientific-core CI gates, including authoritative bout-review contracts.
- Made UMAP explicitly optional with `UMAP Neighbors = 0`, validated clustering parameter bounds, and removed first-run JIT stalls from the test and sanity-check paths.
- Made core logic and Zone Counter explicit wheel packages and added isolated wheel-import validation.
- Kept TandemYTC pose-export class IDs synchronized with their current class label instead of relying on a conditionally defined prediction variable.

## 2026-07-08

### Tab 1 frame-transfer safety overhaul

- Reworked frame flattening so Tab 1 can preview, copy, move, or dry-run transfers safely.
- Added optional short image names using the format `IMG####_F###_######.ext` to reduce Windows path-length issues while keeping source-folder grouping.
- Changed the UI default to copy files and shorten image names, preserving original image folders by default.
- Added automatic `frame_transfer_manifest.csv` and `frame_transfer_summary.json` outputs for traceability and reproducibility.
- Added collision handling, hidden-file/folder skipping, destination-in-source skipping, path-length warnings, and preview examples.
- Added fail-fast blocking for unsafe output paths before copy/move starts, preventing slow or hanging Windows long-path file operations.
- Added destination-root length warnings because a very deep output folder can still make copy operations slow even when shortened filenames are used.
- Switched frame-transfer image copying from metadata-preserving `copy2` to faster `copyfile`; image provenance is preserved in the manifest instead.
- Made frame-transfer preview run in a background thread and added clearer status updates during planning and transfer.
- Made blocked/error/warning frame-transfer messages persist in the Tab 1 status label and increased toast duration for messages users need time to read.
- Made dry-run frame-transfer runs explicitly report that no image files were copied or moved, and changed transfer progress status to concise frame counts instead of long paths.
- Restored/expanded Tab 1 tooltips across frame extraction, crop, and transfer controls; renamed the transfer handoff button to clarify that it sends the destination folder to the Setup tab.
- Added focused tests for short-name planning, copy/move behavior, dry-run manifest output, collision renaming, and long-path warnings.

### Tab 1 frame extraction upgrades

- Kept the existing `stride`, `random`, and `interactive` extraction modes and added `time_balanced`, `motion_rich`, and `hybrid` modes.
- Added `frame_extraction_manifest.csv` output for extracted frames, including source video, frame index, timestamp, mode, score/reason, and output path length.
- Added extraction path preflight checks so unsafe Windows output paths are blocked before frame writing begins.
- Added clearer extraction progress/status messages for saved frames, motion scanning, path warnings, and completion.
- Updated crop UI/log wording from generic CUDA language to NVIDIA encoder wording for `h264_nvenc`; crop behavior remains unchanged.

### GPU backend autodetection and ROCm/CUDA centralization

- Expanded `integra_pose/utils/torch_backend.py` to centralize:
  - backend detection: CPU, NVIDIA CUDA, AMD ROCm, and MPS
  - AMP support and dtype resolution
  - pinned-memory and non-blocking transfer flags
  - CUDA-compatible seeding
  - cuDNN/MIOpen benchmark enabling
  - ROCm reporting via `torch.version.hip`
- Wired remaining train/infer internals:
  - TandemYTC `train_y.py`: AMP, GradScaler, autocast device type, pinned memory, non-blocking transfers
  - TandemYTC `infer_y.py`: `auto` default, AMP dtype, cuDNN benchmark, autocast device type
  - TandemYTC metrics logger: backend-aware PyTorch GPU memory checks
  - BehaviorScope training: pinned memory, non-blocking transfers, CUDA-compatible seeding
  - project seed utility and sanity check report now use the backend utility
- Added focused backend tests covering CUDA, ROCm, CPU fallback, Ultralytics `-1`, AMP dtype resolution, benchmark enabling, and CUDA-compatible seeding.

### README GPU installation guidance

- Added a concise README section for choosing the correct PyTorch build:
  - CPU fallback install
  - NVIDIA CUDA PyTorch install path
  - AMD ROCm PyTorch install path
  - AMD ROCm Docker container option for Ubuntu/Linux users
  - one verification command for CPU, CUDA, and ROCm
  - note that AMD ROCm PyTorch still uses `torch.cuda` and `cuda:0` style device strings internally
- Moved the fresh Conda install recipe near the top of the README install section so new users see the recommended order before optional details.
- Reordered README documentation guidance so quick doc links remain near the top, while MkDocs build instructions appear after install and launch.
