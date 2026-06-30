# Changelog

## 2.0.0-alpha.1

- Renamed the plugin surface to **TandemYTC - Tandem YOLO + Temporal Classifier** while keeping the existing package folder for compatibility during the refactor.
- Replaced the old config-file action menu with a single launcher for a Qt project workspace.
- Added a plugin-local Qt annotation workspace adapted from BehaviorScope-Y.
- Made full-video annotation export the primary dataset path.
- Removed user-facing legacy clip export controls from the workspace.
- Added project database export and resolved project config JSON export under the File menu.
- Added project-persistent behavior definitions and editable hotkeys through the Qt workspace.
- Added initial temporal-head choices in the training UI: `tcn`, `gated_attention_lstm`, `attention_lstm`, `lstm`, and `tcn_attention`.
- Ported the full-video NPZ preparation, training, inference, model, crop, annotation, and utility backend into the plugin-local Qt package.
- Added plugin-local social-geometry relation features from BehaviorScope-Y for multi-animal experiments.
- Implemented concrete TCN, TCN+attention, attention LSTM, gated-attention LSTM, and LSTM temporal-head branches in the YOLO-backed multi-animal classifier.
- Renamed the full-video training manifest schema to `yolo-temporal-full-video-v1`.
- Removed the unreferenced legacy `tandem_yolo_classifier` implementation folder and dead class-folder clip preparation panel.
- Added training tooltips plus a temporal-head pre-flight check that summarizes manifest context, selected head capacity, hidden width, layers, and attention settings before launch.
- Added optional annotation-review skeleton overlays backed by per-video YOLO-pose caches.
- Rewrote the plugin README and docs page as user-facing TandemYTC guides.
- Added a MARS seeding utility that imports MARS `.annot` files as approved GUI annotations and writes a ready-to-train full-video export manifest.
- Spruced up the inference path with GUI real-time mode, camera/stream source guidance, live telemetry for FPS/windows/sec/latency/resource use, metrics cadence controls, and explicit non-PyTorch install dependencies for `pip install ".[dev]"`.
- Tightened annotation review synchronization by polling playback position during video playback, updating only the playhead cursor between full timeline redraws, opening project databases in the current window, and exporting BENTO annotation times from stored frame boundaries.
- Added slow-motion annotation playback rates at `0.1x`, `0.25x`, and `0.5x`, with bindable hotkey actions for each rate.
- Moved the default approve shortcut from plain `A` to `Ctrl+Enter` so behavior labels such as Attack can use `A`; behavior hotkeys now take precedence if a project contains duplicate bindings.
- Added an explicit `File -> Save project` action with default `Ctrl+S`, documented autosave/export semantics, and exposed training run artifacts, resume, progress tracking, and graceful-vs-forced stop behavior in the Qt training UI.
- Renamed the Prepare Full Video output field from generic `Manifest path` to `Training manifest output` and documented the input/output manifest handoff.
- Normalized Qt `file:///...` path values before building TandemYTC workflow commands so local files are passed to Python CLIs as standard filesystem paths.
- Added user-facing documentation of TandemYTC prepared storage contents, including pose arrays, reliability masks, social geometry, debug geometry, and train-time derived pose features.
- Added compact HDF5 full-video preparation storage as the default, with loader support for HDF5-backed pose/social manifest samples and automatic `.seq` to cached-MP4 conversion before YOLO processing.
- Removed TandemYTC's former image-crop feature path so training and inference now consume only YOLO-pose, reliability, tracking, and social-geometry tensors.
- Fixed HDF5 preparation finalization so completed videos no longer fail when the NPZ writer is absent.
- Allowed TandemYTC training to consume compatible full-video NPZ caches by validating the shared pose/social tensor contract and ignoring unrelated extra arrays.
- Restored live inference playback with a `--preview` backend flag and Qt **Live preview** control for annotated real-time behavior display.
- Clarified skeleton review cache behavior in the Qt annotation workspace and disabled the **Skeletons** overlay checkbox until a cache is available for the selected video.
- Expanded the TandemYTC MkDocs guide with scientific feature-calculation details for pose-self features, reliability signals, pairwise social geometry, PCA body-axis estimation, and temporal inputs.
- Removed obsolete plugin-local image-feature modules and Qt controls so the TandemYTC GUI now exposes a lean pose/social temporal-classifier workflow.
- Added real TandemYTC Qt GUI screenshots under MkDocs assets as maintained documentation images.
- Added fail-fast training-manifest validation so pre-flight, the Train tab, and CLI clearly reject `source_manifest.csv` and other non-training manifests before training starts.
- Added temporal-head comparison tables to the TandemYTC MkDocs page and plugin README.
