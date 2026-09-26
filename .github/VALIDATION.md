# Continuous validation

`.github/workflows/validation.yml` runs on every push, pull request, and manual
dispatch. It is intentionally split into a broad compatibility gate and a
focused scientific regression gate.

## Covered

The `package-compatibility` job runs on Python 3.10 and 3.11, the complete
supported range declared by `setup.cfg`. On both interpreters it:

- byte-compiles the package and tests;
- checks the Python classifiers, `Requires-Python` contract, and dependency
  declarations;
- builds both a wheel and source distribution;
- installs the wheel without resolving the optional runtime stack;
- imports the installed package in isolated mode; and
- verifies the installed version and all three console entry points.

The `scientific-core` job runs on Python 3.11 with the pinned dependencies in
`.github/requirements-scientific-core.txt`. It covers:

- safe dataset splitting and frame-transfer collision handling;
- operation, project-save, and batch cancellation/failure outcomes;
- reproducibility-bundle path, size, and transaction protections;
- the YOLO pose-label producer/parser contract used by downstream analytics;
- Behavior Clustering feature construction, confidence filtering, and frame-gap behavior;
- immutable detected-bout snapshots, append-only review decisions,
  authoritative reviewed outputs, and transactional review saves;
- bout construction, ROI membership, ROI entry/re-entry, occupancy, and the
  Tab 6 analytics service;
- batch preflight, resume, reviewed-bout export, and pipeline outcomes;
- per-subject statistical units and multiple-comparison behavior;
- gait/decision-dynamics scientific contracts; and
- Fura alignment, ratio, baseline, and validation contracts.

Matplotlib uses the non-interactive `Agg` backend. The job installs the desktop
`opencv-python` distribution because several analytics modules import `cv2`,
but it does not create a GUI window. It also fails before testing if Roboflow,
`opencv-python-headless`, Ultralytics, PyTorch, or torchvision appears in the
environment.

## Not covered

This workflow does not claim to validate:

- interactive Tk or Qt rendering, window lifecycle, or display-server behavior;
- real-video end-to-end inference, tracking, annotation rendering, or training;
- CPU performance or CUDA, ROCm, MPS, and multi-GPU execution;
- Ultralytics model loading or model-file compatibility;
- Roboflow, cloud APIs, downloads, or any network-backed workflow;
- every optional plugin or its full optional dependency stack;
- camera hardware, video codecs beyond synthetic unit-test fixtures, or OS-
  specific behavior outside the explicitly mocked/tested contracts; or
- Python versions earlier than 3.10 or Python 3.12 and newer, which are outside
  the declared support range.

The separate `docs.yml` workflow remains responsible for strict MkDocs builds
and GitHub Pages deployment.
