# Citations and Acknowledgements

## Citing IntegraPose

If IntegraPose contributes to your analysis pipeline, include the project citation in your methods section, software statement, or acknowledgements.

> Augustine, F., O'Sullivan, S., Murray, V., Ogura, T., Lin, W., & Singer, H. S. (2025). *IntegraPose: A unified framework for simultaneous pose estimation and behavior classification*. Neuroscience, 590, 1-22. https://doi.org/10.1016/j.neuroscience.2025.10.020

## Recommended Citation Practice

Most manuscripts are clearer when they cite three layers of the workflow:

1. IntegraPose itself
2. the model family or training framework used for inference
3. any major downstream analysis framework that materially shaped the results

That makes the computational pipeline easier for collaborators and reviewers to follow.

## Open Source Foundations

IntegraPose depends on a broader open-source ecosystem. The project benefits directly from tools that make modern computer vision, numerical analysis, visualization, and model training practical in research settings.

| Project | Why it matters here |
| --- | --- |
| PyTorch | Supports deep-learning execution and GPU-backed model workflows |
| Ultralytics YOLO | Provides the backbone for core detection and pose training and inference flows |
| OpenCV | Handles important video IO, image processing, and overlay operations |
| NumPy | Powers array-based numerical computation throughout the pipeline |
| SciPy | Supports scientific and signal-processing style analysis utilities |
| Pandas | Drives table-oriented results handling and export workflows |
| Matplotlib | Supports plotting and report-style visualization |
| Pillow | Supports image loading, conversion, and GUI-friendly image handling |
| Supervision | Provides useful computer-vision workflow helpers and overlay utilities |

IntegraPose also benefits from the broader open-source research software community that continues to improve model tooling, scientific Python infrastructure, and reproducible analytics workflows.

## Acknowledgement

If you want a short acknowledgement sentence for a manuscript, thesis, or poster, this is a reasonable starting point:

> This work used IntegraPose, a workflow that builds on open-source computer-vision and scientific Python projects including PyTorch, Ultralytics YOLO, OpenCV, NumPy, SciPy, Pandas, Matplotlib, Pillow, and Supervision.
