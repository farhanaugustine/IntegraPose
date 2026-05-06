# Third-Party Notices (Summary)

To keep the repository within GitHub size limits, the full license bundle is **not** tracked in git. Publish the complete output as a release asset (e.g., `third-party-notices-full.md` or a zipped variant) and keep this summary in the tree.

## Regenerate the notices

Run these commands from a clean environment that has the same dependencies you ship (install any extras such as `plugins` if you redistribute them):

```bash
pip install -e ".[dev]"
python tools/generate_third_party_notices.py --output THIRD_PARTY_NOTICES.md              # summary
python tools/generate_third_party_notices.py --full third-party-notices-full.md          # full texts for releases
```

Upload the `--full` output with your release and record its SHA256 checksum in the release notes or an accompanying `CHECKSUMS.txt`.

## Key licenses to be aware of

| Component group | Upstream license(s) | Notes |
| --- | --- | --- |
| Ultralytics YOLO | AGPL-3.0-only | All training/inference flows rely on Ultralytics; commercial users need an Ultralytics enterprise license to avoid AGPL obligations. |
| torch / torchvision | BSD-3-Clause | Install the hardware-appropriate wheels from pytorch.org; the GUI does not pin CUDA/CPU variants. |
| Supervision | MIT | Used for overlay rendering and analytics. |
| OpenCV-Python | Apache-2.0 | Video I/O, frame extraction, and ROI tools. |
| NumPy / pandas / SciPy / scikit-learn / UMAP / HDBSCAN | BSD-3-Clause | Core analytics and clustering stack. |
| Pillow | HPND (Pillow) | Imaging utilities and the annotator. |
| Plotly / Seaborn / Matplotlib | MIT / BSD-3-Clause | Visualization layers across analytics tabs. |
| tqdm | MPL-2.0 | Progress indicators for long-running jobs. |
| sv-ttk | MIT | Optional modern ttk theming. |
| Plugin-only extras (pyEDM, hmmlearn stack, etc.) | Project-specific (mostly Apache-2.0/OSL/BSD; check full bundle) | Only required when enabling optional plugins; include their licenses when you redistribute them. |

The `tools/generate_third_party_notices.py` script is the source of truth. Always regenerate after changing dependencies or publishing a new release.
