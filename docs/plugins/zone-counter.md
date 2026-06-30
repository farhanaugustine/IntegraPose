# Zone Counter Plugin

!!! note "Plugin status - research in progress"
    The IntegraPose plugin ecosystem evolves with active research. Some plugins are stable, others are works in progress, and the set may change as research needs shift. Pin to a commit if you depend on a specific plugin for an in-flight project.

!!! note
    Zone Counter ships as an optional plugin. Enable it from **Plugins -> Manage Plugins...**, then launch it from **Plugins -> Zone Counter**.

The Zone Counter plugin overlays a user-defined polygon on live inference streams and tracks how many detections enter the region. It is useful for tallying arena visits, nose pokes, or dwell-like events in real time.

## Launching the plugin

1. Start a file or webcam inference session with overlays enabled.
2. Open **Plugins -> Zone Counter**.
3. The control window shows the current count and drawing controls.

## Drawing and editing the zone

- Click **Draw Zone** to open the ROI sketcher. Trace a polygon on the current frame and press **Enter** to commit it.
- The polygon is stored in normalized coordinates so it can be reused across sessions at different resolutions.
- Use **Reset Count** to clear the tally without redrawing the zone.

## How counting works

- The plugin samples active detections during the run and increments counts when detections intersect the polygon.
- The overlay shades the region on the preview so you can validate placement while streaming.
- If overlays are unavailable in the current run mode, the plugin shows a warning.

## Tips

- Pair Zone Counter with the ROI Analytics tab to compare live counts against post-hoc bout metrics.
- Repeat runs with different polygons to profile multiple arena regions.
- Review the main Log tab to audit zone redraws and counter resets.
