# Plugin Zone Counter

## Overview
- Adds an interactive viewer for counting detections entering a user-defined polygonal zone during inference.
- Hooks into the overlay pipeline so counts render on top of live frames while the dialog stays in sync.

## Entry Points
- Module: `integra_pose.plugins.plugin_zone_counter.plugin`
- Registration: function `register_plugin` instantiates `IntegraPosePlugin`
- Overlay key: `zone_counter` (registered with the active runner when available)

## Dependencies
- Required: `numpy`, `supervision` (polygon utilities), `opencv-python` (through overlay pipeline)
- Optional: none beyond the IntegraPose runtime dependencies

## Configuration
- Uses ROI drawing helpers from `integra_pose.utils.roi_drawing_tool` to capture polygons.
- Persists state only for the current session; clearing or redrawing a zone resets the counter.

## Usage
1. Launch **Plugins → Zone Counter** while an inference runner is active.
2. Click **Draw Zone** to sketch a polygon over the current frame, then confirm with Enter.
3. Monitor counts in the dialog; use **Reset Count** when you need to start over.

## Development Notes
- The plugin polls the active runner; keep overlay registration logic resilient to runners that lack support.
- Extend behaviour (for example, multiple zones) by expanding the overlay management helpers inside the plugin class.
