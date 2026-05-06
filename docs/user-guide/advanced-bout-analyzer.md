# Advanced Bout Scorer

Use the Advanced Bout Scorer when you need frame-accurate verification or manual annotation of behavioral events. The window opens from **Bout Analytics → Open Advanced Bout Scorer** or by double-clicking a bout in the analytics table.

## Workspace overview

- **Video preview** (left) – Scrub through the source video to confirm each event. Playback respects the project’s frame rate and resizes automatically to fit the window.
- **Transport controls** – Play/pause, drag the timeline slider, or jump by single frames. The status label displays both frame counts and elapsed seconds.
- **Scoring panel** (top right) – Choose the animal (track ID), pick the behavior label, and set start/end frames for the current bout.
- **Scored bouts table** (bottom right) – Accumulates every manual entry with duration in seconds for quick auditing before export.

## Annotating a bout

1. Select the track ID and behavior from the drop-down menus. IntegraPose seeds the track list from the analytics results.
2. Navigate to the desired start frame, then click **Set Start Frame**. Repeat for **Set End Frame**.
3. Click **Log Bout** to add the event to the table. Duration is calculated automatically from the configured FPS.
4. Use **Clear** to reset the start/end fields if you make a mistake.

## Keyboard & playback tips

- Pause playback before dragging the timeline for precise positioning.
- The video scale control mirrors the current frame number; you can drag it while holding the left mouse button for coarse navigation.
- If you toggle play while already viewing the final frame, the player stops automatically to avoid wrap-around.

## Exporting results

- Click **Export CSV** to write the scored bouts to disk. Files are named `manually_scored_<video>.csv` by default and include track ID, behavior, start frame, end frame, and duration.
- The export routine uses UTF-8 encoding and can be reopened directly in the Bout Analytics tab or external statistics packages.

## Error handling

- If the requested video cannot be opened, the scorer shows a descriptive error and keeps the window responsive so you can choose another file.
- Frame inputs are validated to ensure the start frame precedes the end frame.
- All fatal exceptions are logged to the main IntegraPose console for later diagnosis.
