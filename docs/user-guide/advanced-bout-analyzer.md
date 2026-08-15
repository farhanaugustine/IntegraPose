# Manual Bout Scorer

Use **Manual Bout Scorer** when you want to watch a video and create or edit a
separate table of behavior events. It is useful for exploratory scoring,
behaviors that are not represented by the current model, or a fully manual
record that you plan to analyze separately.

If your goal is to check IntegraPose predictions, correct them, and use the
reviewed results in later IntegraPose summaries, use the
[Bout Review Workspace](bout-confirmation.md) instead.

## Open the scorer

From Bout Analytics, choose **Manual Bout Scorer...**. The scorer opens the
current source video and, when available, loads the current behavior bouts,
animal IDs, and behavior names. Double-clicking a compatible row in the Bout
Analytics table opens the same scorer at that event.

## Workspace overview

- **Video preview** - watch the source video and move to the event of interest.
- **Playback controls** - play, pause, drag the timeline, or move one frame at
  a time with **Prev Frame** and **Next Frame**.
- **Scoring panel** - choose the animal and behavior, set the first and last
  frame, and add reviewer notes.
- **Editable Bouts** - inspect, revisit, update, or delete saved events.

## Add a manual bout

1. Select the **Animal ID** and **Behavior**.
2. Move to the first frame of the event and choose **Set Current** beside
   **Start Frame**.
3. Move to the last frame and choose **Set Current** beside **End Frame**.
4. Add a short note if it will help explain the scoring decision later.
5. Choose **Add New Bout**.
6. Check the new row and calculated duration in **Editable Bouts**.

The start and end frames are inclusive. For example, frames 10 through 14
contain five frames. Duration in seconds is calculated from the video frame
rate used by Bout Analytics.

Use **Clear Form** to discard an unfinished entry. The **Jump** buttons return
the video to the start or end frame currently entered in the form.

## Edit existing bouts

1. Select a row in **Editable Bouts**. Double-click it to jump to the event.
2. Change the animal, behavior, boundaries, or reviewer notes.
3. Choose **Update Selected**.

Use **Previous Bout** and **Next Bout** to move through the table. Use **Delete
Selected** only when the event should be removed from this separate manual
record.

## Save and export

**Save Progress** saves through the Bout Analytics workflow when a save
location is available. Progress is also saved when you add, update, or delete a
bout and when the window closes.

Choose **Export CSV...** to save a copy wherever you choose. The suggested name
is normally:

```text
advanced_bout_scores_<video>.csv
```

The CSV includes the animal and behavior, inclusive start and end frames,
duration, reviewer notes, the source video, and fields that preserve the
original values when a loaded event was edited.

## Which review tool should I use?

| Goal | Use |
| --- | --- |
| Create an exploratory or fully manual event table | **Manual Bout Scorer** |
| Confirm, reject, relabel, split, merge, or adjust predicted bouts | **Bout Review Workspace** |
| Review ROI visits or object interactions | **Bout Review Workspace** |
| Measure prediction-to-review agreement and feed reviewed results into batch summaries | **Bout Review Workspace** |

The Manual Bout Scorer CSV remains a separate manual record. It does not mark
the integrated behavior, ROI, or object review as complete and does not replace
the preferred Bout Analytics results.
