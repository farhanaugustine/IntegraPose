# Skeleton Editor

Design custom keypoint layouts and edge connections for Supervision overlays or downstream analytics. Launch the editor from the Pose Clustering tab (**Edit Skeleton**) or any workflow that exposes the skeleton controls.

## Load a reference frame

- The editor automatically tries to open the video selected in the analytics configuration. Click **Load Video** to choose a different file.
- Use the frame slider to pick a representative pose before adding keypoints; the status label displays the current frame index.

## Add and manage keypoints

- Left-click the preview canvas to place a keypoint. IntegraPose prompts for a descriptive name (letters, numbers, spaces, underscores, or hyphens).
- The **Keypoints** list mirrors the current setup. Select a row and press **Remove** to delete it; connected edges are removed automatically.
- Keypoint coordinates are stored in pixel space so that Supervision can scale them correctly during inference.

## Draw skeleton edges

- Click **Add Edge**, then select the two keypoints you want to join by left-clicking them on the canvas. Repeat to build the skeleton structure.
- Existing edges appear in the **Edges** list (`A -> B`). Highlight an entry and choose **Remove Edge** to delete it.
- Right-click the canvas to cancel an in-progress edge before selecting the second keypoint.

## Save and reuse

- Press **Save Skeleton** to export the configuration as JSON. The file contains the ordered keypoint names and the index pairs that define each edge.
- Saving automatically switches the inference overlay settings to use the new skeleton and stores the file path in the active project.
- Custom skeletons can be shared across projects; point the inference tab’s **Skeleton Source** at the saved JSON to reuse your definitions.

## Tips

- Use a frame that clearly shows every joint to minimise manual adjustments.
- Keep names consistent with your dataset YAML so that analytics and clustering views line up without additional remapping.
- If you need to tweak coordinates later, reopen the skeleton file, load the same video, and adjust the keypoints before saving again.
