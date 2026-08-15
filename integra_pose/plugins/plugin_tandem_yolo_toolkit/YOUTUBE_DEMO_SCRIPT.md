# Why clip-based model training isn't the best for action classification

**Format:** YouTube plugin demo  
**Target runtime:** about 9½ minutes at a natural speaking pace  
**Delivery:** direct, screen-led explanation from someone familiar with the workflow  
**Plugin shown:** TandemYTC — Tandem YOLO + Temporal Classifier

## 0:00–0:45 — The actual problem

**On screen:** Play a short attack clip. Then reveal where that clip came from on the full-video timeline, with investigation or approach before it and disengagement afterward.

**Narration:**

If I train on this three-second attack clip, I have already removed one of the hardest parts of the problem.

The classifier no longer has to determine when investigation becomes attack, distinguish an approach that never develops into attack, or locate the end of the bout. The clip edit made those decisions.

At inference, the model is not handed `attack.mp4`. It receives a continuous recording containing the behavior, its transitions, and similar-looking non-events. TandemYTC’s full-video workflow is built around that mismatch.

## 0:45–1:45 — Why clip-only datasets can mislead

**On screen:** Compare a folder of class-named clips with one source-video timeline. Show two neighboring clips from the same source being incorrectly placed in different dataset splits.

**Narration:**

Clips are useful for collecting rare examples or testing whether a behavior has a detectable signal.

The problem is training only on each bout’s clean center. That removes much of the “other” class: ordinary locomotion, incomplete approaches, occlusion, detection failures, and transitions. Those frames are common sources of false positives and unstable boundaries in continuous video.

There is also leakage risk. If clips from the same trial, animal pair, or camera session enter both training and validation, scores can benefit from shared source context.

TandemYTC assigns complete source videos to a split, keeping all windows from one video together.

## 1:45–2:35 — What full-video annotation changes

**On screen:** Show approved Investigation, Attack, Mount, Grooming, and Locomotion spans on their timeline lanes. Leave unmarked regions visible.

**Narration:**

The source remains intact in the annotation view. I mark behavior bouts; frames outside approved spans remain background.

Preparation reads those labels across the full timeline. By default, it creates 32-frame windows every 16 frames. A window keeps its dominant class when the configured threshold is met; ties or windows below that threshold are skipped, and background windows can be subsampled.

Full video is not automatically superior. It requires more annotation, and poor labels remain poor labels. Its advantage is a training task that includes transitions, hard negatives, and uninterrupted context.

## 2:35–3:10 — What TandemYTC is

**On screen:** Open **Plugins → TandemYTC** and briefly show the five tabs: **Annotate**, **Prepare Full Video**, **Train**, **Inference**, and **Batch**.

**Narration:**

TandemYTC connects annotation, full-video preparation, temporal training, inference, and batch processing.

YOLO pose and tracking provide each frame’s observations. A TCN, LSTM, or attention-based head combines them across time.

The classifier uses pose and social geometry—not RGB appearance. Training and inference therefore require compatible pose checkpoints with the same keypoint layout.

## 3:10–4:35 — Annotation and project management

**On screen:** Open a project database, import a folder, and show the video queue. Open **Edit → Project metadata**, then **Manage behaviors**.

**Narration:**

The SQLite project database is the working record. It stores video metadata, behavior definitions, hotkeys, annotation spans, review state, and the last review position.

Project metadata can include the study, condition, annotator, reviewer, summary, and peer-review notes. Each behavior receives a label, color, hotkey, and an operational definition shared by reviewers.

**On screen:** Select Investigation, mark a span, zoom to its boundary, step frame by frame, resize it, relabel it by dragging to another lane, and enter a note.

**Narration:**

On the timeline, I can scan quickly, slow down for a boundary, step frame by frame, zoom, resize a span, or drag it to another behavior lane.

Each span can carry confidence, ambiguity, a position lock, and a note—for example, that a boundary is uncertain because one subject is occluded.

Draft, Ready, Approved, and Rejected states separate annotation from review. Only approved spans enter training export; rejected spans remain for traceability.

The project can also be exported as a portable database with a metadata manifest, or as a JSON configuration.

## 4:35–5:55 — The information used by the classifier

**On screen:** Enable **Skeletons**. Freeze on two tracked subjects and build the feature explanation progressively: `Pose`, `Reliability`, `Tracking`, and `Pairwise geometry`.

**Narration:**

The temporal head receives numerical features, not the rendered skeleton overlay.

Per-subject pose features come from normalized keypoints, centered coordinates, within-pose distances, and keypoint speeds.

Reliability features—presence and pose masks, pose and detection confidence, track confidence, and track age—describe whether those observations are stable. Tracking maintains subject order through the sequence; IDs are retained for provenance.

For every subject pair, relational features describe centroid distance, box overlap, relative motion, minimum keypoint distance, body-axis alignment, approach velocity, and a contact estimate.

These are geometric measurements, not behavior labels. A contact estimate may support a classification, but it is not proof of attack, mount, or investigation.

## 5:55–6:55 — What caching means here

**On screen:** Click **Build review skeleton cache**, then replay the video with the skeleton overlay. Next, open **Prepare Full Video** and point to **Storage mode: hdf5**.

**Narration:**

The workflow uses two caches.

The optional review cache saves YOLO skeletons and boxes for the GUI overlay, avoiding another pose pass during replay. It should be rebuilt after changing pose weights or subject count.

Prepare Full Video creates the training feature store. HDF5—the default—stores one source video’s pose and social arrays, while the manifest records window ranges into that file. NPZ instead writes individual window files for inspection or debugging.

This avoids recomputing YOLO features for every epoch or temporal head. It does not accelerate live pose inference or correct a poor detection.

## 6:55–8:10 — Classification and latency

**On screen:** Animate a 32-frame window filling, producing a class probability, then sliding forward. Show the Inference telemetry strip with **FPS**, **Windows/s**, **Window ms**, and **Classifier ms**.

**Narration:**

At inference, detections fill a rolling buffer matching the trained window. Once full, TandemYTC assembles the pose, reliability, and relational tensors and outputs class probabilities.

Real-time mode defaults to stride one, so after warm-up it can update on each incoming frame.

Latency has three parts: temporal context, YOLO pose and tracking, and the classifier forward pass. A 32-frame window at 30 frames per second needs roughly one second of observations.

Because the classifier processes compact features rather than full images, its added compute can be small—but this must be measured. The GUI separates Classifier milliseconds from total Window milliseconds and reports FPS, windows per second, CPU, GPU, and memory.

TCN is the lowest-latency option provided. Smoothing can stabilize labels but adds delay, so it should be disabled when minimum live latency matters.

## 8:10–8:55 — From annotation to a trained run

**On screen:** Export full-video annotations, show the preflight report, then move through **Prepare Full Video**, **Train**, and **Inference**.

**Narration:**

Export writes the source manifest, class list, annotations, batch metadata, and a preflight report that flags missing classes, unannotated videos, short bouts, or overlaps.

Prepare Full Video creates the feature store and JSON training manifest. Train checks the temporal head and records its configuration, epoch logs, checkpoints, accuracy, macro F1, and telemetry.

Inference accepts videos, folders, cameras, or network streams. Outputs can include predictions, annotated video, per-frame pose, and runtime metrics.

## 8:55–9:35 — Applications beyond ethology

**On screen:** Show brief examples from sport, industrial movement, rehabilitation, and security review. Keep this caption visible: `Requires a domain-specific pose model, labels, data, and validation`.

**Narration:**

The architecture can extend beyond ethology when pose and relational geometry fit the task.

In sport, temporal pose can describe movement phases and pairwise geometry can describe athlete interactions. The same sequence approach can support industrial actions, ergonomics, rehabilitation, gait analysis, or identifying segments of security footage for human review.

These are workflow extensions, not zero-shot capabilities. Each requires a suitable pose model, domain labels, representative full recordings, and held-out validation. Security, safety, and health uses also require privacy review, human oversight, and evaluation of false and missed detections.

## 9:35–9:55 — Conclusion

**On screen:** Finish on the annotated full-video timeline, then show the title and plugin name.

**Narration:**

Clips remain useful, but isolated positive clips are not the same task as continuous classification.

Full-video annotation preserves background, transitions, hard negatives, and source-level splits. TandemYTC carries that data through cached pose features, temporal training, and measured inference.

## Creator notes — not spoken

- Use an actual transition from the project in the opening. If Investigation-to-Attack is not the clearest example in the available footage, substitute the real pair of behaviors without changing the argument.
- Keep the delivery direct and connected to what is visible on screen. Avoid adding generic hooks, rhetorical filler, or jokes that are not specific to this workflow.
- Do not describe cache creation as part of the live inference path. The annotation review cache and prepared training feature store are reusable offline artifacts; live inference still performs pose estimation on incoming frames.
- Avoid saying the temporal classifier “adds no latency.” Show the telemetry and say that its incremental compute is measured separately from total window time.
- The roughly one-second example is algorithmic warm-up for the default 32-frame window at 30 fps. It is not a wall-clock benchmark; capture timing and processing time are separate.
- “Better” in the title means better aligned with continuous deployment when labeling, splits, and validation are done correctly. It is not a promise that every full-video dataset will outperform every clip dataset.
- If neighboring clips from one source must be used, keep all of them in the same train, validation, or test group.
- For non-ethology examples, frame the workflow as transferable and the included animal checkpoint as non-transferable.
