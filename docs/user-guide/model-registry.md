# Model Registry

`Model Registry` is IntegraPose's built-in index of model files you have trained or selected in the GUI.

## At a glance

| Best for | Typical use |
| --- | --- |
| Reusing checkpoints across tabs | Apply the same model in Training export, Inference, or Webcam Inference |

## What it stores

Common registry details include:

- model path and file name
- when the entry was created
- whether it came from training or manual browsing
- available run context and metrics when present
- whether the file is still available on disk

Entries can be:

- project-scoped
- global

## Where you use it

Registry selectors are available in:

- `Model Training`
- `Inference`
- `Webcam Inference`

## How models enter the registry

### Automatically

When a training run completes and the app finds the expected output weights.

### Manually

When you browse to a model file in the Inference, Webcam, or export workflows.

## Good use cases

- applying a recently trained pose model to inference
- switching between multiple known checkpoints without re-browsing paths
- keeping project and global model lists organized

## Practical tips

- Click `Refresh` if a recent model does not appear yet.
- If an entry points to a moved or deleted file, browse the new path and register it again.
- Keep project roots stable if you want project-scoped registry entries to stay meaningful.
