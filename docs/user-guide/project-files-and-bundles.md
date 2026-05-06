# Project Files And Reproducibility Bundles

IntegraPose gives you two different ways to preserve work:

- `Save Project` / `Save Project As...`
- `Export Reproducibility Bundle...`

They are related, but they are not the same thing.

## At a glance

| Tool | File type | Best for | Typical use |
| --- | --- | --- | --- |
| `Save Project` | `.json` | Day-to-day work | Save the current GUI state so you can reopen and continue later |
| `Export Reproducibility Bundle` | `.zip` | Sharing, archiving, traceability | Package the project configuration plus reproducibility context for another machine, collaborator, lab archive, or manuscript record |

## Use `Save Project` for normal work

`Save Project` writes the current GUI state to a project `.json` file.

This is the right choice when you want to:

- stop and continue later
- keep several versions of a workflow setup
- reopen the same configuration on the same machine
- preserve tab settings, paths, and control values

### What a project `.json` is good at

- fast save/load during active work
- lightweight storage
- preserving current GUI settings

### What a project `.json` is not

- not a full archive of the environment
- not a packaged copy of model artifacts by itself
- not a substitute for saving your output folders, source videos, or datasets

## Use `Export Reproducibility Bundle` for sharing and archiving

`Export Reproducibility Bundle...` creates a `.zip` file that includes the saved project configuration plus additional traceability material.

This is the right choice when you want to:

- send a workflow setup to another user
- archive a run for a paper, supplement, or lab record
- preserve the configuration together with model and environment context
- reconstruct a project setup later with better provenance than a plain `.json`

### What the bundle usually contains

- the full saved project configuration as `project_config.json`
- project context metadata
- selected model artifacts when available
- model registry snapshot
- plugin snapshot
- environment traceability files
- manifest and checksum information for the bundle contents

### What the bundle does not replace

- original source videos unless you archive them separately
- generated output folders unless you save them separately
- the need for a compatible runtime environment on the target machine

## Simple rule for users

If you are actively working in the app:

- use `Save Project (.json)`

If you want something portable, traceable, or shareable:

- use `Export Reproducibility Bundle (.zip)`

## Typical examples

### Example 1: continuing tomorrow

You set up an inference run, tuned thresholds, and want to continue later on the same workstation.

Use:

- `Save Project`

### Example 2: sending a workflow to a collaborator

You want another person to load the same project configuration and recover the selected model paths and reproducibility context.

Use:

- `Export Reproducibility Bundle`

### Example 3: preserving a manuscript workflow

You want a traceable archive that documents the configuration used to generate results.

Use:

- `Export Reproducibility Bundle`

You may still also save the plain project `.json` for convenience, but the `.zip` is the better archival artifact.

## Importing a bundle

To reuse a previously exported bundle:

1. Open `File -> Import Reproducibility Bundle...`
2. Select the `.zip`
3. IntegraPose will apply the saved project configuration to the current GUI session
4. When bundled model artifacts are available, IntegraPose will restore them where possible

## Recommended practice

- use `.json` files during iterative work
- export a `.zip` bundle at major milestones
- archive the bundle together with raw data locations and final output folders when reproducibility matters
