# Project Files And Reproducibility Bundles

IntegraPose gives you two different ways to preserve work:

- `Save Project` / `Save Project As...`
- `Export Reproducibility Bundle...`

They are related, but they are not the same thing.

## At a glance

| Tool | File type | Best for | Typical use |
| --- | --- | --- | --- |
| `Save Project` | `.json` | Day-to-day work | Save the current GUI state so you can reopen and continue later |
| `Export Reproducibility Bundle` | `.zip` | Sharing and archiving | Package the project settings with supporting information for another computer, collaborator, lab archive, or manuscript record |

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
- not a packaged copy of model files by itself
- not a substitute for saving your output folders, source videos, or datasets

## Use `Export Reproducibility Bundle` for sharing and archiving

`Export Reproducibility Bundle...` creates a `.zip` file that includes the
saved project settings plus information that helps another person understand
and recreate the setup.

This is the right choice when you want to:

- send a workflow setup to another user
- archive a run for a paper, supplement, or lab record
- preserve the settings together with model and software-environment details
- reconstruct a project setup later with a clearer record than a plain `.json`

### What the bundle usually contains

- the scientific settings from the saved project as `project_config.json`
- basic information about the project
- selected model files when available
- a record of registered models and enabled plugins
- information about the Python environment
- a contents list and file-integrity checks

Machine-local paths and source URLs are cleared from the bundled configuration. This keeps usernames, credentials, and workstation folder layouts out of shared archives. After importing on another machine, select the local dataset, source-video, and output paths for that workstation.

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

You may still save the plain project `.json` for convenience, but the `.zip`
is the better archival record.

## Importing a bundle

To reuse a previously exported bundle:

1. Open `File -> Import Reproducibility Bundle...`
2. Select the `.zip`
3. IntegraPose checks the bundle for damaged files and unsafe paths before loading it
4. When bundled model files are available, IntegraPose restores them into a new import folder without overwriting existing project files
5. Re-select machine-local dataset, source, and output paths

Only import bundles from a trusted source. PyTorch model files can contain code
that runs when the model is loaded.

## Preserve bout-review work with the analysis

Bout-review decisions belong to the completed analytics run. They are not
stored inside the day-to-day project JSON, and the reproducibility bundle does
not replace archiving the generated analysis folder.

For every reviewed run, preserve:

```text
run_manifest.json
bout_review_workspace/
bout_review_exports/
```

Also retain either the annotated analytics video or the original source video.

The review workspace stores decisions and reviewer identities so the work can
be resumed. The dated exports contain the tables, figures, agreement measures,
and review history used for reporting.

IntegraPose records where the original files came from and also stores paths
within the analysis folder. When moving work to another computer or external
drive, copy the complete analytics folder rather than selected CSV files.

YOLO frame-level text files remain important when analytics must be repeated.
They are not required simply to reopen an existing review when the analytics
tables, manifest, and review video remain available.

See [Bout Review Workspace](bout-confirmation.md#keep-reviewed-results-together)
for the complete review portability workflow.

## Recommended practice

- use `.json` files during iterative work
- export a `.zip` bundle at major milestones
- archive reviewed analytics folders with their review workspaces and exports
- archive the bundle together with raw data locations and final output folders when reproducibility matters
