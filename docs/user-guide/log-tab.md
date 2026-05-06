# Log Console

Use the `Log` tab to monitor what the app is doing in the background.

## At a glance

| Best for | Typical use |
| --- | --- |
| Watching long-running jobs and troubleshooting failures | Training, inference, analytics, plugin activity, and warnings |

## What appears in the log

- training progress
- inference messages
- analytics updates
- plugin activity
- warnings and errors

## How to use it well

When something fails:

1. open the Log tab
2. scroll to the first clear error message
3. copy the relevant block before rerunning

## Practical tips

- Keep the Log tab open during long runs when you want fast troubleshooting.
- Use it together with `Save Project` when reproducing a previous workflow.
- If a plugin fails to launch, the Log tab is often the fastest place to see the real reason.
