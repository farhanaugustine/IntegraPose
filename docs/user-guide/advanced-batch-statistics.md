# Advanced Batch Statistics

The Batch Processing Wizard keeps statistical controls collapsed because most users should first confirm their videos, tracking, bouts, ROIs, and metadata.

Open **Advanced Statistics** when you need group comparisons, repeated-measures analysis, or a time-series diagnostic.

## Assign the study design first

IntegraPose gives the three batch metadata fields different roles:

| Field | Statistical role | Example |
| --- | --- | --- |
| **Group** | Comparison factor | Control, Treatment |
| **Subject ID** | Experimental unit and repeated-measures identifier | Mouse12, Rat07 |
| **Time Point** | Time or repeated factor | Baseline, Day7, Week2 |

Subject ID is especially important. Several videos from the same animal do not become several independent animals.

If the same subject has multiple recordings in the same group and time point, IntegraPose averages those recordings for inferential statistics. This prevents technical or session replicates from inflating the sample size.

## Automatic factor discovery

**Automatically add discovered design factors to group statistics** is enabled by default.

When the queue contains usable labels, IntegraPose can recognize:

- Group as the main comparison factor
- Time Point as an additional comparison or repeated factor
- Subject ID as the experimental unit

Subject ID is never treated as an ordinary independent comparison factor.

The **Additional categorical factors** field accepts available batch design fields and common aliases. For example, `cohort` and `condition` refer to Group, while `time` and `visit` refer to Time Point. It does not create a new metadata column. Full Preflight flags a name that is unavailable.

## Which analyses can run?

| Analysis | Use it for | Minimum useful design |
| --- | --- | --- |
| Kruskal-Wallis | An overall comparison across two or more levels | At least two independent subject or video units in every compared group |
| Mann-Whitney | Pairwise comparisons between levels | At least two adequately replicated levels |
| Effect sizes | Magnitude and direction of group differences | A valid overall or pairwise comparison |
| Mixed-effects model | Repeated observations from the same subjects | At least two subjects with repeated observations and more than one group or time point |
| KPSS | Checking whether an ordered time series is stationary | At least five ordered time points within a group |

Full Preflight checks the actual queue and reports which analyses are supported.

## Group comparisons

Group comparisons use one independent value per subject and factor level whenever Subject ID is available.

The exported results report:

- the factor and metric tested
- the number of independent units
- the original number of contributing video rows
- raw and adjusted p-values
- effect-size estimates
- a note when an analysis was skipped or used a video-level fallback

If Subject ID is missing, IntegraPose labels the result as a video-level fallback. That result should not be interpreted as a subject-level biological comparison.

If the same subject appears in more than one compared level, IntegraPose does not present those repeated observations as an independent-sample test. Use the mixed-effects result instead.

## Mixed-effects models

**Mixed-effects models for repeated subjects** is enabled by default.

The analysis runs only when the queue contains a suitable repeated-subject design. Subject ID identifies which observations belong to the same animal.

Use this analysis when, for example:

- the same animals were recorded at Baseline and Day7
- the same animals were recorded before and after treatment
- each animal contributes several sessions

The output includes the tested terms, estimates, uncertainty, p-values, adjusted p-values, subject counts, and the fitted comparison.

A disabled or unavailable mixed-effects result does not prevent per-video analytics from running.

## KPSS stationarity diagnostic

**KPSS stationarity diagnostic** is disabled by default.

KPSS asks whether an ordered series is consistent with a stable pattern over time. It is not a test of whether two experimental groups differ.

IntegraPose recognizes ordered labels such as:

- numeric values
- Baseline
- Day7
- Week2
- Hour24
- Minute30
- Visit3

At least five ordered time points are required within a group. Shorter experiments should normally leave KPSS disabled.

## Multiple-comparison correction

The available choices are:

| Choice | Typical use |
| --- | --- |
| `fdr_bh` | Recommended when testing several related outcomes and controlling the expected false-discovery rate |
| `bonferroni` | More conservative; useful when a small number of confirmatory comparisons were planned |

Report the correction method with the results. Use adjusted p-values when interpreting a family of comparisons.

## Reading Full Preflight

| Preflight value | Meaning |
| --- | --- |
| **Yes** | The current queue supports the analysis |
| **Partial** | The analysis can use only part of the queue or will combine repeated recordings |
| **No** | The analysis is disabled or its requirements are not met |
| **Fix** | A specific metadata or configuration problem needs attention |

Full Preflight names the affected videos and explains which analysis is affected. Missing Group, Subject ID, or Time Point information may limit statistics without preventing valid per-video results.

You can run Full Preflight before the model is ready. Model problems are reported alongside metadata, ROI, object, and statistical checks instead of hiding those checks.

## Output files

Statistical results are available in:

- the statistics sheets of `batch_results.xlsx`
- `group_stats/group_stats_overview.csv`
- `group_stats/group_pairwise_tests.csv`
- `group_stats/group_effect_sizes.csv`
- `group_stats/group_kpss_mixed_effects.csv`

See [Batch Output Map](batch-output-map.md) for the full folder layout.

## Before reporting a result

Confirm that:

- every biological subject has a correct Subject ID
- group and time labels are consistent
- exclusions are visible in `analysis_coverage_table.csv`
- the reported sample size represents independent subjects rather than videos
- the selected statistical test matches the experimental design
- adjusted p-values and effect sizes are reported together

IntegraPose can organize and calculate these analyses, but the researcher remains responsible for confirming that the design and interpretation are appropriate for the experiment.
