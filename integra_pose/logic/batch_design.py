"""Study-design metadata discovery for batch video queues.

The helpers in this module intentionally prefer a missing value over a risky
guess. Explicit filename labels, recognizable subject/time tokens, and
cohort-style directory layouts are accepted. Conflicting candidates remain
blank and are reported for preflight review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


_GROUP_KEY_ALIASES = {"group", "grp", "cohort", "condition"}
_SUBJECT_KEY_ALIASES = {
    "subject",
    "subject_id",
    "subjectid",
    "subj",
    "animal",
    "animal_id",
    "animalid",
}
_TIME_KEY_ALIASES = {"time", "time_point", "timepoint", "tp", "visit"}
_SUBJECT_PREFIXES = (
    "subject",
    "subj",
    "animal",
    "mouse",
    "rat",
    "fly",
    "fish",
    "bird",
    "monkey",
    "macaque",
    "worm",
)
_GENERIC_FOLDERS = {
    "video",
    "videos",
    "raw",
    "data",
    "dataset",
    "datasets",
    "recording",
    "recordings",
    "session",
    "sessions",
    "camera",
    "cameras",
    "batch",
    "input",
    "inputs",
    "source",
    "sources",
}
_GENERIC_SUBJECT_SUFFIXES = {
    "analysis",
    "behavior",
    "behaviour",
    "data",
    "experiment",
    "folder",
    "object",
    "openfield",
    "pose",
    "recording",
    "roi",
    "session",
    "social",
    "study",
    "time",
    "tracking",
    "trial",
    "video",
}
_KNOWN_GROUP_LABELS = {
    "control": "Control",
    "ctrl": "Control",
    "vehicle": "Vehicle",
    "veh": "Vehicle",
    "treatment": "Treatment",
    "treated": "Treatment",
    "tx": "Treatment",
    "sham": "Sham",
    "placebo": "Placebo",
    "experimental": "Experimental",
    "wildtype": "WT",
    "wild-type": "WT",
    "wt": "WT",
    "knockout": "KO",
    "knock-out": "KO",
    "ko": "KO",
}


@dataclass(frozen=True, slots=True)
class InferredDesignMetadata:
    group: str = ""
    subject_id: str = ""
    time_point: str = ""
    sources: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _split_tokens(value: Any) -> list[str]:
    return [
        token
        for token in re.split(r"[\s_\-]+", str(value or "").strip())
        if token
    ]


def _pretty_label(value: Any) -> str:
    text = re.sub(r"[\s_\-]+", " ", str(value or "").strip()).strip()
    if not text:
        return ""
    if text.casefold() in _KNOWN_GROUP_LABELS:
        return _KNOWN_GROUP_LABELS[text.casefold()]
    if text.islower():
        return text.title()
    return text.replace(" ", "_")


def _extract_key_value(segment: str, aliases: set[str]) -> str:
    """Extract the strongest ``key=value`` or ``key:value`` convention."""

    raw = str(segment or "").strip()
    if not raw:
        return ""
    alias_pattern = "|".join(
        sorted(
            (
                re.escape(alias).replace(r"\_", r"[\s_\-]*")
                for alias in aliases
            ),
            key=len,
            reverse=True,
        )
    )
    match = re.search(
        rf"(?i)(?:^|[\s_\-])(?:{alias_pattern})\s*[=:]\s*([A-Za-z0-9][A-Za-z0-9.\-]*)",
        raw,
    )
    if match:
        return str(match.group(1) or "").strip()
    return ""


def _extract_explicit_value(segment: str, aliases: set[str]) -> str:
    """Extract explicit and conventional ``key-value`` metadata labels."""

    raw = str(segment or "").strip()
    if not raw:
        return ""
    strong_value = _extract_key_value(raw, aliases)
    if strong_value:
        return strong_value

    normalized_aliases = {_normalized_key(alias) for alias in aliases}
    tokens = _split_tokens(raw)
    for index in range(len(tokens) - 1):
        for width in (2, 1):
            value_index = index + width
            if value_index >= len(tokens):
                continue
            candidate_key = _normalized_key("".join(tokens[index:value_index]))
            if candidate_key in normalized_aliases:
                return str(tokens[value_index] or "").strip()
    return ""


def _normalize_time_label(value: Any, *, prefix: str = "") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"[\s_\-]+", "", raw)
    lowered = compact.casefold()
    special = {
        "baseline": "Baseline",
        "base": "Baseline",
        "pre": "Pre",
        "pretest": "Pre",
        "post": "Post",
        "posttest": "Post",
    }
    if lowered in special:
        return special[lowered]

    match = re.fullmatch(
        r"(?i)(day|d|week|wk|w|hour|hr|h|minute|min|timepoint|time|tp|visit)?(\d+(?:\.\d+)?)",
        compact,
    )
    if not match:
        return _pretty_label(raw)
    unit = str(match.group(1) or prefix or "Time").casefold()
    number = str(match.group(2))
    unit_label = {
        "day": "Day",
        "d": "Day",
        "week": "Week",
        "wk": "Week",
        "w": "Week",
        "hour": "Hour",
        "hr": "Hour",
        "h": "Hour",
        "minute": "Minute",
        "min": "Minute",
        "visit": "Visit",
        "timepoint": "Time",
        "time": "Time",
        "tp": "Time",
    }.get(unit, "Time")
    return f"{unit_label}{number}"


def parse_time_point_numeric(value: Any) -> float | None:
    """Convert common behavioral time labels to an ordered day-scale value."""

    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass

    compact = re.sub(r"[\s_\-]+", "", raw).casefold()
    if compact in {"baseline", "base", "pre", "pretest"}:
        return 0.0
    match = re.fullmatch(
        r"(day|d|week|wk|w|hour|hr|h|minute|min|timepoint|time|tp|visit)(\d+(?:\.\d+)?)",
        compact,
    )
    if not match:
        return None
    unit = match.group(1)
    number = float(match.group(2))
    if unit in {"week", "wk", "w"}:
        return number * 7.0
    if unit in {"hour", "hr", "h"}:
        return number / 24.0
    if unit in {"minute", "min"}:
        return number / (24.0 * 60.0)
    return number


def _subject_prefix_match(segment: str) -> str:
    token_pattern = "|".join(_SUBJECT_PREFIXES)
    match = re.search(
        rf"(?i)(?:^|[\s_\-])({token_pattern})[\s_\-]*([A-Za-z0-9]+)(?:$|[\s_\-])",
        str(segment or ""),
    )
    if not match:
        return ""
    prefix = str(match.group(1))
    identifier = str(match.group(2))
    if identifier.casefold() in _GENERIC_SUBJECT_SUFFIXES:
        return ""
    if (
        prefix.casefold() not in {"subject", "subj", "animal"}
        and not any(character.isdigit() for character in identifier)
        and len(identifier) > 3
    ):
        return ""
    if prefix.casefold() in {"subject", "subj", "animal"}:
        return f"Subject{identifier}"
    return f"{prefix.title()}{identifier}"


def _subject_match(segment: str) -> str:
    explicit = _extract_explicit_value(segment, _SUBJECT_KEY_ALIASES)
    if explicit:
        return _pretty_label(explicit)
    return _subject_prefix_match(segment)


def _standalone_time_match(segment: str) -> str:
    raw = str(segment or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"[\s_\-]+", "", raw)
    if re.fullmatch(r"(?i)(baseline|base|pretest|posttest|pre|post)", compact):
        return _normalize_time_label(compact)
    match = re.fullmatch(
        r"(?i)(day|d|week|wk|w|hour|hr|h|minute|min|timepoint|time|tp|visit)(\d+(?:\.\d+)?)",
        compact,
    )
    if not match:
        return ""
    return _normalize_time_label(match.group(2), prefix=match.group(1))


def _time_match(segment: str) -> str:
    strong_explicit = _extract_key_value(segment, _TIME_KEY_ALIASES)
    if strong_explicit:
        return _normalize_time_label(strong_explicit)
    explicit = _extract_explicit_value(segment, _TIME_KEY_ALIASES)
    if explicit:
        normalized_explicit = _normalize_time_label(explicit)
        if (
            parse_time_point_numeric(normalized_explicit) is not None
            or normalized_explicit in {"Pre", "Post", "Baseline"}
        ):
            return normalized_explicit
        return ""

    raw = str(segment or "")
    special = re.search(
        r"(?i)(?:^|[\s_\-])(baseline|base|pretest|posttest|pre|post)(?:$|[\s_\-])",
        raw,
    )
    if special:
        return _normalize_time_label(special.group(1))
    match = re.search(
        r"(?i)(?:^|[\s_\-])(day|d|week|wk|w|hour|hr|h|minute|min|timepoint|time|tp|visit)[\s_\-]*(\d+(?:\.\d+)?)(?:$|[\s_\-])",
        raw,
    )
    if not match:
        return ""
    return _normalize_time_label(match.group(2), prefix=match.group(1))


def _explicit_group_match(segment: str) -> str:
    explicit = _extract_explicit_value(segment, _GROUP_KEY_ALIASES)
    return _pretty_label(explicit) if explicit else ""


def _known_group_match(segment: str) -> str:
    for token in _split_tokens(segment):
        mapped = _KNOWN_GROUP_LABELS.get(token.casefold())
        if mapped:
            return mapped
    return ""


def _unique_candidate(
    candidates: Iterable[tuple[str, str]],
    *,
    field_label: str,
) -> tuple[str, str, str]:
    by_value: dict[str, tuple[str, str]] = {}
    for value, source in candidates:
        clean = str(value or "").strip()
        if not clean:
            continue
        by_value.setdefault(clean.casefold(), (clean, source))
    if not by_value:
        return "", "", ""
    if len(by_value) == 1:
        value, source = next(iter(by_value.values()))
        return value, source, ""
    values = ", ".join(sorted(value for value, _source in by_value.values()))
    return "", "", f"Conflicting {field_label} candidates were found ({values}); assign this field manually."


def _relative_folder_parts(path: Path, source_root: Path | None) -> tuple[str, ...]:
    if source_root is None:
        return ()
    try:
        relative = path.resolve().relative_to(source_root.resolve())
    except (OSError, ValueError):
        return ()
    return tuple(str(part) for part in relative.parts[:-1])


def infer_design_metadata(
    video_paths: Sequence[str | Path],
    *,
    source_path: str | Path = "",
) -> dict[str, InferredDesignMetadata]:
    """Infer group, subject, and time metadata for a discovered video cohort."""

    resolved_paths = [Path(path).expanduser().resolve() for path in video_paths]
    source = Path(str(source_path or "").strip()).expanduser() if str(source_path or "").strip() else None
    if source is not None:
        try:
            source_root = source.resolve()
            if source_root.is_file():
                source_root = source_root.parent
        except OSError:
            source_root = None
    else:
        source_root = None
    source_root_group = ""
    source_root_subject = ""
    source_root_time = ""
    if source_root is not None:
        # A selected root is often named after the dataset itself, so only
        # accept well-known group labels here. Arbitrary group names are
        # inferred from an actual multi-cohort folder layout instead.
        source_root_group = _known_group_match(source_root.name)
        source_root_subject = _subject_prefix_match(source_root.name)
        source_root_time = _standalone_time_match(source_root.name)

    folder_parts = {
        str(path).casefold(): _relative_folder_parts(path, source_root)
        for path in resolved_paths
    }
    first_folder_values: list[str] = []
    for path in resolved_paths:
        parts = folder_parts[str(path).casefold()]
        if not parts:
            continue
        first = str(parts[0] or "").strip()
        if not first or first.casefold() in _GENERIC_FOLDERS:
            continue
        if _subject_match(first) or _time_match(first):
            continue
        first_folder_values.append(first)
    distinct_first_folders = {value.casefold() for value in first_folder_values}
    cohort_folder_layout = len(distinct_first_folders) >= 2

    inferred: dict[str, InferredDesignMetadata] = {}
    for path in resolved_paths:
        stem = path.stem
        parts = folder_parts[str(path).casefold()]
        searchable = [(stem, "filename")]
        searchable.extend((part, f"folder:{part}") for part in reversed(parts))

        subject_candidates = [
            (match, source)
            for segment, source in searchable
            if (match := _subject_match(segment))
        ]
        if source_root_subject:
            subject_candidates.append(
                (
                    source_root_subject,
                    f"source_folder:{source_root.name}",
                )
            )
        time_candidates = [
            (match, source)
            for segment, source in searchable
            if (match := _time_match(segment))
        ]
        if source_root_time:
            time_candidates.append(
                (
                    source_root_time,
                    f"source_folder:{source_root.name}",
                )
            )
        group_candidates = [
            (match, source)
            for segment, source in searchable
            if (match := _explicit_group_match(segment))
        ]
        known_groups = [
            (match, source)
            for segment, source in searchable
            if (match := _known_group_match(segment))
        ]
        group_candidates.extend(known_groups)
        if source_root_group:
            group_candidates.append(
                (_pretty_label(source_root_group), f"source_folder:{source_root.name}")
            )

        if parts:
            first = str(parts[0] or "").strip()
            first_is_valid = (
                bool(first)
                and first.casefold() not in _GENERIC_FOLDERS
                and not _subject_match(first)
                and not _time_match(first)
            )
            if first_is_valid and (
                cohort_folder_layout
                or bool(_known_group_match(first))
                or bool(_explicit_group_match(first))
            ):
                group_candidates.append((_pretty_label(first), f"folder:{first}"))

        group, group_source, group_warning = _unique_candidate(
            group_candidates,
            field_label="group",
        )
        subject_id, subject_source, subject_warning = _unique_candidate(
            subject_candidates,
            field_label="subject",
        )
        time_point, time_source, time_warning = _unique_candidate(
            time_candidates,
            field_label="time point",
        )
        sources: dict[str, str] = {}
        if group:
            sources["group"] = group_source
        if subject_id:
            sources["subject_id"] = subject_source
        if time_point:
            sources["time_point"] = time_source
        warnings = tuple(
            warning
            for warning in (group_warning, subject_warning, time_warning)
            if warning
        )
        inferred[str(path).casefold()] = InferredDesignMetadata(
            group=group,
            subject_id=subject_id,
            time_point=time_point,
            sources=sources,
            warnings=warnings,
        )
    return inferred


def apply_design_metadata_inference(
    items: Sequence[Any],
    *,
    source_path: str | Path = "",
    overwrite: bool = False,
) -> dict[str, int]:
    """Fill confident design fields on queue items and return assignment counts."""

    guesses = infer_design_metadata(
        [str(getattr(item, "video_path", "") or "") for item in items],
        source_path=source_path,
    )
    counts = {
        "group": 0,
        "subject_id": 0,
        "time_point": 0,
        "ambiguous": 0,
    }
    for item in items:
        try:
            path_key = str(Path(str(getattr(item, "video_path", "") or "")).expanduser().resolve()).casefold()
        except OSError:
            path_key = str(getattr(item, "video_path", "") or "").casefold()
        guess = guesses.get(path_key)
        if guess is None:
            continue
        sources = dict(getattr(item, "metadata_sources", {}) or {})
        for field_name in ("group", "subject_id", "time_point"):
            current = str(getattr(item, field_name, "") or "").strip()
            value = str(getattr(guess, field_name, "") or "").strip()
            if value and (overwrite or not current):
                setattr(item, field_name, value)
                sources[field_name] = str(guess.sources.get(field_name, "inferred"))
                counts[field_name] += 1
        if hasattr(item, "metadata_sources"):
            item.metadata_sources = sources
        if hasattr(item, "metadata_warnings"):
            def _warning_is_unresolved(warning: Any) -> bool:
                lowered = str(warning or "").casefold()
                if "group" in lowered:
                    return not str(getattr(item, "group", "") or "").strip()
                if "subject" in lowered:
                    return not str(
                        getattr(item, "subject_id", "") or ""
                    ).strip()
                if "time point" in lowered or "timepoint" in lowered:
                    return not str(
                        getattr(item, "time_point", "") or ""
                    ).strip()
                return True

            existing_warnings = [
                str(warning)
                for warning in list(
                    getattr(item, "metadata_warnings", []) or []
                )
                if _warning_is_unresolved(warning)
            ]
            inferred_warnings = [
                str(warning)
                for warning in guess.warnings
                if _warning_is_unresolved(warning)
            ]
            item.metadata_warnings = list(
                dict.fromkeys([*existing_warnings, *inferred_warnings])
            )
            counts["ambiguous"] += len(inferred_warnings)
    return counts
