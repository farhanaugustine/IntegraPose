from __future__ import annotations

import os
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, Iterable, Mapping, Any


def _safe(value: Any) -> str:
    if value is None:
        return ""
    return escape(str(value))


def _slugify(value: str | None) -> str:
    if not value:
        return "value"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "value"


def _build_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    header_html = "".join(f"<th>{_safe(title)}</th>" for title in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_safe(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        body_rows.append("<tr><td colspan='{0}'>No data</td></tr>".format(len(list(headers))))
    return f"""
    <table class="data-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>
            {''.join(body_rows)}
        </tbody>
    </table>
    """


def _format_file_links(paths: Mapping[str, str | None], base_dir: str) -> str:
    if not paths:
        return "<p><em>No exported files recorded.</em></p>"
    items = []
    for label, path in paths.items():
        if not path:
            continue
        rel = os.path.relpath(path, base_dir)
        link = escape(rel)
        items.append(f"<li><strong>{_safe(label)}:</strong> <code>{link}</code></li>")
    if not items:
        return "<p><em>No exported files recorded.</em></p>"
    return "<ul>{0}</ul>".format("".join(items))


def generate_cluster_report(
    output_folder: str,
    base_name: str,
    suffix: str,
    cluster_data: Dict[int, Dict[str, Any]],
) -> str:
    """
    Produce a standalone HTML summary of clustering outputs for non-technical review.

    Args:
        output_folder: Folder where clustering artifacts are written.
        base_name: Base name of the processed video.
        suffix: Run suffix to disambiguate multiple clustering runs.
        cluster_data: Aggregated clustering dictionary returned by PoseClustering.

    Returns:
        Path to the generated HTML report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_base = _slugify(base_name)
    safe_suffix = _slugify(suffix)
    report_name = f"cluster_summary_{safe_base}_{safe_suffix}_{timestamp}.html"
    report_path = os.path.join(output_folder, report_name)

    overview_rows = []
    morph_rows = []
    timeline_sections = []
    transition_sections = []
    export_sections = []

    total_clusters = 0
    total_tracks = len(cluster_data)

    for track_id, data in sorted(cluster_data.items(), key=lambda item: item[0]):
        stats = data.get('stats', {})
        morphometrics = data.get('morphometrics', {})
        timeline = data.get('timeline', {}) or {}
        transitions = data.get('transitions', []) or []
        behavior = data.get('behavior', 'Multiple')

        for cluster_id, stat in sorted(stats.items(), key=lambda item: item[0]):
            total_clusters += 1
            overview_rows.append(
                (
                    track_id,
                    cluster_id,
                    behavior,
                    stat.get('num_poses', 0),
                    f"{stat.get('avg_dist_to_centroid', 0.0):.4f}",
                    f"{stat.get('dispersion', 0.0):.4f}",
                    stat.get('medoid_frame', ""),
                )
            )

        for cluster_id, metrics in sorted(morphometrics.items(), key=lambda item: item[0]):
            morph_rows.append(
                (
                    track_id,
                    cluster_id,
                    f"{metrics.get('centroid_dispersion', 0.0):.4f}",
                    f"{metrics.get('bbox_area', 0.0):.4f}",
                    f"{metrics.get('mean_skeleton_length', 0.0):.4f}",
                    f"{metrics.get('orientation_deg', 0.0):.2f}",
                )
            )

        segments = timeline.get('segments', [])
        if segments:
            timeline_rows = []
            for segment in segments:
                duration = segment['end'] - segment['start'] + 1
                timeline_rows.append(
                    (
                        segment['cluster'],
                        segment['start'],
                        segment['end'],
                        duration,
                    )
                )
            timeline_sections.append(
                f"""
                <h3>Track {track_id} – Timeline</h3>
                {_build_table(
                    ["Cluster", "Start Frame", "End Frame", "Duration (frames)"],
                    timeline_rows,
                )}
                """
            )

        if transitions:
            transition_rows = []
            for entry in transitions:
                delta = ""
                if entry.get('previous_frame') is not None:
                    delta = entry['frame'] - entry['previous_frame']
                transition_rows.append(
                    (
                        entry.get('from'),
                        entry.get('to'),
                        entry.get('frame'),
                        entry.get('previous_frame', ""),
                        delta,
                    )
                )
            transition_sections.append(
                f"""
                <h3>Track {track_id} – Transitions</h3>
                {_build_table(
                    ["From Cluster", "To Cluster", "Frame", "Previous Frame", "∆ Frames"],
                    transition_rows,
                )}
                """
            )

        export_paths = data.get('export_paths', {})
        if export_paths:
            export_sections.append(
                f"""
                <h3>Track {track_id} – Exported Files</h3>
                {_format_file_links(export_paths, output_folder)}
                """
            )

    overview_table = _build_table(
        ["Track", "Cluster", "Behavior", "Num Poses", "Avg Dist to Centroid", "Dispersion", "Medoid Frame"],
        overview_rows,
    )
    morph_table = _build_table(
        ["Track", "Cluster", "Centroid Dispersion", "BBox Area", "Mean Skeleton Length", "Orientation (deg)"],
        morph_rows,
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Behavior Clustering Summary – {escape(base_name)}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 30px;
            background: #f9fafc;
            color: #1f2933;
        }}
        h1 {{
            color: #0b7285;
        }}
        h2 {{
            border-bottom: 2px solid #0b7285;
            padding-bottom: 6px;
            margin-top: 40px;
        }}
        .data-table {{
            border-collapse: collapse;
            width: 100%;
            margin: 14px 0;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.1);
        }}
        .data-table th {{
            background: #0b7285;
            color: #ffffff;
            text-align: left;
            padding: 8px 12px;
        }}
        .data-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #e1e7ef;
        }}
        .data-table tr:nth-child(even) {{
            background: #f1f5f9;
        }}
        code {{
            background: #e8f5f8;
            padding: 2px 4px;
            border-radius: 4px;
        }}
        .section {{
            margin-bottom: 32px;
        }}
    </style>
</head>
<body>
    <h1>Behavior Clustering Summary</h1>
    <p><strong>Video:</strong> {escape(base_name)} | <strong>Run:</strong> {escape(suffix)} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Tracks:</strong> {total_tracks} | <strong>Clusters:</strong> {total_clusters}</p>

    <div class="section">
        <h2>Cluster Overview</h2>
        {overview_table}
    </div>

    <div class="section">
        <h2>Morphometric Signatures</h2>
        {morph_table}
    </div>

    <div class="section">
        <h2>Timeline Summaries</h2>
        {''.join(timeline_sections) if timeline_sections else '<p><em>No timeline segments available.</em></p>'}
    </div>

    <div class="section">
        <h2>Transition Hotspots</h2>
        {''.join(transition_sections) if transition_sections else '<p><em>No transitions recorded.</em></p>'}
    </div>

    <div class="section">
        <h2>Exported Artifacts</h2>
        {''.join(export_sections) if export_sections else '<p><em>No exported cluster files recorded.</em></p>'}
    </div>
</body>
</html>
"""
    Path(report_path).write_text(html_content, encoding="utf-8")
    return report_path


__all__ = ["generate_cluster_report"]
