import ast
import cv2
import pandas as pd
import numpy as np
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

def draw_star(image, center, color, size):
    """Draws a 5-pointed star on the image."""
    x, y = center
    rotation = np.pi / 2
    points = []
    for i in range(5):
        angle1 = rotation + (i * 2 * np.pi / 5)
        x1 = int(x + size * np.cos(angle1))
        y1 = int(y - size * np.sin(angle1))
        points.append((x1, y1))
        
        angle2 = angle1 + (np.pi / 5)
        x2 = int(x + (size / 2) * np.cos(angle2))
        y2 = int(y - (size / 2) * np.sin(angle2))
        points.append((x2, y2))
        
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)  # Outline
    cv2.fillPoly(image, [pts], color)  # Fill


def _extract_frame_index(filename: str):
    """Best-effort frame index extraction matching loader logic elsewhere in the app."""
    match = re.search(r'_(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    matches = re.findall(r'\d+', filename)
    if matches:
        return int(matches[-1])
    return None


def _looks_like_track_column(series: pd.Series) -> bool:
    cleaned = series.dropna()
    if cleaned.empty:
        return False
    try:
        values = cleaned.to_numpy(dtype=float)
    except Exception:
        return False
    if not np.isfinite(values).all():
        return False
    rounded = np.round(values)
    return np.all(np.isclose(values, rounded)) and np.all(rounded >= 0)


_DETECTION_COLUMNS = ['class', 'x_center', 'y_center', 'w', 'h', 'track_id']

_COLORBLIND_SAFE_PALETTE_BGR = [
    (178, 114, 0),   # blue (#0072B2)
    (0, 94, 213),    # vermillion (#D55E00)
    (115, 158, 0),   # bluish green (#009E73)
    (167, 121, 204), # reddish purple (#CC79A7)
    (233, 180, 86),  # sky blue (#56B4E9)
    (0, 159, 230),   # orange (#E69F00)
]
_OVERLAY_TEXT_COLOR = (245, 245, 245)
_OVERLAY_MUTED_TEXT_COLOR = (210, 210, 210)
_OVERLAY_DARK_BG = (22, 22, 22)
_OVERLAY_BORDER_COLOR = (0, 0, 0)


def _stable_overlay_color(name: str, *, offset: int = 0) -> Tuple[int, int, int]:
    key = str(name or '').strip()
    base = sum(ord(ch) for ch in key) if key else 0
    return _COLORBLIND_SAFE_PALETTE_BGR[(base + offset) % len(_COLORBLIND_SAFE_PALETTE_BGR)]


def _measure_text(
    text: str,
    *,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> Tuple[Tuple[int, int], int]:
    return cv2.getTextSize(str(text or ''), font_face, max(0.1, float(font_scale)), max(1, int(thickness)))


def _ellipsize_text_to_width(
    text: str,
    max_width: int,
    *,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    thickness: int = 1,
) -> str:
    clean = ' '.join(str(text or '').split())
    if not clean or max_width <= 0:
        return clean
    if _measure_text(clean, font_face=font_face, font_scale=font_scale, thickness=thickness)[0][0] <= max_width:
        return clean
    ellipsis = '...'
    if _measure_text(ellipsis, font_face=font_face, font_scale=font_scale, thickness=thickness)[0][0] > max_width:
        return ''
    trimmed = clean
    while trimmed:
        candidate = f"{trimmed}{ellipsis}"
        if _measure_text(candidate, font_face=font_face, font_scale=font_scale, thickness=thickness)[0][0] <= max_width:
            return candidate
        trimmed = trimmed[:-1].rstrip()
    return ellipsis


def _fit_text_to_width(
    text: str,
    max_width: int,
    *,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    thickness: int = 1,
    min_scale: float = 0.35,
) -> Tuple[str, float]:
    clean = ' '.join(str(text or '').split())
    if not clean:
        return '', max(min_scale, float(font_scale))
    scale = max(min_scale, float(font_scale))
    if max_width > 0:
        text_width = _measure_text(clean, font_face=font_face, font_scale=scale, thickness=thickness)[0][0]
        if text_width > max_width:
            scale = max(min_scale, scale * (max_width / max(text_width, 1)))
            if _measure_text(clean, font_face=font_face, font_scale=scale, thickness=thickness)[0][0] > max_width:
                clean = _ellipsize_text_to_width(
                    clean,
                    max_width,
                    font_face=font_face,
                    font_scale=scale,
                    thickness=thickness,
                )
    return clean, scale


def _wrap_text_to_width(
    text: str,
    max_width: int,
    *,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    thickness: int = 1,
    max_lines: int = 3,
) -> List[str]:
    clean = ' '.join(str(text or '').split())
    if not clean:
        return []
    if max_width <= 0:
        return [clean]
    words = clean.split(' ')
    lines: List[str] = []
    current = words[0]
    for idx, word in enumerate(words[1:], start=1):
        candidate = f"{current} {word}"
        if _measure_text(candidate, font_face=font_face, font_scale=font_scale, thickness=thickness)[0][0] <= max_width:
            current = candidate
            continue
        lines.append(_ellipsize_text_to_width(current, max_width, font_face=font_face, font_scale=font_scale, thickness=thickness))
        current = word
        if len(lines) >= max_lines - 1:
            remaining = ' '.join([current] + words[idx + 1 :])
            lines.append(_ellipsize_text_to_width(remaining, max_width, font_face=font_face, font_scale=font_scale, thickness=thickness))
            return [line for line in lines if line]
    lines.append(_ellipsize_text_to_width(current, max_width, font_face=font_face, font_scale=font_scale, thickness=thickness))
    return [line for line in lines if line]


def _draw_text_box(
    image: np.ndarray,
    text: str,
    top_left: Tuple[int, int],
    *,
    box_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = _OVERLAY_TEXT_COLOR,
    border_color: Tuple[int, int, int] = _OVERLAY_BORDER_COLOR,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    thickness: int = 1,
    max_width: int | None = None,
    min_scale: float = 0.35,
    padding: int = 4,
    margin: int = 6,
) -> int:
    available_width = int(max_width if max_width is not None else image.shape[1] - (margin * 2))
    text_budget = max(24, available_width - (padding * 2))
    fitted_text, fitted_scale = _fit_text_to_width(
        text,
        text_budget,
        font_face=font_face,
        font_scale=font_scale,
        thickness=thickness,
        min_scale=min_scale,
    )
    if not fitted_text:
        return int(top_left[1])
    (text_w, text_h), baseline = _measure_text(
        fitted_text,
        font_face=font_face,
        font_scale=fitted_scale,
        thickness=thickness,
    )
    box_w = text_w + (padding * 2)
    box_h = text_h + baseline + (padding * 2)
    max_x = max(margin, image.shape[1] - box_w - margin)
    max_y = max(margin, image.shape[0] - box_h - margin)
    x = int(min(max(int(top_left[0]), margin), max_x))
    y = int(min(max(int(top_left[1]), margin), max_y))
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), box_color, -1)
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), border_color, 1)
    cv2.putText(
        image,
        fitted_text,
        (x + padding, y + padding + text_h),
        font_face,
        fitted_scale,
        text_color,
        thickness,
    )
    return y + box_h


def _draw_sidebar_text(
    image: np.ndarray,
    text: str,
    *,
    x: int,
    y: int,
    max_width: int,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = _OVERLAY_TEXT_COLOR,
    thickness: int = 1,
    max_lines: int = 2,
    line_gap: int = 4,
    bottom_margin: int = 14,
) -> int:
    current_y = int(y)
    lines = _wrap_text_to_width(
        text,
        max_width,
        font_face=font_face,
        font_scale=font_scale,
        thickness=thickness,
        max_lines=max_lines,
    )
    for line in lines:
        (text_w, text_h), baseline = _measure_text(line, font_face=font_face, font_scale=font_scale, thickness=thickness)
        if current_y + baseline > image.shape[0] - bottom_margin:
            break
        cv2.putText(image, line, (x, current_y), font_face, font_scale, color, thickness)
        current_y += text_h + baseline + line_gap
    return current_y


def _estimate_sidebar_text_height(
    text: str,
    *,
    max_width: int,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    thickness: int = 1,
    max_lines: int = 2,
    line_gap: int = 4,
) -> int:
    total = 0
    lines = _wrap_text_to_width(
        text,
        max_width,
        font_face=font_face,
        font_scale=font_scale,
        thickness=thickness,
        max_lines=max_lines,
    )
    for line in lines:
        (_, text_h), baseline = _measure_text(
            line,
            font_face=font_face,
            font_scale=font_scale,
            thickness=thickness,
        )
        total += text_h + baseline + line_gap
    return total


def _metric_row(
    label: str,
    value: str | int | float,
    *,
    label_color: Tuple[int, int, int] = _OVERLAY_MUTED_TEXT_COLOR,
    value_color: Tuple[int, int, int] = _OVERLAY_TEXT_COLOR,
    font_scale: float | None = None,
    thickness: int = 1,
    min_gap: int = 12,
) -> dict:
    return {
        "kind": "metric",
        "label": str(label or ""),
        "value": str(value if value is not None else ""),
        "label_color": label_color,
        "value_color": value_color,
        "font_scale": float(font_scale if font_scale is not None else 0.5),
        "thickness": int(thickness),
        "min_gap": int(min_gap),
    }


def _estimate_sidebar_row_height(
    row: dict,
    *,
    max_width: int,
    body_scale: float,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    line_gap: int = 4,
) -> int:
    row_kind = str(row.get("kind", "text") or "text").strip().lower()
    font_scale = float(row.get("font_scale", body_scale))
    thickness = int(row.get("thickness", 1))
    if row_kind == "metric":
        label = str(row.get("label", "") or "").strip()
        value = str(row.get("value", "") or "").strip()
        min_gap = max(8, int(row.get("min_gap", 12)))
        value_width_budget = max(48, int(max_width * 0.42))
        fitted_value, fitted_scale = _fit_text_to_width(
            value,
            value_width_budget,
            font_face=font_face,
            font_scale=font_scale,
            thickness=thickness,
            min_scale=0.36,
        )
        value_width = _measure_text(
            fitted_value,
            font_face=font_face,
            font_scale=fitted_scale,
            thickness=thickness,
        )[0][0] if fitted_value else 0
        label_width_budget = max(40, max_width - value_width - min_gap)
        fitted_label = _ellipsize_text_to_width(
            label,
            label_width_budget,
            font_face=font_face,
            font_scale=font_scale,
            thickness=thickness,
        )
        (_, label_h), label_base = _measure_text(
            fitted_label,
            font_face=font_face,
            font_scale=font_scale,
            thickness=thickness,
        ) if fitted_label else ((0, 0), 0)
        (_, value_h), value_base = _measure_text(
            fitted_value,
            font_face=font_face,
            font_scale=fitted_scale,
            thickness=thickness,
        ) if fitted_value else ((0, 0), 0)
        return max(label_h, value_h) + max(label_base, value_base) + line_gap
    return _estimate_sidebar_text_height(
        str(row.get("text", "")),
        max_width=max_width,
        font_face=font_face,
        font_scale=font_scale,
        thickness=thickness,
        max_lines=int(row.get("max_lines", 2)),
        line_gap=line_gap,
    )


def _draw_sidebar_metric_row(
    image: np.ndarray,
    *,
    label: str,
    value: str,
    x: int,
    y: int,
    max_width: int,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    label_color: Tuple[int, int, int] = _OVERLAY_MUTED_TEXT_COLOR,
    value_color: Tuple[int, int, int] = _OVERLAY_TEXT_COLOR,
    thickness: int = 1,
    min_gap: int = 12,
    line_gap: int = 4,
    bottom_margin: int = 14,
) -> int:
    value_budget = max(48, int(max_width * 0.42))
    fitted_value, fitted_scale = _fit_text_to_width(
        value,
        value_budget,
        font_face=font_face,
        font_scale=font_scale,
        thickness=thickness,
        min_scale=0.36,
    )
    value_size = _measure_text(
        fitted_value,
        font_face=font_face,
        font_scale=fitted_scale,
        thickness=thickness,
    ) if fitted_value else ((0, 0), 0)
    (value_w, value_h), value_base = value_size
    label_budget = max(40, max_width - value_w - max(8, int(min_gap)))
    fitted_label = _ellipsize_text_to_width(
        label,
        label_budget,
        font_face=font_face,
        font_scale=font_scale,
        thickness=thickness,
    )
    label_size = _measure_text(
        fitted_label,
        font_face=font_face,
        font_scale=font_scale,
        thickness=thickness,
    ) if fitted_label else ((0, 0), 0)
    (label_w, label_h), label_base = label_size
    baseline = max(label_base, value_base)
    row_height = max(label_h, value_h) + baseline + line_gap
    if y + baseline > image.shape[0] - bottom_margin:
        return int(y)
    if fitted_label:
        cv2.putText(image, fitted_label, (x, y), font_face, font_scale, label_color, thickness)
    if fitted_value:
        value_x = int(x + max_width - value_w)
        cv2.putText(image, fitted_value, (value_x, y), font_face, fitted_scale, value_color, thickness)
    return int(y + row_height)


def _estimate_sidebar_card_height(
    title: str,
    rows: List[dict],
    *,
    width: int,
    title_scale: float,
    body_scale: float,
    padding: int = 10,
    section_gap: int = 6,
    line_gap: int = 4,
    title_max_lines: int = 2,
) -> int:
    content_width = max(40, int(width) - (padding * 2))
    height = padding
    height += _estimate_sidebar_text_height(
        title,
        max_width=content_width,
        font_scale=title_scale,
        thickness=2,
        max_lines=title_max_lines,
        line_gap=line_gap,
    )
    height += section_gap
    for row in rows:
        height += _estimate_sidebar_row_height(
            row,
            max_width=content_width,
            body_scale=body_scale,
            line_gap=line_gap,
        )
    height += padding
    return height


def _draw_sidebar_card(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    rows: List[dict],
    title_scale: float,
    body_scale: float,
    title_color: Tuple[int, int, int] = _OVERLAY_TEXT_COLOR,
    border_color: Tuple[int, int, int] = (75, 75, 75),
    fill_color: Tuple[int, int, int] = (28, 28, 28),
    padding: int = 10,
    section_gap: int = 6,
    line_gap: int = 4,
    title_max_lines: int = 2,
) -> int:
    card_x = int(max(0, x))
    card_y = int(max(0, y))
    card_w = int(max(40, width))
    card_h = int(max(40, height))
    cv2.rectangle(image, (card_x, card_y), (card_x + card_w, card_y + card_h), fill_color, -1)
    cv2.rectangle(image, (card_x, card_y), (card_x + card_w, card_y + card_h), border_color, 1)
    content_x = card_x + padding
    content_y = card_y + padding + 2
    content_width = max(40, card_w - (padding * 2))
    content_y = _draw_sidebar_text(
        image,
        title,
        x=content_x,
        y=content_y,
        max_width=content_width,
        font_scale=title_scale,
        color=title_color,
        thickness=2,
        max_lines=title_max_lines,
        line_gap=line_gap,
        bottom_margin=padding,
    )
    separator_y = min(card_y + card_h - padding - 2, content_y)
    cv2.line(
        image,
        (content_x, separator_y),
        (card_x + card_w - padding, separator_y),
        border_color,
        1,
    )
    content_y = separator_y + section_gap + 2
    for row in rows:
        row_kind = str(row.get("kind", "text") or "text").strip().lower()
        if row_kind == "metric":
            content_y = _draw_sidebar_metric_row(
                image,
                label=str(row.get("label", "")),
                value=str(row.get("value", "")),
                x=content_x,
                y=content_y,
                max_width=content_width,
                font_scale=float(row.get("font_scale", body_scale)),
                label_color=tuple(row.get("label_color", _OVERLAY_MUTED_TEXT_COLOR)),
                value_color=tuple(row.get("value_color", _OVERLAY_TEXT_COLOR)),
                thickness=int(row.get("thickness", 1)),
                min_gap=int(row.get("min_gap", 12)),
                line_gap=line_gap,
                bottom_margin=padding,
            )
        else:
            content_y = _draw_sidebar_text(
                image,
                str(row.get("text", "")),
                x=content_x,
                y=content_y,
                max_width=content_width,
                font_scale=float(row.get("font_scale", body_scale)),
                color=tuple(row.get("color", _OVERLAY_MUTED_TEXT_COLOR)),
                thickness=int(row.get("thickness", 1)),
                max_lines=int(row.get("max_lines", 2)),
                line_gap=line_gap,
                bottom_margin=padding,
            )
    return card_y + card_h


def _draw_sidebar_card_grid(
    image: np.ndarray,
    cards: List[dict],
    *,
    x: int,
    y: int,
    width: int,
    columns: int,
    gap_x: int = 12,
    gap_y: int = 12,
    title_scale: float,
    body_scale: float,
) -> int:
    if not cards:
        return int(y)
    col_count = max(1, int(columns or 1))
    card_width = max(120, int((width - (gap_x * (col_count - 1))) / col_count))
    heights = [
        _estimate_sidebar_card_height(
            str(card.get("title", "")),
            list(card.get("rows", []) or []),
            width=card_width,
            title_scale=float(card.get("title_scale", title_scale)),
            body_scale=float(card.get("body_scale", body_scale)),
        )
        for card in cards
    ]
    current_y = int(y)
    for row_start in range(0, len(cards), col_count):
        row_cards = cards[row_start:row_start + col_count]
        row_heights = heights[row_start:row_start + col_count]
        row_height = max(row_heights) if row_heights else 0
        for idx, card in enumerate(row_cards):
            card_x = int(x + idx * (card_width + gap_x))
            _draw_sidebar_card(
                image,
                x=card_x,
                y=current_y,
                width=card_width,
                height=row_height,
                title=str(card.get("title", "")),
                rows=list(card.get("rows", []) or []),
                title_scale=float(card.get("title_scale", title_scale)),
                body_scale=float(card.get("body_scale", body_scale)),
                title_color=tuple(card.get("title_color", _OVERLAY_TEXT_COLOR)),
                border_color=tuple(card.get("border_color", (75, 75, 75))),
                fill_color=tuple(card.get("fill_color", (28, 28, 28))),
            )
        current_y += row_height + gap_y
    return current_y


def _estimate_sidebar_card_grid_height(
    cards: List[dict],
    *,
    width: int,
    columns: int,
    gap_x: int = 12,
    gap_y: int = 12,
    title_scale: float,
    body_scale: float,
) -> int:
    if not cards:
        return 0
    col_count = max(1, int(columns or 1))
    card_width = max(120, int((width - (gap_x * (col_count - 1))) / col_count))
    heights = [
        _estimate_sidebar_card_height(
            str(card.get("title", "")),
            list(card.get("rows", []) or []),
            width=card_width,
            title_scale=float(card.get("title_scale", title_scale)),
            body_scale=float(card.get("body_scale", body_scale)),
        )
        for card in cards
    ]
    total = 0
    for row_start in range(0, len(heights), col_count):
        row_heights = heights[row_start:row_start + col_count]
        total += max(row_heights) if row_heights else 0
        total += gap_y
    return total


def _is_point_pair(candidate) -> bool:
    return isinstance(candidate, (list, tuple, np.ndarray)) and len(candidate) >= 2


def _coerce_polygon_groups(roi_payload: Optional[dict]) -> Dict[str, List[np.ndarray]]:
    normalized: Dict[str, List[np.ndarray]] = {}
    if not isinstance(roi_payload, dict):
        return normalized

    for raw_name, payload in roi_payload.items():
        name = str(raw_name or '').strip()
        if not name or payload is None:
            continue

        polygons_payload = payload
        if isinstance(payload, dict):
            if payload.get('enabled', True) is False:
                continue
            polygons_payload = payload.get('polygons', [])

        polygon_groups = []
        if isinstance(polygons_payload, np.ndarray):
            if polygons_payload.ndim == 2 and polygons_payload.shape[1] == 2:
                polygon_groups = [polygons_payload]
            elif polygons_payload.ndim == 3 and polygons_payload.shape[2] == 2:
                polygon_groups = [poly for poly in polygons_payload]
        elif isinstance(polygons_payload, (list, tuple)):
            if polygons_payload and _is_point_pair(polygons_payload[0]) and len(polygons_payload[0]) == 2:
                polygon_groups = [polygons_payload]
            else:
                polygon_groups = [poly for poly in polygons_payload if isinstance(poly, (list, tuple, np.ndarray))]

        valid_polygons: List[np.ndarray] = []
        for polygon in polygon_groups:
            try:
                poly_np = np.asarray(polygon, dtype=np.int32)
            except Exception:
                continue
            if poly_np.ndim != 2 or poly_np.shape[0] < 3 or poly_np.shape[1] != 2:
                continue
            valid_polygons.append(poly_np)
        if valid_polygons:
            normalized[name] = valid_polygons

    return normalized


def _read_optional_csv(path: str) -> pd.DataFrame:
    target = str(path or '').strip()
    if not target or not os.path.isfile(target):
        return pd.DataFrame()
    try:
        return pd.read_csv(target)
    except Exception:
        return pd.DataFrame()


def load_object_overlay_metrics(module_outputs: Optional[dict]) -> Tuple[dict, dict]:
    object_metrics: Dict[str, pd.DataFrame] = {}
    object_events = {'entries': [], 'exits': []}
    if not isinstance(module_outputs, dict):
        return object_metrics, object_events

    object_module = module_outputs.get('object_interactions')
    if not isinstance(object_module, dict):
        return object_metrics, object_events

    files = object_module.get('files')
    if isinstance(files, dict):
        for key in ('summary', 'per_frame', 'approach_retreat_summary'):
            df = _read_optional_csv(str(files.get(key, '') or ''))
            if not df.empty:
                object_metrics[key] = df

    events = object_module.get('events')
    if isinstance(events, dict):
        object_events = {
            'entries': list(events.get('entries', []) or []),
            'exits': list(events.get('exits', []) or []),
        }

    return object_metrics, object_events


def _parse_memberships(value) -> Tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = text
    else:
        parsed = value

    if isinstance(parsed, str):
        return (parsed.strip(),) if parsed.strip() else ()
    if isinstance(parsed, (list, tuple, set)):
        members = []
        for item in parsed:
            text = str(item or '').strip()
            if text:
                members.append(text)
        return tuple(sorted(set(members)))
    return ()


def _build_object_frame_lookup(per_frame_df: Optional[pd.DataFrame]) -> Dict[int, Dict[int, dict]]:
    if per_frame_df is None or per_frame_df.empty:
        return {}
    required_cols = {'frame', 'track_id'}
    if not required_cols.issubset(per_frame_df.columns):
        return {}

    work_df = per_frame_df.copy()
    work_df['frame'] = pd.to_numeric(work_df['frame'], errors='coerce')
    work_df['track_id'] = pd.to_numeric(work_df['track_id'], errors='coerce')
    work_df = work_df.dropna(subset=['frame', 'track_id'])
    if work_df.empty:
        return {}

    frame_lookup: Dict[int, Dict[int, dict]] = defaultdict(dict)
    for _, row in work_df.iterrows():
        frame_idx = int(row.get('frame'))
        track_id = int(row.get('track_id'))
        roi_name = str(row.get('Object Interaction ROI', '') or '').strip()
        memberships = _parse_memberships(row.get('Object Interaction Memberships'))
        interaction_state = str(row.get('Object Interaction State', '') or '').strip()
        distance_px = pd.to_numeric(row.get('Object Interaction Distance (px)'), errors='coerce')
        if memberships or roi_name or pd.notna(distance_px):
            frame_lookup[frame_idx][track_id] = {
                'roi_name': roi_name,
                'memberships': memberships,
                'state': interaction_state,
                'distance_px': float(distance_px) if pd.notna(distance_px) else np.nan,
            }

    return dict(frame_lookup)


def _build_detection_schedule(yolo_txt_folder: str) -> List[Tuple[int, str]]:
    """
    Build an ordered list mapping frame indices to YOLO detection text files.
    The list is sorted by frame index to allow sequential consumption without loading files into memory.
    """
    if not yolo_txt_folder or not os.path.isdir(yolo_txt_folder):
        return []

    detection_files = [
        fname for fname in os.listdir(yolo_txt_folder) if fname.lower().endswith('.txt')
    ]
    if not detection_files:
        return []

    def _sort_key(filename: str) -> Tuple[int, str]:
        frame_idx = _extract_frame_index(filename)
        return (frame_idx if frame_idx is not None else np.iinfo(np.int32).max, filename)

    schedule: List[Tuple[int, str]] = []
    for filename in sorted(detection_files, key=_sort_key):
        frame_idx = _extract_frame_index(filename)
        if frame_idx is None:
            continue
        schedule.append((frame_idx, os.path.join(yolo_txt_folder, filename)))
    return schedule


def _parse_detection_file(txt_path: str) -> Optional[pd.DataFrame]:
    """
    Parse a YOLO detection text file into a DataFrame compatible with downstream rendering logic.
    Returns None when the file cannot be parsed; returns an empty DataFrame when the file has no detections.
    """
    try:
        df_raw = pd.read_csv(
            txt_path,
            sep=r'\s+',
            header=None,
            dtype=np.float32,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_DETECTION_COLUMNS)
    except FileNotFoundError:
        print(f"Warning: detection file missing during video render: {txt_path}")
        return None
    except Exception as exc:
        print(f"Warning: could not parse detection file {txt_path}: {exc}")
        return None

    if df_raw.empty:
        return pd.DataFrame(columns=_DETECTION_COLUMNS)

    df_numeric = df_raw.apply(pd.to_numeric, errors='coerce')
    if df_numeric.empty or df_numeric.shape[1] < 5:
        return pd.DataFrame(columns=_DETECTION_COLUMNS)

    total_cols = df_numeric.shape[1]
    track_col_idx = total_cols - 1 if _looks_like_track_column(df_numeric.iloc[:, -1]) else None

    detections_df = pd.DataFrame(
        {
            'class': df_numeric.iloc[:, 0].round().astype('Int64'),
            'x_center': df_numeric.iloc[:, 1],
            'y_center': df_numeric.iloc[:, 2],
            'w': df_numeric.iloc[:, 3],
            'h': df_numeric.iloc[:, 4],
        }
    )

    if track_col_idx is not None:
        detections_df['track_id'] = df_numeric.iloc[:, track_col_idx].round().astype('Int64')
    else:
        detections_df['track_id'] = pd.Series([0] * len(detections_df), dtype='Int64')

    finite_mask = np.isfinite(detections_df[['x_center', 'y_center', 'w', 'h']]).all(axis=1)
    if not finite_mask.all():
        detections_df = detections_df.loc[finite_mask].reset_index(drop=True)

    return detections_df

def create_annotated_video(
    video_path,
    output_path,
    yolo_txt_folder,
    rois,
    detailed_bouts_df,
    roi_metrics=None,
    roi_events=None,
    object_rois=None,
    object_metrics=None,
    object_events=None,
) -> bool:
    """
    Creates an annotated video with bounding boxes, track IDs, ROIs, a sidebar dashboard,
    object/stimulus overlays, and a star marker in the center of each bounding box.

    Args:
        video_path (str): Path to the source video.
        output_path (str): Path to save the new annotated video.
        yolo_txt_folder (str): Folder containing YOLO detection .txt files.
        rois (dict): Dictionary of ROI names to polygon points.
        detailed_bouts_df (pd.DataFrame): DataFrame with detailed bout information.
        roi_metrics (dict, optional): Aggregated ROI metrics for overlay summaries.
        roi_events (dict, optional): Frame-indexed ROI entry/exit events for dynamic counters.
        object_rois (dict, optional): Dictionary of stimulus/object ROI polygons.
        object_metrics (dict, optional): Object interaction summary/per-frame metrics.
        object_events (dict, optional): Object entry/exit events for dynamic counters.

    Returns:
        bool: True when the annotated video is written successfully; False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("Warning: Invalid FPS detected in source video, defaulting to 30.")
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    target_video_width = 800
    if orig_width <= 0 or orig_height <= 0:
        probe_ok, probe_frame = cap.read()
        if not probe_ok:
            print("FATAL: Could not read frames from video for rendering.")
            cap.release()
            return False
        orig_height, orig_width = probe_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    aspect_ratio = orig_height / orig_width if orig_width else 1.0
    target_video_height = int(target_video_width * aspect_ratio) if orig_width else target_video_width
    sidebar_width = max(480, min(640, int(target_video_width * 0.68)))
    card_columns = 2 if sidebar_width >= 520 else 1
    overlay_label_scale = max(0.45, min(0.65, target_video_width / 1300.0))
    overlay_meta_scale = max(0.4, min(0.55, target_video_width / 1450.0))
    sidebar_title_scale = max(0.72, min(0.9, target_video_height / 760.0))
    sidebar_header_scale = max(0.58, min(0.78, target_video_height / 920.0))
    sidebar_text_scale = max(0.46, min(0.6, target_video_height / 1120.0))

    detection_schedule = _build_detection_schedule(yolo_txt_folder)
    detection_idx = 0
    detection_len = len(detection_schedule)
    normalized_rois = _coerce_polygon_groups(rois)
    normalized_object_rois = _coerce_polygon_groups(object_rois)

    required_cols = {'Track ID', 'Behavior', 'Start Frame', 'End Frame'}
    if not required_cols.issubset(detailed_bouts_df.columns):
        print("Warning: Detailed bouts data missing required columns; behavior overlay may be incomplete.")
    valid_bouts = detailed_bouts_df.dropna(subset=list(required_cols.intersection(detailed_bouts_df.columns)))

    observed_behaviors: List[str] = []
    if 'Behavior' in valid_bouts.columns and not valid_bouts.empty:
        observed_behaviors = sorted({str(val) for val in valid_bouts['Behavior'].dropna().unique()})

    track_ids: List[int] = []
    if 'Track ID' in valid_bouts.columns and not valid_bouts.empty:
        for track_val in valid_bouts['Track ID'].dropna().unique():
            try:
                track_ids.append(int(track_val))
            except (TypeError, ValueError):
                continue
        track_ids = sorted(set(track_ids))

    bout_counts: Dict[int, Dict[str, int]] = {
        track_id: {behavior: 0 for behavior in observed_behaviors}
        for track_id in track_ids
    }

    start_events: Dict[int, List[Tuple[int, str, int]]] = defaultdict(list)
    for _, bout in valid_bouts.iterrows():
        try:
            track_id = int(bout['Track ID'])
            behavior = str(bout['Behavior'])
            start_frame = int(bout['Start Frame'])
            end_frame = int(bout['End Frame'])
        except (KeyError, TypeError, ValueError):
            continue
        start_events[start_frame].append((track_id, behavior, end_frame))
        track_counts = bout_counts.setdefault(track_id, {})
        if behavior not in track_counts:
            track_counts[behavior] = 0
        observed_behaviors.append(behavior)

    observed_behaviors = sorted({behavior for behavior in observed_behaviors if behavior})

    roi_metrics = roi_metrics or {}
    roi_events = roi_events or {}
    object_metrics = object_metrics or {}
    object_events = object_events or {'entries': [], 'exits': []}
    roi_entry_counts = defaultdict(int)
    roi_exit_counts = defaultdict(int)
    entries_by_frame = defaultdict(list)
    exits_by_frame = defaultdict(list)
    object_entry_counts = defaultdict(int)
    object_exit_counts = defaultdict(int)
    object_entries_by_frame = defaultdict(list)
    object_exits_by_frame = defaultdict(list)

    for event in roi_events.get('entries', []):
        try:
            frame_idx = int(event.get('frame', 0))
            roi_name = event.get('roi_name')
        except (AttributeError, TypeError, ValueError):
            continue
        if roi_name:
            entries_by_frame[frame_idx].append(str(roi_name))

    for event in roi_events.get('exits', []):
        try:
            frame_idx = int(event.get('frame', 0))
            roi_name = event.get('roi_name')
        except (AttributeError, TypeError, ValueError):
            continue
        if roi_name:
            exits_by_frame[frame_idx].append(str(roi_name))

    for event in object_events.get('entries', []):
        try:
            frame_idx = int(event.get('frame', 0))
            roi_name = event.get('object_roi')
        except (AttributeError, TypeError, ValueError):
            continue
        if roi_name:
            object_entries_by_frame[frame_idx].append(str(roi_name))

    for event in object_events.get('exits', []):
        try:
            frame_idx = int(event.get('frame', 0))
            roi_name = event.get('object_roi')
        except (AttributeError, TypeError, ValueError):
            continue
        if roi_name:
            object_exits_by_frame[frame_idx].append(str(roi_name))

    roi_overview = roi_metrics.get('entries_exits') if isinstance(roi_metrics, dict) else None
    object_summary = object_metrics.get('summary') if isinstance(object_metrics, dict) else None
    object_per_frame = object_metrics.get('per_frame') if isinstance(object_metrics, dict) else None
    object_approach_summary = object_metrics.get('approach_retreat_summary') if isinstance(object_metrics, dict) else None
    object_frame_lookup = _build_object_frame_lookup(object_per_frame if isinstance(object_per_frame, pd.DataFrame) else None)

    object_summary_lookup: Dict[str, dict] = {}
    if isinstance(object_summary, pd.DataFrame) and not object_summary.empty and 'Object ROI' in object_summary.columns:
        for _, row in object_summary.iterrows():
            name = str(row.get('Object ROI', '') or '').strip()
            if name:
                object_summary_lookup[name] = row.to_dict()

    object_approach_lookup: Dict[str, dict] = {}
    if (
        isinstance(object_approach_summary, pd.DataFrame)
        and not object_approach_summary.empty
        and 'Object ROI' in object_approach_summary.columns
    ):
        for _, row in object_approach_summary.iterrows():
            name = str(row.get('Object ROI', '') or '').strip()
            if name:
                object_approach_lookup[name] = row.to_dict()

    def _card_row(
        text: str,
        *,
        color: Tuple[int, int, int] = _OVERLAY_MUTED_TEXT_COLOR,
        font_scale: float | None = None,
        thickness: int = 1,
        max_lines: int = 3,
    ) -> dict:
        return {
            "text": str(text or ""),
            "color": color,
            "font_scale": float(font_scale if font_scale is not None else 0.5),
            "thickness": int(thickness),
            "max_lines": int(max_lines),
        }

    def _build_overview_card(frame_idx: int, *, visible_tracks: int, active_objects: int) -> dict:
        current_time_s = (float(frame_idx) / fps) if fps and fps > 0 else 0.0
        return {
            "title": "Validation Dashboard",
            "title_color": _OVERLAY_TEXT_COLOR,
            "border_color": (85, 85, 85),
            "fill_color": (24, 24, 24),
            "rows": [
                _card_row(os.path.basename(str(video_path or "")) or "Annotated video", color=(225, 225, 225), max_lines=3),
                _card_row(f"Frame {int(frame_idx)} | Time {current_time_s:.2f}s", color=_OVERLAY_TEXT_COLOR),
                _card_row(f"Visible tracks: {int(visible_tracks)} | Active objects: {int(active_objects)}", color=_OVERLAY_MUTED_TEXT_COLOR),
            ],
        }

    def _build_event_card(
        *,
        roi_entries: List[str],
        roi_exits: List[str],
        object_entries: List[str],
        object_exits: List[str],
    ) -> dict:
        rows: List[dict] = []
        if roi_entries:
            rows.append(_card_row(f"ROI entry: {', '.join(roi_entries[:4])}", color=(220, 240, 255), max_lines=3))
        if roi_exits:
            rows.append(_card_row(f"ROI exit: {', '.join(roi_exits[:4])}", color=(220, 240, 255), max_lines=3))
        if object_entries:
            rows.append(_card_row(f"Object entry: {', '.join(object_entries[:4])}", color=(255, 228, 205), max_lines=3))
        if object_exits:
            rows.append(_card_row(f"Object exit: {', '.join(object_exits[:4])}", color=(255, 228, 205), max_lines=3))
        if not rows:
            rows.append(_card_row("No ROI or object entry/exit events on this frame.", color=_OVERLAY_MUTED_TEXT_COLOR, max_lines=3))
        return {
            "title": "Current Frame Events",
            "title_color": _OVERLAY_TEXT_COLOR,
            "border_color": (95, 95, 95),
            "fill_color": (24, 24, 24),
            "rows": rows,
        }

    def _build_track_cards(track_specs: List[dict]) -> List[dict]:
        if not track_specs:
            return [
                {
                    "title": "Tracks In View",
                    "title_color": _OVERLAY_TEXT_COLOR,
                    "border_color": (85, 85, 85),
                    "fill_color": (24, 24, 24),
                    "rows": [_card_row("No tracked detections on this frame.", color=_OVERLAY_MUTED_TEXT_COLOR, max_lines=3)],
                }
            ]

        cards: List[dict] = []
        max_cards = 4
        shown_specs = track_specs[:max_cards]
        for spec in shown_specs:
            track_id = spec.get("track_id", "?")
            accent = tuple(spec.get("accent_color", (80, 200, 80)))
            rows = [
                _card_row(f"Behavior: {spec.get('behavior', 'N/A')}", color=_OVERLAY_TEXT_COLOR, max_lines=2),
                _card_row(f"Total bouts: {int(spec.get('total_bouts', 0))}", color=_OVERLAY_MUTED_TEXT_COLOR),
            ]
            object_text = str(spec.get("object_text", "") or "").strip()
            if object_text:
                rows.append(_card_row(object_text, color=_OVERLAY_MUTED_TEXT_COLOR, max_lines=3))
            cards.append(
                {
                    "title": f"Track {track_id}",
                    "title_color": accent,
                    "border_color": accent,
                    "fill_color": (24, 24, 24),
                    "rows": rows,
                }
            )
        if len(track_specs) > max_cards:
            cards.append(
                {
                    "title": "More Tracks",
                    "title_color": _OVERLAY_TEXT_COLOR,
                    "border_color": (95, 95, 95),
                    "fill_color": (24, 24, 24),
                    "rows": [_card_row(f"+{len(track_specs) - max_cards} additional track(s) visible.", color=_OVERLAY_MUTED_TEXT_COLOR)],
                }
            )
        return cards

    def _build_roi_cards(current_entries: dict, current_exits: dict) -> List[dict]:
        cards: List[dict] = []
        if isinstance(roi_overview, pd.DataFrame) and not roi_overview.empty:
            for _, row in roi_overview.iterrows():
                name = str(row.get('ROI Name', '') or '').strip()
                if not name:
                    continue
                roi_color = _stable_overlay_color(name, offset=0)
                total_entries = pd.to_numeric(row.get('Entries'), errors='coerce')
                total_exits = pd.to_numeric(row.get('Exits'), errors='coerce')
                time_in_roi = pd.to_numeric(row.get('Time in ROI (s)'), errors='coerce')
                mean_dwell = pd.to_numeric(row.get('Mean Dwell Duration (s)'), errors='coerce')
                dwell_events = pd.to_numeric(row.get('Dwell Events'), errors='coerce')
                total_dwell = pd.to_numeric(row.get('Total Dwell Time (s)'), errors='coerce')
                rows = [
                    _metric_row(
                        "Entries",
                        f"{int(current_entries.get(name, 0))} / {int(total_entries) if pd.notna(total_entries) else 0}",
                    ),
                    _metric_row(
                        "Exits",
                        f"{int(current_exits.get(name, 0))} / {int(total_exits) if pd.notna(total_exits) else 0}",
                    ),
                ]
                if pd.notna(time_in_roi):
                    rows.append(_metric_row("Time in ROI", f"{float(time_in_roi):.1f}s"))
                if pd.notna(total_dwell):
                    rows.append(_metric_row("Total dwell", f"{float(total_dwell):.1f}s"))
                if pd.notna(dwell_events):
                    rows.append(_metric_row("Dwell events", int(dwell_events), value_color=(220, 220, 220)))
                if pd.notna(mean_dwell):
                    rows.append(_metric_row("Mean dwell", f"{float(mean_dwell):.1f}s", value_color=(220, 220, 220)))
                cards.append(
                    {
                        "title": name,
                        "title_color": roi_color,
                        "border_color": roi_color,
                        "fill_color": (24, 24, 24),
                        "rows": rows,
                    }
                )
        return cards

    def _build_object_cards(current_entries: dict, current_exits: dict, active_counts: dict) -> List[dict]:
        object_names_local = list(normalized_object_rois.keys())
        if not object_names_local and object_summary_lookup:
            object_names_local = sorted(object_summary_lookup.keys())
        cards: List[dict] = []
        for object_name in object_names_local:
            summary_row = object_summary_lookup.get(object_name, {})
            approach_row = object_approach_lookup.get(object_name, {})
            object_color = _stable_overlay_color(object_name, offset=2)
            total_entries = pd.to_numeric(summary_row.get('Entries'), errors='coerce')
            total_exits = pd.to_numeric(summary_row.get('Exits'), errors='coerce')
            total_time = pd.to_numeric(summary_row.get('Time Interacting (s)'), errors='coerce')
            dwell_events = pd.to_numeric(summary_row.get('Dwell Events'), errors='coerce')
            contact_time = pd.to_numeric(summary_row.get('Time Contact (s)'), errors='coerce')
            proximity_time = pd.to_numeric(summary_row.get('Time Proximity Only (s)'), errors='coerce')
            mean_dwell = pd.to_numeric(summary_row.get('Mean Dwell Duration (s)'), errors='coerce')
            approach_events = pd.to_numeric(approach_row.get('Approach Events'), errors='coerce')
            retreat_events = pd.to_numeric(approach_row.get('Retreat Events'), errors='coerce')
            rows = [
                _metric_row("Active now", int(active_counts.get(object_name, 0)), value_color=_OVERLAY_TEXT_COLOR),
                _metric_row(
                    "Entries",
                    f"{int(current_entries.get(object_name, 0))} / {int(total_entries) if pd.notna(total_entries) else 0}",
                ),
                _metric_row(
                    "Exits",
                    f"{int(current_exits.get(object_name, 0))} / {int(total_exits) if pd.notna(total_exits) else 0}",
                ),
            ]
            if pd.notna(total_time):
                rows.append(_metric_row("Interact", f"{float(total_time):.1f}s"))
            if pd.notna(dwell_events):
                rows.append(_metric_row("Dwell events", int(dwell_events), value_color=(220, 220, 220)))
            if pd.notna(mean_dwell):
                rows.append(_metric_row("Mean dwell", f"{float(mean_dwell):.1f}s", value_color=(220, 220, 220)))
            if pd.notna(contact_time):
                rows.append(_metric_row("Contact", f"{float(contact_time):.1f}s", value_color=(220, 220, 220)))
            if pd.notna(proximity_time):
                rows.append(_metric_row("Proximity", f"{float(proximity_time):.1f}s", value_color=(220, 220, 220)))
            if pd.notna(approach_events):
                rows.append(_metric_row("Approach", int(approach_events), value_color=(220, 220, 220)))
            if pd.notna(retreat_events):
                rows.append(_metric_row("Retreat", int(retreat_events), value_color=(220, 220, 220)))
            cards.append(
                {
                    "title": object_name,
                    "title_color": object_color,
                    "border_color": object_color,
                    "fill_color": (24, 24, 24),
                    "rows": rows,
                }
            )
        return cards

    preview_header_card = _build_overview_card(0, visible_tracks=max(len(track_ids), 0), active_objects=0)
    preview_event_card = _build_event_card(roi_entries=[], roi_exits=[], object_entries=[], object_exits=[])
    preview_track_cards = _build_track_cards(
        [
            {
                "track_id": track_id,
                "behavior": "N/A",
                "total_bouts": 0,
                "object_text": "",
                "accent_color": (80, 200, 80),
            }
            for track_id in track_ids[:4]
        ]
    )
    preview_roi_cards = _build_roi_cards(defaultdict(int), defaultdict(int))
    preview_object_cards = _build_object_cards(defaultdict(int), defaultdict(int), defaultdict(int))

    def _estimate_section_heading_height(text: str) -> int:
        return _estimate_sidebar_text_height(
            text,
            max_width=max(80, sidebar_width - 24),
            font_scale=sidebar_header_scale,
            thickness=2,
            max_lines=1,
            line_gap=6,
        )

    required_sidebar_height = 36
    required_sidebar_height += _estimate_sidebar_card_grid_height(
        [preview_header_card],
        width=sidebar_width - 24,
        columns=1,
        title_scale=sidebar_header_scale,
        body_scale=sidebar_text_scale,
    )
    required_sidebar_height += _estimate_sidebar_card_grid_height(
        [preview_event_card],
        width=sidebar_width - 24,
        columns=1,
        title_scale=sidebar_header_scale,
        body_scale=sidebar_text_scale,
    )
    required_sidebar_height += _estimate_section_heading_height("Tracks In View") + 8
    required_sidebar_height += _estimate_sidebar_card_grid_height(
        preview_track_cards,
        width=sidebar_width - 24,
        columns=card_columns,
        title_scale=sidebar_header_scale,
        body_scale=sidebar_text_scale,
    )
    if preview_roi_cards:
        required_sidebar_height += _estimate_section_heading_height("ROI Metrics") + 8
        required_sidebar_height += _estimate_sidebar_card_grid_height(
            preview_roi_cards,
            width=sidebar_width - 24,
            columns=card_columns,
            title_scale=sidebar_text_scale,
            body_scale=sidebar_text_scale,
        )
    if preview_object_cards:
        required_sidebar_height += _estimate_section_heading_height("Stimulus Objects") + 8
        required_sidebar_height += _estimate_sidebar_card_grid_height(
            preview_object_cards,
            width=sidebar_width - 24,
            columns=card_columns,
            title_scale=sidebar_text_scale,
            body_scale=sidebar_text_scale,
        )
    required_sidebar_height += 46

    total_height = max(target_video_height, 640, min(1600, required_sidebar_height))
    frame_pad_bottom = max(0, total_height - target_video_height)
    total_width = target_video_width + sidebar_width
    sidebar_title_scale = max(0.72, min(0.92, total_height / 760.0))
    sidebar_header_scale = max(0.58, min(0.8, total_height / 920.0))
    sidebar_text_scale = max(0.46, min(0.62, total_height / 1120.0))

    writer = None
    chosen_codec = None
    for codec in ('mp4v', 'avc1', 'MJPG'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
        if candidate.isOpened():
            writer = candidate
            chosen_codec = codec
            break
        candidate.release()
    if writer is None or not writer.isOpened():
        print("FATAL: Could not open video writer. Video will not be saved.")
        cap.release()
        return False
    if chosen_codec and chosen_codec != 'avc1':
        print(f"Info: Using codec '{chosen_codec}' for annotated video export.")

    # Per-track list of currently-open bouts, each entry: {'behavior', 'start_frame', 'end_frame'}.
    # Bouts can overlap in time on the same track because analyze_bouts gap-fills per
    # (track_id, class_id) independently. The panel displays the most-recently-started
    # bout that is still open, so when an inner bout ends we fall back to the outer
    # bout that is still active rather than going to N/A or showing a stale label.
    open_bouts_by_track: Dict[int, List[Dict[str, int]]] = defaultdict(list)

    def _current_behavior_for_track(track_id: int) -> str:
        open_list = open_bouts_by_track.get(track_id)
        if not open_list:
            return "N/A"
        # Tie-break: latest start_frame wins; if equal, latest end_frame wins.
        latest = max(open_list, key=lambda b: (b['start_frame'], b['end_frame']))
        return str(latest['behavior'])

    scale_x = target_video_width / orig_width if orig_width else 1.0
    scale_y = target_video_height / orig_height if orig_height else 1.0
    pbar_total = frame_count if frame_count > 0 else None
    pbar = tqdm(total=pbar_total, desc="Creating Annotated Video")
    current_frame_idx = 0
    frames_written = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (target_video_width, target_video_height))
            if frame_pad_bottom:
                frame = cv2.copyMakeBorder(
                    frame,
                    0,
                    frame_pad_bottom,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            sidebar = np.full((total_height, sidebar_width, 3), 18, dtype=np.uint8)

            for roi_name in entries_by_frame.get(current_frame_idx, []):
                roi_entry_counts[roi_name] += 1
            for roi_name in exits_by_frame.get(current_frame_idx, []):
                roi_exit_counts[roi_name] += 1
            for roi_name in object_entries_by_frame.get(current_frame_idx, []):
                object_entry_counts[roi_name] += 1
            for roi_name in object_exits_by_frame.get(current_frame_idx, []):
                object_exit_counts[roi_name] += 1

            current_object_states = object_frame_lookup.get(current_frame_idx, {})
            current_object_active_counts = Counter()
            for state in current_object_states.values():
                for object_name in state.get('memberships', ()):
                    current_object_active_counts[str(object_name)] += 1

            # 1. Retire bouts whose end_frame is now in the past. Cleanup is keyed
            #    on end_frame only, so overlapping bouts cannot accidentally pop
            #    each other (the previous match-by-behavior logic could leave a
            #    still-open outer bout invisible after an inner bout ended).
            for track_id in list(open_bouts_by_track.keys()):
                surviving = [
                    bout for bout in open_bouts_by_track[track_id]
                    if int(bout['end_frame']) >= current_frame_idx
                ]
                if surviving:
                    open_bouts_by_track[track_id] = surviving
                else:
                    open_bouts_by_track.pop(track_id, None)

            # 2. Open new bouts that start on this frame. We append rather than
            #    overwrite, so an inner bout starting mid-way through an outer
            #    bout doesn't drop the outer from the open list.
            for track_id, behavior, end_frame in start_events.get(current_frame_idx, []):
                open_bouts_by_track[track_id].append({
                    'behavior': behavior,
                    'start_frame': current_frame_idx,
                    'end_frame': int(end_frame),
                })
                track_counts = bout_counts.setdefault(track_id, {})
                track_counts.setdefault(behavior, 0)
                track_counts[behavior] += 1

            for roi_name, polygons_seq in normalized_rois.items():
                roi_color = _stable_overlay_color(roi_name, offset=0)
                for polygon in polygons_seq:
                    scaled_points = []
                    for point in polygon:
                        try:
                            px, py = point
                        except (TypeError, ValueError):
                            continue
                        scaled_points.append((int(px * scale_x), int(py * scale_y)))
                    if not scaled_points:
                        continue
                    pts = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=_OVERLAY_BORDER_COLOR, thickness=4)
                    cv2.polylines(frame, [pts], isClosed=True, color=roi_color, thickness=2)

            for roi_name, polygons_seq in normalized_object_rois.items():
                object_color = _stable_overlay_color(roi_name, offset=2)
                for polygon in polygons_seq:
                    scaled_points = []
                    for point in polygon:
                        try:
                            px, py = point
                        except (TypeError, ValueError):
                            continue
                        scaled_points.append((int(px * scale_x), int(py * scale_y)))
                    if not scaled_points:
                        continue
                    pts = np.array(scaled_points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=_OVERLAY_BORDER_COLOR, thickness=4)
                    cv2.polylines(frame, [pts], isClosed=True, color=object_color, thickness=2)

            collected_detections: List[pd.DataFrame] = []
            while detection_idx < detection_len and detection_schedule[detection_idx][0] < current_frame_idx:
                detection_idx += 1
            while detection_idx < detection_len and detection_schedule[detection_idx][0] == current_frame_idx:
                _, detection_path = detection_schedule[detection_idx]
                detection_df = _parse_detection_file(detection_path)
                if detection_df is not None and not detection_df.empty:
                    collected_detections.append(detection_df)
                detection_idx += 1
            detections_df = None
            if collected_detections:
                detections_df = (
                    collected_detections[0]
                    if len(collected_detections) == 1
                    else pd.concat(collected_detections, ignore_index=True)
                )

            if detections_df is not None and not detections_df.empty:
                for _, row in detections_df.iterrows():
                    track_val = row.get('track_id', 0)
                    try:
                        track_id = int(track_val)
                    except (TypeError, ValueError):
                        track_id = 0
                    x_center, y_center, w, h = row['x_center'], row['y_center'], row['w'], row['h']
                    if not np.all(np.isfinite([x_center, y_center, w, h])):
                        continue

                    x1 = int((x_center - w / 2) * target_video_width)
                    y1 = int((y_center - h / 2) * target_video_height)
                    x2 = int((x_center + w / 2) * target_video_width)
                    y2 = int((y_center + h / 2) * target_video_height)

                    x1 = max(0, min(x1, target_video_width - 1))
                    y1 = max(0, min(y1, target_video_height - 1))
                    x2 = max(x1 + 1, min(x2, target_video_width - 1))
                    y2 = max(y1 + 1, min(y2, target_video_height - 1))

                    object_state = current_object_states.get(track_id, {})
                    memberships = tuple(object_state.get('memberships', ()) or ())
                    interacting_object = str(object_state.get('roi_name', '') or '').strip()
                    if not interacting_object and memberships:
                        interacting_object = str(memberships[0])
                    primary_object_name = interacting_object or (str(memberships[0]) if memberships else '')
                    box_color = _stable_overlay_color(primary_object_name, offset=2) if primary_object_name else (80, 200, 80)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), _OVERLAY_BORDER_COLOR, 4)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if 0 <= bbox_center[0] < target_video_width and 0 <= bbox_center[1] < target_video_height:
                        star_size = max(6, min(10, int(min(x2 - x1, y2 - y1) * 0.18)))
                        draw_star(frame, bbox_center, color=(0, 0, 255), size=star_size)

                    label = f"ID {track_id}"
                    text_max_width = min(180, max(90, target_video_width - x1 - 12))
                    _draw_text_box(
                        frame,
                        label,
                        (x1, max(6, y1 - 28)),
                        box_color=box_color,
                        font_scale=overlay_meta_scale,
                        thickness=1,
                        max_width=text_max_width,
                        min_scale=0.35,
                    )

            current_track_specs: List[dict] = []
            if detections_df is not None and not detections_df.empty:
                seen_tracks = set()
                for _, det_row in detections_df.iterrows():
                    track_val = det_row.get('track_id', 0)
                    try:
                        track_id = int(track_val)
                    except (TypeError, ValueError):
                        track_id = 0
                    if track_id in seen_tracks:
                        continue
                    seen_tracks.add(track_id)
                    object_state = current_object_states.get(track_id, {})
                    memberships = tuple(object_state.get('memberships', ()) or ())
                    interaction_name = str(object_state.get('roi_name', '') or '').strip()
                    if not interaction_name and memberships:
                        interaction_name = str(memberships[0])
                    interaction_state = str(object_state.get('state', '') or '').strip()
                    distance_px = pd.to_numeric(object_state.get('distance_px', np.nan), errors='coerce')
                    object_bits = []
                    if interaction_name:
                        object_bits.append(f"Object: {interaction_name}")
                    elif memberships:
                        object_bits.append(f"Objects: {', '.join(memberships[:2])}")
                    if interaction_state:
                        object_bits.append(f"State: {interaction_state}")
                    if pd.notna(distance_px):
                        object_bits.append(f"Distance: {float(distance_px):.1f}px")
                    current_track_specs.append(
                        {
                            "track_id": track_id,
                            "behavior": _current_behavior_for_track(track_id),
                            "total_bouts": int(sum(bout_counts.get(track_id, {}).values())),
                            "object_text": " | ".join(object_bits),
                            "accent_color": _stable_overlay_color(interaction_name or str(track_id), offset=2)
                            if interaction_name or memberships
                            else (80, 200, 80),
                        }
                    )
            current_track_specs = sorted(current_track_specs, key=lambda item: int(item.get("track_id", 0)))

            sidebar_x = 12
            sidebar_max_width = max(80, sidebar_width - (sidebar_x * 2))
            dash_y = 18
            overview_card = _build_overview_card(
                current_frame_idx,
                visible_tracks=len(current_track_specs),
                active_objects=sum(current_object_active_counts.values()),
            )
            event_card = _build_event_card(
                roi_entries=list(entries_by_frame.get(current_frame_idx, [])),
                roi_exits=list(exits_by_frame.get(current_frame_idx, [])),
                object_entries=list(object_entries_by_frame.get(current_frame_idx, [])),
                object_exits=list(object_exits_by_frame.get(current_frame_idx, [])),
            )
            track_cards = _build_track_cards(current_track_specs)
            roi_cards = _build_roi_cards(roi_entry_counts, roi_exit_counts)
            object_cards = _build_object_cards(object_entry_counts, object_exit_counts, current_object_active_counts)

            dash_y = _draw_sidebar_card_grid(
                sidebar,
                [overview_card],
                x=sidebar_x,
                y=dash_y,
                width=sidebar_max_width,
                columns=1,
                title_scale=sidebar_header_scale,
                body_scale=sidebar_text_scale,
            )
            dash_y = _draw_sidebar_card_grid(
                sidebar,
                [event_card],
                x=sidebar_x,
                y=dash_y,
                width=sidebar_max_width,
                columns=1,
                title_scale=sidebar_header_scale,
                body_scale=sidebar_text_scale,
            )

            dash_y = _draw_sidebar_text(
                sidebar,
                "Tracks In View",
                x=sidebar_x,
                y=dash_y,
                max_width=sidebar_max_width,
                font_scale=sidebar_header_scale,
                color=_OVERLAY_TEXT_COLOR,
                thickness=2,
                max_lines=1,
                line_gap=6,
            )
            dash_y += 4
            dash_y = _draw_sidebar_card_grid(
                sidebar,
                track_cards,
                x=sidebar_x,
                y=dash_y,
                width=sidebar_max_width,
                columns=card_columns,
                title_scale=sidebar_header_scale,
                body_scale=sidebar_text_scale,
            )

            if roi_cards:
                dash_y = _draw_sidebar_text(
                    sidebar,
                    "ROI Metrics",
                    x=sidebar_x,
                    y=dash_y,
                    max_width=sidebar_max_width,
                    font_scale=sidebar_header_scale,
                    color=_OVERLAY_TEXT_COLOR,
                    thickness=2,
                    max_lines=1,
                    line_gap=6,
                )
                dash_y += 4
                dash_y = _draw_sidebar_card_grid(
                    sidebar,
                    roi_cards,
                    x=sidebar_x,
                    y=dash_y,
                    width=sidebar_max_width,
                    columns=card_columns,
                    title_scale=sidebar_text_scale,
                    body_scale=sidebar_text_scale,
                )

            if object_cards:
                dash_y = _draw_sidebar_text(
                    sidebar,
                    "Stimulus Objects",
                    x=sidebar_x,
                    y=dash_y,
                    max_width=sidebar_max_width,
                    font_scale=sidebar_header_scale,
                    color=_OVERLAY_TEXT_COLOR,
                    thickness=2,
                    max_lines=1,
                    line_gap=6,
                )
                dash_y += 4
                dash_y = _draw_sidebar_card_grid(
                    sidebar,
                    object_cards,
                    x=sidebar_x,
                    y=dash_y,
                    width=sidebar_max_width,
                    columns=card_columns,
                    title_scale=sidebar_text_scale,
                    body_scale=sidebar_text_scale,
                )

            _draw_sidebar_text(
                sidebar,
                "Review cue: verify event cards against the frame and cumulative ROI/object cards.",
                x=sidebar_x,
                y=min(total_height - 38, dash_y),
                max_width=sidebar_max_width,
                font_scale=overlay_meta_scale,
                color=(180, 180, 180),
                thickness=1,
                max_lines=2,
                line_gap=2,
            )

            if frame.shape[0] != sidebar.shape[0]:
                target_height = max(frame.shape[0], sidebar.shape[0])
                if frame.shape[0] < target_height:
                    frame = cv2.copyMakeBorder(
                        frame,
                        0,
                        target_height - frame.shape[0],
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                if sidebar.shape[0] < target_height:
                    sidebar = cv2.copyMakeBorder(
                        sidebar,
                        0,
                        target_height - sidebar.shape[0],
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )

            final_frame = np.hstack((frame, sidebar))
            writer.write(final_frame)
            frames_written += 1
            pbar.update(1)
            current_frame_idx += 1
    except Exception as exc:
        print(f"ERROR: Annotated video render failed: {exc}")
        return False
    finally:
        pbar.close()
        cap.release()
        writer.release()

    return frames_written > 0
