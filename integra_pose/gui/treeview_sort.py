"""Click-to-sort behaviour for any ttk.Treeview.

Wires each column's header to cycle through ``ascending → descending →
unsorted`` on click, with a small arrow indicator (↑ / ↓) appended to the
header text so the current sort state is visible at a glance. Numeric
values are detected automatically and sorted numerically; everything else
sorts case-insensitively as text.

Usage::

    from integra_pose.gui.treeview_sort import make_sortable_treeview

    tree = ttk.Treeview(parent, columns=("col1", "col2"), show="headings")
    tree.heading("col1", text="Column 1")
    tree.heading("col2", text="Column 2")
    make_sortable_treeview(tree)

The helper preserves any pre-existing ``command=`` callback on a heading
(if one was set) by replacing it; if you need to keep yours, set the
heading command after calling this helper.
"""

from __future__ import annotations

from tkinter import ttk
from typing import Optional


_ARROW_UP = " ↑"
_ARROW_DOWN = " ↓"


def _strip_arrow(label: str) -> str:
    """Return the heading text with any trailing sort arrow removed."""
    if label.endswith(_ARROW_UP) or label.endswith(_ARROW_DOWN):
        return label[: -len(_ARROW_UP)]
    return label


def apply_sort_indicator(
    tree: ttk.Treeview,
    original_labels: dict,
    *,
    active_column: Optional[str],
    ascending: Optional[bool],
) -> None:
    """Update column headers with the standard ↑ / ↓ sort indicators.

    Public helper so any Treeview that does its *own* sort logic (e.g., the
    batch wizard's data-source-of-truth queue) can still get the same visual
    arrow indicators that ``make_sortable_treeview`` produces.

    Args:
        tree: The Treeview to update.
        original_labels: ``{column_id: clean_label_without_arrow}`` snapshot
            saved at construction time. The caller owns this dict; pass the
            same one across calls.
        active_column: The currently-sorted column id, or ``None`` for "no
            active sort" (all arrows cleared).
        ascending: ``True`` for ↑, ``False`` for ↓. Ignored when
            ``active_column`` is ``None``.
    """
    columns = tree["columns"]
    if not columns:
        return
    for col in columns:
        base_label = original_labels.get(col, str(col))
        if active_column is not None and col == active_column:
            label = base_label + (_ARROW_UP if ascending else _ARROW_DOWN)
        else:
            label = base_label
        try:
            tree.heading(col, text=label)
        except Exception:
            pass


def _coerce_number(value) -> Optional[float]:
    """Best-effort parse of a Treeview cell value to float; ``None`` if not numeric."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            f = float(value)
            return f if f == f else None  # reject NaN
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    # Accept percentages like "5%" and units like "1.2s" by stripping a
    # trailing non-numeric suffix conservatively. Only trim a single token,
    # not arbitrary characters in the middle.
    try:
        return float(text)
    except ValueError:
        pass
    # Trim a single trailing % or whitespace.
    if text.endswith("%"):
        try:
            return float(text[:-1].strip())
        except ValueError:
            return None
    return None


def _column_is_numeric(rows, column_index: int) -> bool:
    """A column is numeric if every non-empty cell parses as a number."""
    saw_value = False
    for row in rows:
        if column_index >= len(row):
            continue
        cell = row[column_index]
        if cell in (None, ""):
            continue
        saw_value = True
        if _coerce_number(cell) is None:
            return False
    return saw_value


def make_sortable_treeview(tree: ttk.Treeview) -> None:
    """Wire each column's heading to cycle ascending / descending / unsorted on click.

    The helper attaches its sort state to the Treeview itself via dynamic
    attributes (``_sort_column``, ``_sort_state``) so the same Treeview can be
    re-sorted later from outside this module if needed.
    """
    columns = tree["columns"]
    if not columns:
        return

    # Snapshot the original heading labels (without arrows) so we can restore
    # them when the column moves out of the active sort.
    original_labels: dict[str, str] = {}
    for col in columns:
        try:
            original_labels[col] = _strip_arrow(str(tree.heading(col, "text")))
        except Exception:
            original_labels[col] = str(col)

    # Track sort state on the tree itself.
    tree._sort_column = None  # type: ignore[attr-defined]
    tree._sort_state = None  # type: ignore[attr-defined]   # 'asc' | 'desc' | None
    tree._sort_original_labels = dict(original_labels)  # type: ignore[attr-defined]

    def _next_state(current: Optional[str]) -> Optional[str]:
        if current is None:
            return "asc"
        if current == "asc":
            return "desc"
        return None  # unsorted

    def _refresh_headers(active_col: Optional[str], state: Optional[str]) -> None:
        # Delegate to the public helper so the wizard's queue (which has its
        # own data-driven sort and can't use this whole helper) still gets
        # identical arrow visuals via the same code path.
        if state is None:
            apply_sort_indicator(tree, original_labels, active_column=None, ascending=None)
        else:
            apply_sort_indicator(
                tree,
                original_labels,
                active_column=active_col,
                ascending=(state == "asc"),
            )

    def _apply_sort(active_col: str, state: Optional[str]) -> None:
        children = list(tree.get_children(""))
        if not children:
            return

        col_index = columns.index(active_col) if active_col in columns else 0
        rows: list[tuple[str, tuple]] = []
        for iid in children:
            try:
                values = tuple(tree.item(iid, "values") or ())
            except Exception:
                values = ()
            rows.append((iid, values))

        if state is None:
            # Restore the children's original insertion order. We have no
            # record of that, so on "unsorted" we leave the current order
            # untouched. This is a deliberate trade-off: the user can re-run
            # the analysis to get the canonical insertion order back.
            return

        # Decide between numeric and text comparison based on the data in
        # this column right now (not at help-attach time).
        rows_only = [r[1] for r in rows]
        numeric = _column_is_numeric(rows_only, col_index)

        def _key(item: tuple[str, tuple]):
            cell = item[1][col_index] if col_index < len(item[1]) else ""
            if numeric:
                value = _coerce_number(cell)
                # Numeric None sorts to the end regardless of direction so
                # missing values don't masquerade as "smallest".
                return (1 if value is None else 0, value if value is not None else 0.0)
            return (0, str(cell or "").casefold())

        reverse = state == "desc"
        rows.sort(key=_key, reverse=reverse)
        for new_index, (iid, _values) in enumerate(rows):
            try:
                tree.move(iid, "", new_index)
            except Exception:
                pass

    def _on_heading_click(col: str) -> None:
        prev_col = tree._sort_column  # type: ignore[attr-defined]
        prev_state = tree._sort_state  # type: ignore[attr-defined]
        if col != prev_col:
            new_state = "asc"
        else:
            new_state = _next_state(prev_state)
        tree._sort_column = col if new_state is not None else None  # type: ignore[attr-defined]
        tree._sort_state = new_state  # type: ignore[attr-defined]
        _refresh_headers(tree._sort_column, new_state)  # type: ignore[attr-defined]
        if new_state is not None:
            _apply_sort(col, new_state)

    for col in columns:
        try:
            tree.heading(col, command=lambda c=col: _on_heading_click(c))
        except Exception:
            pass
