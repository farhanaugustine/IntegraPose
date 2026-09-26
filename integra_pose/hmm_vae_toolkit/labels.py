"""Cluster identity helpers for current and legacy bout records."""

import re


def parse_bout_label(bout):
    state = str(bout.get("state", "")).strip()
    if re.fullmatch(r"\d+:\d+", state):
        return tuple(int(part) for part in state.split(":"))
    parent = str(bout.get("class_id", "")).strip()
    if re.fullmatch(r"\d+", state) and re.fullmatch(r"\d+", parent):
        return int(parent), int(state)
    return None
