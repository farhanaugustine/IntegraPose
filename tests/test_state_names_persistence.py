"""Tests for ``state_names.json`` run-ID isolation.

Cluster identifiers can change when parameters change. A name assigned to
``0:1`` in one run must not be applied to ``0:1`` in another run. The tests
verify that names are loaded only when run IDs match.

Pure-logic tests against the module-level ``load_state_names`` helper —
no Tk required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from integra_pose.hmm_vae_toolkit.naming_dialog import (
    STATE_NAMES_FILENAME,
    load_state_names,
)


class TestLoadStateNames(unittest.TestCase):
    def _write(self, folder: str, payload: dict) -> str:
        path = os.path.join(folder, STATE_NAMES_FILENAME)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    def test_no_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(load_state_names(td), {})
            self.assertEqual(load_state_names(td, "any-run-id"), {})

    def test_empty_folder_returns_empty(self) -> None:
        self.assertEqual(load_state_names("", "run-x"), {})

    def test_loads_when_run_id_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write(td, {
                "schema_version": 2,
                "run_id": "abc-123",
                "state_names": {"0:1": "forward-locomotion", "0:2": "rear"},
            })
            names = load_state_names(td, "abc-123")
            self.assertEqual(names, {"0:1": "forward-locomotion", "0:2": "rear"})

    def test_returns_empty_on_run_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write(td, {
                "schema_version": 2,
                "run_id": "old-uuid",
                "state_names": {"0:1": "this-name-belongs-to-prior-run"},
            })
            names = load_state_names(td, "new-uuid")
            self.assertEqual(names, {})

    def test_rejects_v1_schema_for_a_known_new_run(self) -> None:
        # Schema v1 has no provenance. Applying its cluster names to a known
        # new run could silently relabel different sub-behaviors.
        with tempfile.TemporaryDirectory() as td:
            self._write(td, {
                "schema_version": 1,
                "state_names": {"0:1": "rear-up"},
            })
            names = load_state_names(td, "any-uuid")
            self.assertEqual(names, {})

    def test_loads_when_caller_has_no_run_id(self) -> None:
        # If the caller doesn't know the current run_id, loader returns
        # whatever the file has. (Better than refusing to load anything.)
        with tempfile.TemporaryDirectory() as td:
            self._write(td, {
                "run_id": "some-uuid",
                "state_names": {"1:0": "groom"},
            })
            self.assertEqual(load_state_names(td, ""), {"1:0": "groom"})
            self.assertEqual(load_state_names(td, None), {"1:0": "groom"})

    def test_rejects_blank_file_run_id_for_a_known_new_run(self) -> None:
        # A blank file run id is also unprovenanced and must not be applied
        # automatically when the caller knows the current run.
        with tempfile.TemporaryDirectory() as td:
            self._write(td, {
                "run_id": "",
                "state_names": {"0:1": "saved-without-stamp"},
            })
            self.assertEqual(load_state_names(td, "current-uuid"), {})

    def test_corrupt_json_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, STATE_NAMES_FILENAME)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            self.assertEqual(load_state_names(td, "any"), {})

    def test_state_names_not_dict_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._write(td, {
                "run_id": "uuid",
                "state_names": ["not", "a", "dict"],
            })
            self.assertEqual(load_state_names(td, "uuid"), {})

    def test_string_coercion_of_keys_and_values(self) -> None:
        # If somehow ints land in the dict, coerce to strings so callers
        # can rely on a uniform type.
        with tempfile.TemporaryDirectory() as td:
            self._write(td, {
                "run_id": "u",
                "state_names": {0: 1, "1:0": 42},
            })
            names = load_state_names(td, "u")
            # str keys, str values
            for k, v in names.items():
                self.assertIsInstance(k, str)
                self.assertIsInstance(v, str)


if __name__ == "__main__":
    unittest.main()
