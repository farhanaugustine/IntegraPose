from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from integra_pose.gui.services.inference_lifecycle_service import InferenceLifecycleService
from integra_pose.utils.operation_result import OperationStatus


class _ImmediateRoot:
    def after(self, _delay, callback):
        callback()


class _Runner:
    def __init__(self, error: Exception | None = None):
        self.stop_event = threading.Event()
        self._error = error

    def run(self):
        if self._error is not None:
            raise self._error


def _app():
    return SimpleNamespace(
        root=_ImmediateRoot(),
        log_message=MagicMock(),
        _set_process_activity=MagicMock(),
        _active_supervision_runner=None,
        _supervision_stop_event=None,
        _supervision_thread=None,
    )


def test_lifecycle_reports_success() -> None:
    app = _app()
    service = InferenceLifecycleService(app)
    finished = threading.Event()
    results = []

    assert service.start(_Runner(), lambda result: (results.append(result), finished.set()))
    assert finished.wait(2)
    assert results[0].status is OperationStatus.SUCCESS


def test_lifecycle_reports_runner_failure(monkeypatch) -> None:
    app = _app()
    service = InferenceLifecycleService(app)
    finished = threading.Event()
    results = []
    monkeypatch.setattr(
        "integra_pose.gui.services.inference_lifecycle_service.messagebox.showerror",
        MagicMock(),
    )

    assert service.start(
        _Runner(RuntimeError("model failed")),
        lambda result: (results.append(result), finished.set()),
    )
    assert finished.wait(2)
    assert results[0].status is OperationStatus.FAILED
    assert results[0].error == "model failed"
