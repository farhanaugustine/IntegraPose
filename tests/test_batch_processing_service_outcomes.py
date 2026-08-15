from __future__ import annotations

import threading
from types import SimpleNamespace

from integra_pose.gui.services.batch_processing_service import BatchProcessingService
from integra_pose.logic.batch_pipeline import BatchRunResult
from integra_pose.utils.operation_result import OperationStatus


class _ImmediateRoot:
    def after(self, _delay, callback) -> None:
        callback()


class _App:
    def __init__(self) -> None:
        self.root = _ImmediateRoot()


def test_service_preserves_legacy_callback_and_adds_explicit_partial_status() -> None:
    service = BatchProcessingService(_App())
    service.pipeline = SimpleNamespace(
        run=lambda *_args, **_kwargs: BatchRunResult(
            status=OperationStatus.PARTIAL,
            message="One video failed.",
            workbook_path="batch.xlsx",
            video_results=[{"video_id": "ok"}],
            total_count=2,
            completed_count=1,
            failed_count=1,
        )
    )
    finished = threading.Event()
    calls: list[tuple] = []

    def _finish(session, cancelled, message, payload) -> None:
        calls.append((session, cancelled, message, payload))
        finished.set()

    session = SimpleNamespace(videos=[])
    assert service.start_batch_run(session, on_finish=_finish) is True
    assert finished.wait(timeout=2.0)

    assert len(calls) == 1
    _session, cancelled, message, payload = calls[0]
    assert cancelled is False
    assert message == "One video failed."
    assert payload["status"] == "partial"
    assert payload["completed_count"] == 1
    assert payload["failed_count"] == 1
    assert service.last_run_result == payload


def test_service_exception_callback_is_explicit_failure_not_success() -> None:
    service = BatchProcessingService(_App())

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    service.pipeline = SimpleNamespace(run=_raise)
    finished = threading.Event()
    calls: list[tuple] = []

    def _finish(*args) -> None:
        calls.append(args)
        finished.set()

    session = SimpleNamespace(videos=[SimpleNamespace(excluded=False)])
    assert service.start_batch_run(session, on_finish=_finish) is True
    assert finished.wait(timeout=2.0)

    _session, cancelled, message, payload = calls[0]
    assert cancelled is False
    assert message == "Batch run failed: boom"
    assert payload["status"] == "failed"
    assert payload["failed_count"] == 1
    assert service.last_run_result == payload


def test_service_runs_reviewed_result_finalization_with_explicit_result_kind() -> None:
    service = BatchProcessingService(_App())
    service.pipeline = SimpleNamespace(
        finalize_reviewed_results=lambda *_args, **_kwargs: BatchRunResult(
            status=OperationStatus.SUCCESS,
            message="Finalized.",
            workbook_path="batch.xlsx",
            session_json_path="batch_session.json",
            total_count=2,
            completed_count=2,
        )
    )
    finished = threading.Event()
    calls: list[tuple] = []

    def _finish(*args) -> None:
        calls.append(args)
        finished.set()

    session = SimpleNamespace(videos=[])
    assert service.start_finalize_reviewed_results(
        session, on_finish=_finish
    ) is True
    assert finished.wait(timeout=2.0)

    _session, cancelled, message, payload = calls[0]
    assert cancelled is False
    assert message == "Finalized."
    assert payload["status"] == "success"
    assert payload["result_kind"] == "finalization"
    assert payload["completed_count"] == 2
