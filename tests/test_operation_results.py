from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

from integra_pose.logic.process_manager import ProcessManager
from integra_pose.utils.operation_result import OperationStatus
from integra_pose.utils.operation_result import OperationResult


class _ImmediateRoot:
    def after(self, _delay, callback):
        callback()


class _CompletedProcess:
    def __init__(self, returncode: int):
        self.stdout = io.StringIO("output\n")
        self._returncode = returncode

    def wait(self):
        return self._returncode


def _manager() -> ProcessManager:
    app = SimpleNamespace(
        root=_ImmediateRoot(),
        log_queue=SimpleNamespace(put=MagicMock()),
        log_message=MagicMock(),
    )
    return ProcessManager(app)


def test_process_callback_receives_success_only_for_zero_exit() -> None:
    manager = _manager()
    results = []

    manager._stream_process_output(_CompletedProcess(0), results.append)

    assert results[0].status is OperationStatus.SUCCESS
    assert results[0].returncode is None


def test_process_callback_receives_failure_for_nonzero_exit() -> None:
    manager = _manager()
    results = []

    manager._stream_process_output(_CompletedProcess(7), results.append)

    assert results[0].status is OperationStatus.FAILED
    assert results[0].returncode == 7


def test_partial_operation_is_distinct_from_success_and_failure() -> None:
    result = OperationResult.partial_success("some items failed", completed=2, failed=1)

    assert result.partial is True
    assert result.succeeded is False
    assert result.failed is False
    assert result.artifacts == {"completed": 2, "failed": 1}
