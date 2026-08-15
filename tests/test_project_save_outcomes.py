from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from integra_pose.gui.services.project_io_service import ProjectIOService
from integra_pose.utils.operation_result import OperationResult


def _app_with_result(result: OperationResult):
    return SimpleNamespace(
        root=object(),
        config=SimpleNamespace(
            save_project=MagicMock(return_value=result),
            save_project_as=MagicMock(return_value=result),
        ),
        update_status=MagicMock(),
        log_message=MagicMock(),
    )


def test_cancelled_save_does_not_report_success(monkeypatch) -> None:
    app = _app_with_result(OperationResult.cancel("cancelled"))
    showinfo = MagicMock()
    monkeypatch.setattr("integra_pose.gui.services.project_io_service.messagebox.showinfo", showinfo)

    ProjectIOService(app).save_project()

    showinfo.assert_not_called()
    app.update_status.assert_not_called()


def test_failed_save_does_not_report_success(monkeypatch) -> None:
    app = _app_with_result(OperationResult.failure("failed", error="disk full"))
    showinfo = MagicMock()
    monkeypatch.setattr("integra_pose.gui.services.project_io_service.messagebox.showinfo", showinfo)

    ProjectIOService(app).save_project_as()

    showinfo.assert_not_called()
    app.update_status.assert_not_called()


def test_successful_save_reports_success(monkeypatch) -> None:
    app = _app_with_result(OperationResult.success("saved"))
    showinfo = MagicMock()
    monkeypatch.setattr("integra_pose.gui.services.project_io_service.messagebox.showinfo", showinfo)

    ProjectIOService(app).save_project()

    app.update_status.assert_called_once_with("saved")
    showinfo.assert_called_once()
