"""Shared outcome type for user-visible operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperationStatus(str, Enum):
    SUCCESS = "success"
    CANCELLED = "cancelled"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class OperationResult:
    status: OperationStatus
    message: str = ""
    error: str = ""
    returncode: int | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status is OperationStatus.SUCCESS

    @property
    def cancelled(self) -> bool:
        return self.status is OperationStatus.CANCELLED

    @property
    def partial(self) -> bool:
        return self.status is OperationStatus.PARTIAL

    @property
    def failed(self) -> bool:
        return self.status is OperationStatus.FAILED

    @classmethod
    def success(cls, message: str = "", **artifacts: Any) -> "OperationResult":
        return cls(OperationStatus.SUCCESS, message=message, artifacts=dict(artifacts))

    @classmethod
    def cancel(cls, message: str = "") -> "OperationResult":
        return cls(OperationStatus.CANCELLED, message=message)

    @classmethod
    def partial_success(cls, message: str = "", **artifacts: Any) -> "OperationResult":
        return cls(OperationStatus.PARTIAL, message=message, artifacts=dict(artifacts))

    @classmethod
    def failure(
        cls,
        message: str,
        *,
        error: str = "",
        returncode: int | None = None,
    ) -> "OperationResult":
        return cls(
            OperationStatus.FAILED,
            message=message,
            error=error,
            returncode=returncode,
        )
