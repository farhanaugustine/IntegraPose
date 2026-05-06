import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _resolve_log_dir() -> Path:
    env_override = os.environ.get("INTEGRAPOSE_LOG_DIR")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    home_logs = Path.home() / ".integrapose" / "logs"
    candidates.append(home_logs)
    candidates.append(Path.cwd() / "integrapose_logs")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except Exception:
            continue
    return Path.cwd()


LOG_DIR = _resolve_log_dir()
LOG_PATH = LOG_DIR / "app.log"

_logger = None


def get_logger():
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger("integra_pose")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    _logger = logger
    return _logger
