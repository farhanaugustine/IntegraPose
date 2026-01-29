import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "BehaviorScope",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a logger with a consistent format for console and optional file output.
    
    Args:
        name: The name of the logger (e.g., 'Trainer', 'Inference').
        log_file: Optional path to a log file.
        level: Logging level (default: INFO).
        
    Returns:
        configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers are already added to avoid duplicates
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
