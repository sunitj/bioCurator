"""Structured JSON logging implementation."""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any

import structlog
from pythonjsonlogger import jsonlogger

from .context import get_current_context


class ContextInjector:
    """Processor to inject logging context into log records."""

    def __call__(self, _logger, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Inject current context into log event."""
        context = get_current_context()
        context_dict = context.to_dict()

        # Only add non-None context values
        for key, value in context_dict.items():
            if value is not None:
                event_dict[key] = value

        return event_dict


class ErrorHandler:
    """Structured error handler for consistent error logging."""

    def __call__(self, logger, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Process error information in log events."""
        # Extract exception information if present
        exception = event_dict.pop("exception", None)
        if exception:
            event_dict["error"] = {
                "type": exception.__class__.__name__,
                "message": str(exception),
                "module": getattr(exception.__class__, "__module__", None),
            }

            # Add traceback in debug mode
            if logger.level <= logging.DEBUG:
                import traceback

                event_dict["error"]["traceback"] = traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )

        return event_dict


def setup_logging(
    log_level: str = "INFO",
    log_path: str | None = None,
    enable_json: bool = True,
    enable_console: bool = True,
) -> None:
    """
    Set up structured logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_path: Optional path for log files
        enable_json: Whether to use JSON formatting
        enable_console: Whether to log to console
    """
    # Clear any existing configuration
    logging.getLogger().handlers.clear()

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        ContextInjector(),
        ErrorHandler(),
        structlog.processors.UnicodeDecoder(),
    ]

    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handlers = []

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        if enable_json:
            console_formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            console_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    if log_path:
        # Ensure log directory exists
        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Application log file
        app_log_file = log_dir / "biocurator.log"
        file_handler = logging.FileHandler(app_log_file)

        if enable_json:
            file_formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

        # Error log file (errors and above)
        error_log_file = log_dir / "biocurator-errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        handlers.append(error_handler)

    # Configure root logger
    logging.basicConfig(level=getattr(logging, log_level.upper()), handlers=handlers, force=True)

    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


# Default logger configuration
def configure_default_logging():
    """Configure default logging based on environment."""
    try:
        from ..config import settings

        setup_logging(
            log_level=settings.log_level,
            log_path=settings.paths.log_path if settings.paths.log_path != "/logs" else None,
            enable_json=True,
            enable_console=True,
        )
    except ImportError:
        # Fallback configuration if settings not available
        setup_logging(
            log_level="INFO",
            log_path=None,
            enable_json=True,
            enable_console=True,
        )


# Configure logging on module import
configure_default_logging()
