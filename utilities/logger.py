import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

_CONFIG_CACHE: Optional[dict] = None


def _load_configuration() -> dict:
    """Load configuration.yml located beside this module."""
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = Path(__file__).with_name("configuration.yml")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            _CONFIG_CACHE = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        _CONFIG_CACHE = {}
    except yaml.YAMLError:
        _CONFIG_CACHE = {}

    return _CONFIG_CACHE


def _resolve_level(level: Optional[str]) -> str:
    """Resolve effective log level using explicit value, env var, or configuration."""
    if level:
        return level

    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        return env_level

    configuration = _load_configuration()
    return configuration.get("simulation_log_level", "INFO")

class StructuredLogger:
    """
    Structured logging utility for consistent log formatting across the application.
    """
    
    def __init__(self, name: str = __name__, level: Optional[str] = None):
        self.logger = logging.getLogger(name)
        resolved_level = _resolve_level(level)
        numeric_level = getattr(logging, resolved_level.upper(), logging.INFO)

        self.logger.setLevel(numeric_level)
        self.level_name = resolved_level.upper()

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
            handler.setLevel(numeric_level)
            self.logger.addHandler(handler)
        else:
            for handler in self.logger.handlers:
                handler.setLevel(numeric_level)
    
    def _log_structured(self, level: str, message: str, **kwargs) -> None:
        """Create structured log entry with consistent format."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "component": kwargs.pop("component", "orchestrator"),
            **kwargs
        }
        
        # Use the tag format from existing code for compatibility
        tag = f"[{level}]"
        formatted_message = f"{tag} {json.dumps(log_entry, separators=(',', ':'))}"
        
        log_level = getattr(logging, level, logging.INFO) if isinstance(level, str) else level
        self.logger.log(log_level, formatted_message)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info level message."""
        self._log_structured("INFO", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error level message."""
        self._log_structured("ERROR", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message."""
        self._log_structured("WARNING", message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level message."""
        self._log_structured("DEBUG", message, **kwargs)
    
    def timer(self, message: str, **kwargs) -> None:
        """Log timer information."""
        # Use INFO level for timer messages but keep TIMER in the structured data
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "TIMER",
            "message": message,
            "component": kwargs.pop("component", "orchestrator"),
            **kwargs
        }
        
        tag = "[TIMER]"
        formatted_message = f"{tag} {json.dumps(log_entry, separators=(',', ':'))}"
        self.logger.log(logging.INFO, formatted_message)

def get_logger(name: str = __name__, level: Optional[str] = None) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name, level)


def bind_logger_to_print(logger: StructuredLogger):
    """Return a print-compatible function that routes messages through the logger."""

    level_map = {
        "ERROR": logger.error,
        "WARNING": logger.warning,
        "INFO": logger.info,
        "DEBUG": logger.debug,
        "TIMER": logger.timer,
        "SUCCESS": logger.info,
        "ACTION": logger.info,
        "PROGRESS": logger.info,
        "ESTIMATE": logger.info,
        "FAILURE": logger.error,
    }

    def _print(*args, **kwargs):
        if not args and not kwargs:
            logger.info("")
            return

        sep = kwargs.pop("sep", " ")
        end = kwargs.pop("end", "\n")
        file_target = kwargs.pop("file", None)
        kwargs.pop("flush", None)

        message = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            message = f"{message}{end}"

        message = message.rstrip("\n")

        if not message:
            logger.info("")
            return

        normalized = message.lstrip()
        match = re.match(r"\[(\w+)\]\s*(.*)", normalized)

        if match:
            tag = match.group(1).upper()
            remainder = match.group(2)
            handler = level_map.get(tag, logger.info)
            payload = remainder if remainder else message
            handler(payload, tag=tag, raw=message)
        else:
            if file_target is sys.stderr:
                logger.error(message, tag="STDERR", raw=message)
            else:
                logger.info(message)

    return _print


# Global logger instance (module-level fallback)
logger = StructuredLogger()
