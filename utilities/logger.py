import json
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class StructuredLogger:
    """
    Structured logging utility for consistent log formatting across the application.
    """
    
    def __init__(self, name: str = __name__, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Create console handler if not already exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(getattr(logging, level.upper()))
            self.logger.addHandler(handler)
    
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
        
        self.logger.log(getattr(logging, level), formatted_message)
    
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

# Global logger instance
logger = StructuredLogger()

def get_logger(name: str = __name__, level: str = "INFO") -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name, level)
