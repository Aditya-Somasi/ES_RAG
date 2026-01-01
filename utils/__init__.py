"""Utils package - configuration, logging, and token counting utilities."""

from .config import config
from .logging import setup_logger
from .token_counter import TokenCounter

# Backwards compatibility alias
setup_logging = setup_logger

__all__ = [
    "config",
    "setup_logger",
    "setup_logging",
    "TokenCounter",
]
