"""utils package initializer.

This package exposes a compatibility layer so code that does
``from utils import setup_logging, clean_text, ...`` keeps working.

Behavior:
- Prefer the package-local logger implementation in ``utils/logging.py``
  and expose it under the historical name ``setup_logging`` (some code
  expects that name).
- For other generic helpers (``clean_text``, ``count_words``,
  ``get_file_size``, ``cleanup_temp_files``) we load them from the
  project's top-level ``utils.py`` so existing imports continue to work
  when both a top-level module and a package exist.

Long-term recommendation: consolidate the helpers into the package and
remove/rename the top-level ``utils.py`` to avoid this ambiguity.
"""

from importlib import util
from pathlib import Path

# 1) Prefer the package-local structured logger (utils/logging.py).
try:
	from .logging import setup_logger as _pkg_setup_logger
except Exception:
	_pkg_setup_logger = None


# Provide a small delegating `setup_logging` function so callers (e.g.
# the Elasticsearch UI in `app.py`) get the project's preferred logging
# style. We prefer the top-level `utils.py` implementation if available
# (keeps the ES-specific logger format), otherwise fall back to the
# package-local structured logger, and finally a minimal stdlib logger.
def setup_logging(name: str, *args, **kwargs):
	# Try lazy root utils delegate first (this will only execute the
	# top-level module when a logger is actually requested).
	try:
		root = _load_root_utils()
		root_fn = getattr(root, "setup_logging", None)
		if callable(root_fn):
			return root_fn(name, *args, **kwargs)
	except Exception:
		# If loading root utils fails (missing deps, etc.), continue to
		# other fallbacks.
		pass

	# Next, prefer the package-local logger implementation
	if _pkg_setup_logger is not None:
		return _pkg_setup_logger(name, *args, **kwargs)

	# Last-resort: basic stdlib logger to avoid raising at import-time
	import logging as _logging
	logger = _logging.getLogger(name)
	if not logger.handlers:
		handler = _logging.StreamHandler()
		logger.addHandler(handler)
	return logger

# Provide small, safe local implementations of commonly used helpers so
# importing the package does not execute the top-level `utils.py` (which
# may import heavy optional dependencies like `elasticsearch`). These
# implementations match the behavior the rest of the project expects and
# avoid side-effects at import time.
import os
import re
from typing import List


def clean_text(text: str) -> str:
	"""Normalize and clean text (lightweight, same behavior as the
	original helper used across the project).
	"""
	if not text:
		return ""

	text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
	text = re.sub(r"[\u00ad\uFFFE\uFFFF￾]", "", text)
	text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
	text = re.sub(r"\s+", " ", text)

	return text.strip()


def count_words(text: str) -> int:
	return len(text.split()) if text else 0


def get_file_size(filepath: str) -> int:
	try:
		return os.path.getsize(filepath)
	except Exception:
		return 0


def cleanup_temp_files(directory: str) -> None:
	if not os.path.exists(directory):
		return
	try:
		for item in os.listdir(directory):
			item_path = os.path.join(directory, item)
			try:
				if os.path.isfile(item_path):
					os.remove(item_path)
				elif os.path.isdir(item_path):
					import shutil

					shutil.rmtree(item_path)
			except Exception:
				# Best-effort cleanup; avoid raising from import-time helper
				continue
	except Exception:
		return

__all__ = [
	"setup_logging",
	"clean_text",
	"count_words",
	"get_file_size",
	"cleanup_temp_files",
]

# Lazy loader for the project's top-level utils.py so we can delegate
# heavy or ES-related helpers without importing them at package import
# time. This avoids early import-time side effects and missing-dependency
# errors when the venv isn't active.
_root_utils_module = None


def _load_root_utils():
	"""Load and cache the top-level utils.py module from the project root.
	Returns the module object. Raises any import errors from executing
	that module so callers see the original failure.
	"""
	global _root_utils_module
	if _root_utils_module is not None:
		return _root_utils_module

	root_path = Path(__file__).resolve().parent.parent / "utils.py"
	if not root_path.exists():
		raise ImportError(f"Top-level utils.py not found at {root_path}")

	spec = util.spec_from_file_location("_root_utils", str(root_path))
	module = util.module_from_spec(spec)
	# Execute the module (this may import heavy deps like `elasticsearch`)
	spec.loader.exec_module(module)  # type: ignore
	_root_utils_module = module
	return _root_utils_module


# Delegate wrappers -----------------------------------------------------
def get_es_client(*args, **kwargs):
	return _load_root_utils().get_es_client(*args, **kwargs)


def bulk_index_documents(*args, **kwargs):
	return _load_root_utils().bulk_index_documents(*args, **kwargs)


def get_document_count(*args, **kwargs):
	return _load_root_utils().get_document_count(*args, **kwargs)


def print_progress(*args, **kwargs):
	return _load_root_utils().print_progress(*args, **kwargs)


def preprocess_query(*args, **kwargs):
	return _load_root_utils().preprocess_query(*args, **kwargs)


# Export the delegated names as well
__all__.extend([
	"get_es_client",
	"bulk_index_documents",
	"get_document_count",
	"print_progress",
	"preprocess_query",
])


