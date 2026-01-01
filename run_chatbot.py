#!/usr/bin/env python
"""
Launcher script for RAG Chatbot UI.
Sets up PYTHONPATH automatically.
"""

import os
import sys
import subprocess

# Get project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Set PYTHONPATH
os.environ["PYTHONPATH"] = project_root

# Launch streamlit
ui_path = os.path.join(project_root, "app", "ui.py")
subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
