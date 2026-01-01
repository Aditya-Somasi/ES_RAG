"""
Setup script for Elasticsearch Multi-Document Search System
Creates necessary directories and validates configuration
"""

import os
import sys
from config import (
    create_data_directories,
    validate_config,
    print_config
)

def setup():
    print("\n" + "="*70)
    print("ELASTICSEARCH MULTI-DOCUMENT SEARCH SYSTEM - SETUP")
    print("="*70)
    
    print("\n1. Creating data directories...")
    create_data_directories()
    print("   ✅ Data directories created")
    
    print("\n2. Printing configuration...")
    print_config()
    
    print("\n3. Validating configuration...")
    if validate_config():
        print("   ✅ Configuration is valid")
    else:
        print("   ⚠️ Configuration has warnings (non-critical)")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Start Elasticsearch server")
    print("2. Run: streamlit run app.py")
    print("3. Or run pipeline directly: python pipeline.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    setup()