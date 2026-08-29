#!/usr/bin/env python
"""
Quick start script to verify project setup and run initial exploration.
Run this after activating the virtual environment to check everything works.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Verify Python 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_data_file():
    """Check if creditcard.csv exists"""
    data_path = Path('data/creditcard.csv')
    if not data_path.exists():
        print(f"❌ Missing {data_path}")
        return False
    
    size_mb = data_path.stat().st_size / (1024*1024)
    print(f"✓ Found {data_path} ({size_mb:.1f} MB)")
    return True

def check_imports():
    """Verify all required packages are installed"""
    required = [
        'pandas', 'numpy', 'sklearn', 'imblearn', 'xgboost',
        'shap', 'matplotlib', 'seaborn', 'jupyter'
    ]
    
    all_ok = True
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} - run: pip install -r requirements.txt")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 70)
    print("CREDIT CARD FRAUD DETECTION - PROJECT SETUP CHECK")
    print("=" * 70)
    
    print("\n1. Python Version:")
    py_ok = check_python_version()
    
    print("\n2. Data Files:")
    data_ok = check_data_file()
    
    print("\n3. Required Packages:")
    imports_ok = check_imports()
    
    print("\n" + "=" * 70)
    if py_ok and data_ok and imports_ok:
        print("✓ All checks passed! Project is ready.")
        print("\nNext steps:")
        print("  1. cd notebooks")
        print("  2. jupyter notebook")
        print("  3. Open 01_exploration.ipynb")
    else:
        print("❌ Some checks failed. See above for details.")
        sys.exit(1)
    
    print("=" * 70)

if __name__ == '__main__':
    main()
