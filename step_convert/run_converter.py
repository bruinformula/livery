#!/usr/bin/env python3
"""
STEP to USD Converter - Standalone Script

This script can be run directly to convert STEP files to USD format.
It imports the modular package components.
"""

import sys
import os

# Add the current directory to the Python path to allow importing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    step_file = "./Mk10.STEP"
    # step_file = "./test_files/nist_stc_10_asme1_ap242-e2.stp"
    usd_file = "output.usda"

    try:
        main(step_file, usd_file)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Check the STEP file path and try again.")
