#!/usr/bin/env python3
"""
Command-line interface for the pep2prob package.
"""

import sys
import os

def main():
    """Main CLI entry point that delegates to run.py"""
    try:
        # Import and run the main function from run.py in the same package
        from pep2prob.run import main as run_main
        run_main()
    except ImportError as e:
        print(f"Error importing run module: {e}")
        print("Make sure the package is properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running pep2prob: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
