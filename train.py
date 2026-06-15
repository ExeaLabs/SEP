#!/usr/bin/env python3
"""Legacy entry point — use main.py instead.

    python main.py --mode train --epochs 50
"""

import sys
from main import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--mode", "train"] + sys.argv[1:]
    main()
