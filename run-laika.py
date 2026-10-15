#!/usr/bin/env python3
"""Start Laika from a plain clone, no install step needed.

    sudo python3 run-laika.py              # on the robot
    python3 run-laika.py --sim             # on a laptop, no hardware
    python3 run-laika.py --help            # all options
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from laika.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
