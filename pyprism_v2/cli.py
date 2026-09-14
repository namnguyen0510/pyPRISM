"""Console entry point: ``pyprism-benchmark`` forwards to run_benchmark."""
from __future__ import annotations
import os
import runpy
import sys


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(here, 'run_benchmark.py')
    if not os.path.exists(script):
        sys.stderr.write('run_benchmark.py not found next to the package\n')
        return 2
    sys.argv[0] = script
    runpy.run_path(script, run_name='__main__')
    return 0


if __name__ == '__main__':
    sys.exit(main())
