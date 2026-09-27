"""arm_bridge.py - Import the arm IR the way arm's own modules do.

arm/ is a directory of flat scripts that import each other as `import ir`,
not `import arm.ir`. Importing it as a package would create a second copy of
every node class, and an `ir.Bin` from `sexpr.parse` would no longer be an
instance of `arm.ir.Bin`. Putting arm/ on the path and importing it flat keeps
exactly one copy, shared with sexpr.py, grammar.py and backends.py.
"""

import os
import sys

ARM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'arm')
if ARM_DIR not in sys.path:
    sys.path.insert(0, ARM_DIR)

import ir as arm_ir          # noqa: E402
import sexpr as arm_sexpr    # noqa: E402

__all__ = ['arm_ir', 'arm_sexpr']
