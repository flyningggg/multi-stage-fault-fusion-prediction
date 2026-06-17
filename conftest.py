# -*- coding: utf-8 -*-
"""根 conftest：把 program/ 注入 sys.path，使 `import percolation` 等可用。
不修改任何业务模块。"""
import os
import sys

_PROG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "program")
if _PROG not in sys.path:
    sys.path.insert(0, _PROG)
