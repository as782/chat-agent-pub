"""测试配置文件。

负责补齐测试运行时的项目根目录导入路径。
当前阶段不负责共享夹具和复杂测试环境编排。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
