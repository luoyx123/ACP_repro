"""实验室统一入口；运行 --help 查看各功能。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from acp.pipeline.cli import main
if __name__=='__main__':main()
