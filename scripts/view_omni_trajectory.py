"""旧回放命令兼容入口；推荐 acp_pipeline.py view <数据集>。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from acp.visualization.viewer import main
if __name__=='__main__':main()
