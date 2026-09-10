import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_paridade_javascript_python():
    subprocess.run([sys.executable, "tools/export_model.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tests/paridade/gerar_casos.py"], cwd=ROOT, check=True)
    subprocess.run(["node", "tests/paridade/parity.test.mjs"], cwd=ROOT, check=True)
