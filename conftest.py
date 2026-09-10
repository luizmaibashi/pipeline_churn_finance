"""Raiz do projeto no sys.path para os testes: o código vive em `src/` e é
importado como pacote `src.*`. Rodar sempre da raiz: `python -m pytest -q`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
