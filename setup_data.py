"""
Script de inicializacao de dados.
Execute este script para popular o banco SQLite antes de rodar o dashboard.
No Streamlit Cloud, este script e chamado automaticamente se o banco nao existir.
"""

from yahoo_finance_extractor import executar_extracao_completa
from pathlib import Path

DB_PATH = Path(__file__).parent / "yahoo_finance.db"

if __name__ == "__main__":
    print("Populando banco de dados...")
    executar_extracao_completa()
    print(f"\nBanco criado em: {DB_PATH}")
    print(f"Tamanho: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
