"""
Yahoo Finance Data Extractor
Extrai cotações históricas, dados fundamentalistas e dividendos
de ações brasileiras e americanas, armazenando em SQLite.
"""

import yfinance as yf
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO - Edite aqui suas ações
# ============================================================

# Ações brasileiras (adicione o sufixo .SA)
ACOES_BR = [
    "VALE3.SA",
    "PETR4.SA",
    "PETR3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "BBAS3.SA",
    "ABEV3.SA",
    "WEGE3.SA",
    "RENT3.SA",
    "SUZB3.SA",
    "B3SA3.SA",
    "ITSA4.SA",
    "SANB11.SA",
    "EQTL3.SA",
    "CSAN3.SA",
    "PRIO3.SA",
    "RADL3.SA",
    "RAIL3.SA",
    "VIVT3.SA",
    "HAPV3.SA",
    "ENEV3.SA",
]

# Ações americanas
ACOES_US = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
]

# Todas as ações
TODAS_ACOES = ACOES_BR + ACOES_US

# Período padrão para cotações históricas
PERIODO_ANOS = 5

# Caminho do banco de dados
DB_PATH = Path(__file__).parent / "yahoo_finance.db"


# ============================================================
# BANCO DE DADOS
# ============================================================

def criar_banco():
    """Cria as tabelas no SQLite se não existirem."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cotacoes (
            ticker TEXT NOT NULL,
            data DATE NOT NULL,
            abertura REAL,
            maxima REAL,
            minima REAL,
            fechamento REAL,
            fechamento_ajustado REAL,
            volume INTEGER,
            mercado TEXT,
            atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, data)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fundamentalistas (
            ticker TEXT NOT NULL,
            indicador TEXT NOT NULL,
            valor TEXT,
            mercado TEXT,
            atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, indicador)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dividendos (
            ticker TEXT NOT NULL,
            data DATE NOT NULL,
            valor REAL,
            mercado TEXT,
            atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, data)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financeiros_trimestrais (
            ticker TEXT NOT NULL,
            data DATE NOT NULL,
            indicador TEXT NOT NULL,
            valor REAL,
            tipo TEXT,
            mercado TEXT,
            atualizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, data, indicador)
        )
    """)

    conn.commit()
    conn.close()
    print("Banco de dados criado/verificado com sucesso.")


# ============================================================
# EXTRAÇÃO DE COTAÇÕES HISTÓRICAS
# ============================================================

def extrair_cotacoes(tickers: list, periodo_anos: int = PERIODO_ANOS):
    """Extrai cotações históricas e salva no SQLite."""
    conn = sqlite3.connect(DB_PATH)
    data_inicio = datetime.now() - timedelta(days=periodo_anos * 365)
    total_registros = 0

    for ticker in tickers:
        try:
            print(f"  Baixando cotações: {ticker}...", end=" ")
            acao = yf.Ticker(ticker)
            hist = acao.history(start=data_inicio.strftime("%Y-%m-%d"))

            if hist.empty:
                print("Sem dados.")
                continue

            mercado = "BR" if ticker.endswith(".SA") else "US"

            df = pd.DataFrame({
                "ticker": ticker,
                "data": hist.index.strftime("%Y-%m-%d"),
                "abertura": hist["Open"].values,
                "maxima": hist["High"].values,
                "minima": hist["Low"].values,
                "fechamento": hist["Close"].values,
                "fechamento_ajustado": hist["Close"].values,
                "volume": hist["Volume"].values.astype(int),
                "mercado": mercado,
            })

            df.to_sql("cotacoes", conn, if_exists="append", index=False,
                       method="multi")
            total_registros += len(df)
            print(f"{len(df)} registros.")

        except sqlite3.IntegrityError:
            # Registros duplicados, atualizar
            for _, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO cotacoes
                    (ticker, data, abertura, maxima, minima, fechamento,
                     fechamento_ajustado, volume, mercado, atualizado_em)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (row["ticker"], row["data"], row["abertura"],
                      row["maxima"], row["minima"], row["fechamento"],
                      row["fechamento_ajustado"], row["volume"], row["mercado"]))
            conn.commit()
            print(f"{len(df)} registros (atualizados).")

        except Exception as e:
            print(f"Erro: {e}")

    conn.commit()
    conn.close()
    print(f"\nTotal de cotações salvas: {total_registros}")


# ============================================================
# EXTRAÇÃO DE DADOS FUNDAMENTALISTAS
# ============================================================

def extrair_fundamentalistas(tickers: list):
    """Extrai indicadores fundamentalistas e salva no SQLite."""
    conn = sqlite3.connect(DB_PATH)
    total = 0

    indicadores_chave = [
        "trailingPE", "forwardPE", "priceToBook", "dividendYield",
        "trailingEps", "forwardEps", "marketCap", "enterpriseValue",
        "profitMargins", "operatingMargins", "returnOnEquity",
        "returnOnAssets", "revenueGrowth", "earningsGrowth",
        "totalRevenue", "grossProfits", "ebitda", "netIncomeToCommon",
        "totalDebt", "totalCash", "bookValue", "priceToSalesTrailing12Months",
        "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "fiftyDayAverage", "twoHundredDayAverage",
        "sector", "industry", "longName", "currency",
    ]

    for ticker in tickers:
        try:
            print(f"  Baixando fundamentalistas: {ticker}...", end=" ")
            acao = yf.Ticker(ticker)
            info = acao.info

            if not info:
                print("Sem dados.")
                continue

            mercado = "BR" if ticker.endswith(".SA") else "US"
            count = 0

            for indicador in indicadores_chave:
                valor = info.get(indicador)
                if valor is not None:
                    conn.execute("""
                        INSERT OR REPLACE INTO fundamentalistas
                        (ticker, indicador, valor, mercado, atualizado_em)
                        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (ticker, indicador, str(valor), mercado))
                    count += 1

            conn.commit()
            total += count
            print(f"{count} indicadores.")

        except Exception as e:
            print(f"Erro: {e}")

    conn.close()
    print(f"\nTotal de indicadores salvos: {total}")


# ============================================================
# EXTRAÇÃO DE DIVIDENDOS
# ============================================================

def extrair_dividendos(tickers: list, periodo_anos: int = PERIODO_ANOS):
    """Extrai histórico de dividendos e salva no SQLite."""
    conn = sqlite3.connect(DB_PATH)
    data_inicio = datetime.now() - timedelta(days=periodo_anos * 365)
    total = 0

    for ticker in tickers:
        try:
            print(f"  Baixando dividendos: {ticker}...", end=" ")
            acao = yf.Ticker(ticker)
            divs = acao.dividends

            if divs.empty:
                print("Sem dados.")
                continue

            # Filtrar pelo período
            divs = divs[divs.index >= data_inicio.strftime("%Y-%m-%d")]

            if divs.empty:
                print("Sem dados no período.")
                continue

            mercado = "BR" if ticker.endswith(".SA") else "US"

            for data, valor in divs.items():
                conn.execute("""
                    INSERT OR REPLACE INTO dividendos
                    (ticker, data, valor, mercado, atualizado_em)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (ticker, data.strftime("%Y-%m-%d"), float(valor), mercado))

            conn.commit()
            total += len(divs)
            print(f"{len(divs)} registros.")

        except Exception as e:
            print(f"Erro: {e}")

    conn.close()
    print(f"\nTotal de dividendos salvos: {total}")


# ============================================================
# EXTRAÇÃO DE FINANCEIROS TRIMESTRAIS
# ============================================================

def extrair_financeiros_trimestrais(tickers: list):
    """Extrai dados financeiros trimestrais (DRE, balanço) e salva no SQLite."""
    conn = sqlite3.connect(DB_PATH)
    total = 0

    for ticker in tickers:
        try:
            print(f"  Baixando financeiros trimestrais: {ticker}...", end=" ")
            acao = yf.Ticker(ticker)
            mercado = "BR" if ticker.endswith(".SA") else "US"
            count = 0

            # DRE trimestral (Income Statement)
            try:
                income = acao.quarterly_income_stmt
                if income is not None and not income.empty:
                    for col_date in income.columns:
                        data_str = col_date.strftime("%Y-%m-%d")
                        for indicador in income.index:
                            valor = income.loc[indicador, col_date]
                            if pd.notna(valor):
                                conn.execute("""
                                    INSERT OR REPLACE INTO financeiros_trimestrais
                                    (ticker, data, indicador, valor, tipo, mercado, atualizado_em)
                                    VALUES (?, ?, ?, ?, 'income', ?, CURRENT_TIMESTAMP)
                                """, (ticker, data_str, str(indicador), float(valor), mercado))
                                count += 1
            except Exception:
                pass

            # Balanço trimestral (Balance Sheet)
            try:
                balance = acao.quarterly_balance_sheet
                if balance is not None and not balance.empty:
                    for col_date in balance.columns:
                        data_str = col_date.strftime("%Y-%m-%d")
                        for indicador in balance.index:
                            valor = balance.loc[indicador, col_date]
                            if pd.notna(valor):
                                conn.execute("""
                                    INSERT OR REPLACE INTO financeiros_trimestrais
                                    (ticker, data, indicador, valor, tipo, mercado, atualizado_em)
                                    VALUES (?, ?, ?, ?, 'balance', ?, CURRENT_TIMESTAMP)
                                """, (ticker, data_str, str(indicador), float(valor), mercado))
                                count += 1
            except Exception:
                pass

            # Fluxo de Caixa trimestral
            try:
                cashflow = acao.quarterly_cashflow
                if cashflow is not None and not cashflow.empty:
                    for col_date in cashflow.columns:
                        data_str = col_date.strftime("%Y-%m-%d")
                        for indicador in cashflow.index:
                            valor = cashflow.loc[indicador, col_date]
                            if pd.notna(valor):
                                conn.execute("""
                                    INSERT OR REPLACE INTO financeiros_trimestrais
                                    (ticker, data, indicador, valor, tipo, mercado, atualizado_em)
                                    VALUES (?, ?, ?, ?, 'cashflow', ?, CURRENT_TIMESTAMP)
                                """, (ticker, data_str, str(indicador), float(valor), mercado))
                                count += 1
            except Exception:
                pass

            conn.commit()
            total += count
            print(f"{count} registros.")

        except Exception as e:
            print(f"Erro: {e}")

    conn.close()
    print(f"\nTotal de financeiros trimestrais salvos: {total}")


# ============================================================
# CONSULTAS ÚTEIS
# ============================================================

def consultar_cotacoes(ticker: str, ultimos_dias: int = 30):
    """Retorna as últimas cotações de uma ação."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM cotacoes
        WHERE ticker = ?
        ORDER BY data DESC
        LIMIT ?
    """, conn, params=(ticker, ultimos_dias))
    conn.close()
    return df


def consultar_fundamentalistas(ticker: str):
    """Retorna os indicadores fundamentalistas de uma ação."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT indicador, valor FROM fundamentalistas
        WHERE ticker = ?
        ORDER BY indicador
    """, conn, params=(ticker,))
    conn.close()
    return df


def consultar_dividendos(ticker: str):
    """Retorna o histórico de dividendos de uma ação."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT * FROM dividendos
        WHERE ticker = ?
        ORDER BY data DESC
    """, conn, params=(ticker,))
    conn.close()
    return df


def resumo_banco():
    """Mostra um resumo do banco de dados."""
    conn = sqlite3.connect(DB_PATH)

    cotacoes = pd.read_sql_query(
        "SELECT ticker, COUNT(*) as registros, MIN(data) as de, MAX(data) as ate FROM cotacoes GROUP BY ticker",
        conn
    )
    fundamentos = pd.read_sql_query(
        "SELECT ticker, COUNT(*) as indicadores FROM fundamentalistas GROUP BY ticker",
        conn
    )
    dividendos = pd.read_sql_query(
        "SELECT ticker, COUNT(*) as pagamentos, SUM(valor) as total FROM dividendos GROUP BY ticker",
        conn
    )

    conn.close()

    print("\n" + "=" * 60)
    print("RESUMO DO BANCO DE DADOS")
    print("=" * 60)

    print("\n--- COTAÇÕES ---")
    print(cotacoes.to_string(index=False))

    print("\n--- INDICADORES FUNDAMENTALISTAS ---")
    print(fundamentos.to_string(index=False))

    print("\n--- DIVIDENDOS ---")
    print(dividendos.to_string(index=False))


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def executar_extracao_completa():
    """Executa a extração completa de todos os dados."""
    print("=" * 60)
    print("YAHOO FINANCE EXTRACTOR")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ações: {len(TODAS_ACOES)} ({len(ACOES_BR)} BR + {len(ACOES_US)} US)")
    print("=" * 60)

    print("\n[1/5] Criando banco de dados...")
    criar_banco()

    print("\n[2/5] Extraindo cotações históricas...")
    extrair_cotacoes(TODAS_ACOES)

    print("\n[3/5] Extraindo dados fundamentalistas...")
    extrair_fundamentalistas(TODAS_ACOES)

    print("\n[4/5] Extraindo dividendos...")
    extrair_dividendos(TODAS_ACOES)

    print("\n[5/5] Extraindo financeiros trimestrais...")
    extrair_financeiros_trimestrais(TODAS_ACOES)

    print("\n[OK] Extração completa!")
    resumo_banco()


if __name__ == "__main__":
    executar_extracao_completa()
