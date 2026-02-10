"""
Camada de dados para o Dashboard Ibovespa.
Centraliza queries SQLite, transformações e formatações.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

DB_PATH = Path(__file__).parent / "yahoo_finance.db"

IBOVESPA_TOP20 = [
    "VALE3.SA", "PETR4.SA", "PETR3.SA", "ITUB4.SA", "BBDC4.SA",
    "BBAS3.SA", "ABEV3.SA", "WEGE3.SA", "RENT3.SA", "SUZB3.SA",
    "B3SA3.SA", "ITSA4.SA", "SANB11.SA", "EQTL3.SA",
    "PRIO3.SA", "RADL3.SA", "RAIL3.SA", "VIVT3.SA", "HAPV3.SA",
    "CSAN3.SA",
    "ENEV3.SA",
]

# Nomes amigáveis (fallback para encoding ruim do yfinance)
TICKER_NAMES = {
    "VALE3.SA": "Vale",
    "PETR4.SA": "Petrobras PN",
    "PETR3.SA": "Petrobras ON",
    "ITUB4.SA": "Itaú Unibanco",
    "BBDC4.SA": "Bradesco PN",
    "BBAS3.SA": "Banco do Brasil",
    "ABEV3.SA": "Ambev",
    "WEGE3.SA": "WEG",
    "RENT3.SA": "Localiza",
    "SUZB3.SA": "Suzano",
    "B3SA3.SA": "B3",
    "ITSA4.SA": "Itaúsa",
    "SANB11.SA": "Santander Brasil",
    "EQTL3.SA": "Equatorial Energia",
    "CSAN3.SA": "Cosan",
    "PRIO3.SA": "PRIO",
    "RADL3.SA": "Raia Drogasil",
    "RAIL3.SA": "Rumo",
    "VIVT3.SA": "Vivo",
    "HAPV3.SA": "Hapvida",
    "ENEV3.SA": "Eneva",
}

INDICATOR_LABELS = {
    "trailingPE": "P/L",
    "forwardPE": "P/L (Forward)",
    "priceToBook": "P/VP",
    "dividendYield": "Dividend Yield (%)",
    "priceToSalesTrailing12Months": "P/Receita",
    "marketCap": "Market Cap",
    "enterpriseValue": "Valor da Firma (EV)",
    "ebitda": "EBITDA",
    "profitMargins": "Margem Líquida",
    "operatingMargins": "Margem Operacional",
    "returnOnEquity": "ROE",
    "returnOnAssets": "ROA",
    "revenueGrowth": "Cresc. Receita",
    "earningsGrowth": "Cresc. Lucro",
    "totalRevenue": "Receita Total",
    "grossProfits": "Lucro Bruto",
    "netIncomeToCommon": "Lucro Líquido",
    "totalDebt": "Dívida Total",
    "totalCash": "Caixa Total",
    "bookValue": "Valor Patrimonial",
    "beta": "Beta",
    "fiftyTwoWeekHigh": "Máxima 52 Semanas",
    "fiftyTwoWeekLow": "Mínima 52 Semanas",
    "fiftyDayAverage": "Média 50 Dias",
    "twoHundredDayAverage": "Média 200 Dias",
    "sector": "Setor",
    "industry": "Indústria",
    "longName": "Nome",
    "currency": "Moeda",
    "trailingEps": "LPA (Trailing)",
    "forwardEps": "LPA (Forward)",
    "ev_ebitda": "EV/EBITDA",
}

TEXT_INDICATORS = {"sector", "industry", "longName", "currency"}

SECTOR_TRANSLATION = {
    "Basic Materials": "Materiais Básicos",
    "Communication Services": "Comunicação",
    "Consumer Cyclical": "Consumo Cíclico",
    "Consumer Defensive": "Consumo Não Cíclico",
    "Energy": "Energia",
    "Financial Services": "Financeiro",
    "Financial": "Financeiro",
    "Healthcare": "Saúde",
    "Industrials": "Indústria",
    "Real Estate": "Imobiliário",
    "Technology": "Tecnologia",
    "Utilities": "Utilidades Públicas",
}


# ============================================================
# CONEXÃO
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


# ============================================================
# FUNÇÕES DE CARGA
# ============================================================

def _safe_float(val):
    if val is None or val == "None" or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def load_fundamentals_wide():
    """Carrega indicadores fundamentalistas em formato wide (pivotado)."""
    conn = get_connection()
    placeholders = ",".join("?" * len(IBOVESPA_TOP20))
    df = pd.read_sql_query(
        f"SELECT ticker, indicador, valor FROM fundamentalistas "
        f"WHERE ticker IN ({placeholders})",
        conn, params=IBOVESPA_TOP20,
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df_wide = df.pivot(index="ticker", columns="indicador", values="valor").reset_index()

    # Converter colunas numéricas
    for col in df_wide.columns:
        if col not in TEXT_INDICATORS and col != "ticker":
            df_wide[col] = df_wide[col].apply(_safe_float)

    # Normalizar dividendYield (se vier como decimal, converter para %)
    if "dividendYield" in df_wide.columns:
        dy_max = df_wide["dividendYield"].max()
        if dy_max is not None and dy_max < 1:
            df_wide["dividendYield"] = df_wide["dividendYield"] * 100

    # Calcular EV/EBITDA
    if "enterpriseValue" in df_wide.columns and "ebitda" in df_wide.columns:
        df_wide["ev_ebitda"] = df_wide.apply(
            lambda r: r["enterpriseValue"] / r["ebitda"]
            if r["ebitda"] and r["ebitda"] != 0 and r["enterpriseValue"]
            else None, axis=1
        )

    # Adicionar nomes amigáveis
    df_wide["nome"] = df_wide["ticker"].map(TICKER_NAMES)

    # Traduzir setores
    if "sector" in df_wide.columns:
        df_wide["setor"] = df_wide["sector"].map(
            lambda x: SECTOR_TRANSLATION.get(x, x) if x else "N/D"
        )

    return df_wide


def load_cotacoes(ticker: str, start_date: str = None):
    """Carrega cotações históricas de uma ação."""
    conn = get_connection()
    query = "SELECT * FROM cotacoes WHERE ticker = ?"
    params = [ticker]

    if start_date:
        query += " AND data >= ?"
        params.append(start_date)

    query += " ORDER BY data ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["data"] = pd.to_datetime(df["data"])

    return df


def load_cotacoes_multi(tickers: list, start_date: str = None):
    """Carrega cotações de múltiplas ações."""
    conn = get_connection()
    placeholders = ",".join("?" * len(tickers))
    query = f"SELECT * FROM cotacoes WHERE ticker IN ({placeholders})"
    params = list(tickers)

    if start_date:
        query += " AND data >= ?"
        params.append(start_date)

    query += " ORDER BY data ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["data"] = pd.to_datetime(df["data"])

    return df


def load_dividendos(ticker: str):
    """Carrega histórico de dividendos de uma ação."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM dividendos WHERE ticker = ? ORDER BY data ASC",
        conn, params=(ticker,)
    )
    conn.close()

    if not df.empty:
        df["data"] = pd.to_datetime(df["data"])

    return df


def load_dividendos_annual(ticker: str):
    """Agrega dividendos por ano."""
    df = load_dividendos(ticker)
    if df.empty:
        return pd.DataFrame(columns=["ano", "total"])

    df["ano"] = df["data"].dt.year
    df_annual = df.groupby("ano")["valor"].sum().reset_index()
    df_annual.columns = ["ano", "total"]
    return df_annual


# ============================================================
# FINANCEIROS TRIMESTRAIS
# ============================================================

FINANCIAL_INDICATORS = {
    "Total Revenue": "Receita Total",
    "Gross Profit": "Lucro Bruto",
    "Operating Income": "Lucro Operacional",
    "Net Income": "Lucro Liquido",
    "EBITDA": "EBITDA",
    "EBIT": "EBIT",
    "Total Expenses": "Despesas Totais",
    "Cost Of Revenue": "Custo da Receita",
    "Operating Revenue": "Receita Operacional",
    "Pretax Income": "Lucro Antes IR",
    "Tax Provision": "Impostos",
    "Interest Expense": "Despesas Financeiras",
    "Total Assets": "Ativo Total",
    "Total Liabilities Net Minority Interest": "Passivo Total",
    "Stockholders Equity": "Patrimonio Liquido",
    "Total Debt": "Divida Total",
    "Cash And Cash Equivalents": "Caixa e Equivalentes",
    "Net Debt": "Divida Liquida",
    "Working Capital": "Capital de Giro",
    "Free Cash Flow": "Fluxo de Caixa Livre",
    "Operating Cash Flow": "Fluxo de Caixa Operacional",
    "Capital Expenditure": "Capex",
}


def load_financeiros_trimestrais(ticker: str, indicadores: list = None):
    """Carrega dados financeiros trimestrais de uma acao."""
    conn = get_connection()

    if indicadores:
        placeholders = ",".join("?" * len(indicadores))
        query = (
            f"SELECT data, indicador, valor FROM financeiros_trimestrais "
            f"WHERE ticker = ? AND indicador IN ({placeholders}) "
            f"ORDER BY data ASC"
        )
        params = [ticker] + indicadores
    else:
        query = (
            "SELECT data, indicador, valor FROM financeiros_trimestrais "
            "WHERE ticker = ? ORDER BY data ASC"
        )
        params = [ticker]

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["data"] = pd.to_datetime(df["data"])

    return df


def load_financeiros_wide(ticker: str, indicadores: list):
    """Carrega financeiros trimestrais em formato wide (cada indicador e uma coluna)."""
    df = load_financeiros_trimestrais(ticker, indicadores)
    if df.empty:
        return pd.DataFrame()

    df_wide = df.pivot(index="data", columns="indicador", values="valor").reset_index()
    df_wide = df_wide.sort_values("data")
    return df_wide


def get_available_financial_indicators(ticker: str):
    """Retorna lista de indicadores financeiros disponiveis para um ticker."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT DISTINCT indicador FROM financeiros_trimestrais WHERE ticker = ? ORDER BY indicador",
        conn, params=(ticker,)
    )
    conn.close()
    return df["indicador"].tolist()


def compute_quarterly_multiples(ticker: str):
    """Calcula multiplos trimestrais historicos usando cotacoes e financials."""
    # Carregar financials
    indicadores = ["Total Revenue", "Net Income", "EBITDA", "Stockholders Equity"]
    df_fin = load_financeiros_wide(ticker, indicadores)
    if df_fin.empty:
        return pd.DataFrame()

    # Carregar cotacoes para pegar preco no fim de cada trimestre
    df_cot = load_cotacoes(ticker)
    if df_cot.empty:
        return df_fin

    # Para cada trimestre, pegar o preco de fechamento mais proximo
    result_rows = []
    for _, row in df_fin.iterrows():
        data_tri = row["data"]
        # Pegar cotacao mais proxima (ate 10 dias depois)
        mask = (df_cot["data"] >= data_tri) & (df_cot["data"] <= data_tri + pd.Timedelta(days=10))
        cot_close = df_cot[mask]
        if cot_close.empty:
            # Tentar antes
            mask2 = (df_cot["data"] <= data_tri) & (df_cot["data"] >= data_tri - pd.Timedelta(days=10))
            cot_close = df_cot[mask2]

        preco = cot_close["fechamento"].iloc[-1] if not cot_close.empty else None

        row_dict = {"data": data_tri, "preco": preco}

        # Receita, Lucro, EBITDA (valores trimestrais)
        for ind in indicadores:
            row_dict[ind] = row.get(ind)

        result_rows.append(row_dict)

    df_result = pd.DataFrame(result_rows)
    return df_result


# ============================================================
# FUNÇÕES DE CÁLCULO
# ============================================================

def compute_sector_averages(df_wide):
    """Calcula médias setoriais dos múltiplos."""
    if "setor" not in df_wide.columns:
        return pd.DataFrame()

    numeric_cols = ["trailingPE", "forwardPE", "priceToBook", "dividendYield",
                    "ev_ebitda", "returnOnEquity", "returnOnAssets"]
    available_cols = [c for c in numeric_cols if c in df_wide.columns]

    df_avg = df_wide.groupby("setor")[available_cols].mean().reset_index()
    return df_avg


def normalize_prices(df_multi):
    """Normaliza preços para base 100 na data inicial."""
    if df_multi.empty:
        return df_multi

    result = []
    for ticker in df_multi["ticker"].unique():
        df_t = df_multi[df_multi["ticker"] == ticker].copy()
        if not df_t.empty:
            base = df_t["fechamento_ajustado"].iloc[0]
            if base and base != 0:
                df_t["preco_normalizado"] = (df_t["fechamento_ajustado"] / base) * 100
            else:
                df_t["preco_normalizado"] = 100
            result.append(df_t)

    if not result:
        return pd.DataFrame()

    return pd.concat(result, ignore_index=True)


# ============================================================
# FORMATADORES
# ============================================================

def format_brl(value):
    """Formata números grandes em estilo BR: R$ 501,9 bi."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/D"
    if abs(value) >= 1e12:
        return f"R$ {value/1e12:,.1f} tri"
    if abs(value) >= 1e9:
        return f"R$ {value/1e9:,.1f} bi"
    if abs(value) >= 1e6:
        return f"R$ {value/1e6:,.1f} mi"
    return f"R$ {value:,.0f}"


def format_pct(value):
    """Formata percentual."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/D"
    return f"{value:.2f}%"


def format_number(value, decimals=2):
    """Formata número genérico."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/D"
    return f"{value:,.{decimals}f}"


def get_last_price(ticker: str):
    """Retorna último preço de fechamento e variação."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT data, fechamento FROM cotacoes WHERE ticker = ? ORDER BY data DESC LIMIT 2",
        conn, params=(ticker,)
    )
    conn.close()

    if df.empty:
        return {"close": 0, "change": 0, "change_pct": 0, "date": "N/D"}

    close = df["fechamento"].iloc[0]
    if len(df) > 1:
        prev = df["fechamento"].iloc[1]
        change = close - prev
        change_pct = (change / prev) * 100 if prev else 0
    else:
        change = 0
        change_pct = 0

    return {
        "close": close,
        "change": change,
        "change_pct": change_pct,
        "date": df["data"].iloc[0],
    }


def get_last_update_time():
    """Retorna data da última atualização dos dados."""
    conn = get_connection()
    cursor = conn.execute("SELECT MAX(atualizado_em) FROM fundamentalistas")
    result = cursor.fetchone()[0]
    conn.close()
    return result if result else "N/D"


def get_ticker_display(ticker: str):
    """Retorna ticker limpo sem .SA."""
    return ticker.replace(".SA", "")
