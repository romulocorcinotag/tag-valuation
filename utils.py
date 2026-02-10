"""
Camada de dados para o Dashboard Ibovespa.
Centraliza queries SQLite, transformações, cálculos avançados e formatações.
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

# Nomes amigáveis
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
# INDICADORES FINANCEIROS - Labels PT-BR
# ============================================================

FINANCIAL_INDICATORS = {
    # DRE
    "Total Revenue": "Receita Total",
    "Gross Profit": "Lucro Bruto",
    "Operating Income": "Lucro Operacional",
    "Net Income": "Lucro Liquido",
    "Net Income Common Stockholders": "Lucro Liq. Acionistas",
    "EBITDA": "EBITDA",
    "Normalized EBITDA": "EBITDA Normalizado",
    "EBIT": "EBIT",
    "Total Expenses": "Despesas Totais",
    "Cost Of Revenue": "Custo da Receita",
    "Operating Revenue": "Receita Operacional",
    "Pretax Income": "Lucro Antes IR",
    "Tax Provision": "Impostos",
    "Interest Expense": "Despesas Financeiras",
    "Interest Expense Non Operating": "Desp. Fin. Nao Operacional",
    "Interest Income": "Receita Financeira",
    "Net Interest Income": "Resultado Financeiro Liq.",
    "Selling General And Administration": "SG&A",
    "Depreciation And Amortization In Income Statement": "D&A (DRE)",
    "Basic EPS": "LPA Basico",
    "Diluted EPS": "LPA Diluido",
    "Basic Average Shares": "Acoes Media Basica",
    "Diluted Average Shares": "Acoes Media Diluida",
    "Operating Expense": "Despesas Operacionais",
    # Balanco
    "Total Assets": "Ativo Total",
    "Total Liabilities Net Minority Interest": "Passivo Total",
    "Stockholders Equity": "Patrimonio Liquido",
    "Common Stock Equity": "PL Ordinario",
    "Total Debt": "Divida Total",
    "Current Debt": "Divida Curto Prazo",
    "Current Debt And Capital Lease Obligation": "Divida CP + Leasing",
    "Long Term Debt": "Divida Longo Prazo",
    "Long Term Debt And Capital Lease Obligation": "Divida LP + Leasing",
    "Cash And Cash Equivalents": "Caixa e Equivalentes",
    "Cash Cash Equivalents And Short Term Investments": "Caixa + Invest. CP",
    "Net Debt": "Divida Liquida",
    "Working Capital": "Capital de Giro",
    "Current Assets": "Ativo Circulante",
    "Current Liabilities": "Passivo Circulante",
    "Inventory": "Estoques",
    "Accounts Receivable": "Contas a Receber",
    "Accounts Payable": "Contas a Pagar",
    "Net PPE": "Imobilizado Liquido",
    "Goodwill And Other Intangible Assets": "Goodwill + Intangiveis",
    "Invested Capital": "Capital Investido",
    "Retained Earnings": "Lucros Acumulados",
    "Minority Interest": "Participacao Minoritaria",
    "Total Non Current Assets": "Ativo Nao Circulante",
    "Total Non Current Liabilities Net Minority Interest": "Passivo Nao Circulante",
    "Tangible Book Value": "Valor Patrimonial Tangivel",
    "Total Capitalization": "Capitalizacao Total",
    # Fluxo de Caixa
    "Free Cash Flow": "Fluxo de Caixa Livre (FCF)",
    "Operating Cash Flow": "Fluxo de Caixa Operacional (FCO)",
    "Capital Expenditure": "Capex",
    "Investing Cash Flow": "Fluxo de Investimento (FCI)",
    "Financing Cash Flow": "Fluxo de Financiamento (FCF Fin.)",
    "Cash Dividends Paid": "Dividendos Pagos",
    "Common Stock Dividend Paid": "Dividendos Ordinarios Pagos",
    "Issuance Of Debt": "Emissao de Divida",
    "Repayment Of Debt": "Amortizacao de Divida",
    "Net Issuance Payments Of Debt": "Emissao Liq. de Divida",
    "Depreciation And Amortization": "D&A (Fluxo Caixa)",
    "Change In Working Capital": "Var. Capital de Giro",
    "Change In Receivables": "Var. Contas a Receber",
    "Change In Inventory": "Var. Estoques",
    "Change In Payable": "Var. Contas a Pagar",
    "Beginning Cash Position": "Caixa Inicial",
    "End Cash Position": "Caixa Final",
    "Net Income From Continuing Operations": "Lucro Liq. (Fluxo Caixa)",
    "Stock Based Compensation": "Remuneracao em Acoes",
    "Interest Paid Supplemental Data": "Juros Pagos",
    "Income Tax Paid Supplemental Data": "IR Pago",
}


# ============================================================
# CONEXÃO
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


# ============================================================
# FUNÇÕES DE CARGA BÁSICAS
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

    for col in df_wide.columns:
        if col not in TEXT_INDICATORS and col != "ticker":
            df_wide[col] = df_wide[col].apply(_safe_float)

    if "dividendYield" in df_wide.columns:
        dy_max = df_wide["dividendYield"].max()
        if dy_max is not None and dy_max < 1:
            df_wide["dividendYield"] = df_wide["dividendYield"] * 100

    if "enterpriseValue" in df_wide.columns and "ebitda" in df_wide.columns:
        df_wide["ev_ebitda"] = df_wide.apply(
            lambda r: r["enterpriseValue"] / r["ebitda"]
            if r["ebitda"] and r["ebitda"] != 0 and r["enterpriseValue"]
            else None, axis=1
        )

    df_wide["nome"] = df_wide["ticker"].map(TICKER_NAMES)

    if "sector" in df_wide.columns:
        df_wide["setor"] = df_wide["sector"].map(
            lambda x: SECTOR_TRANSLATION.get(x, x) if x else "N/D"
        )

    return df_wide


def load_cotacoes(ticker: str, start_date: str = None):
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

def load_financeiros_trimestrais(ticker: str, indicadores: list = None):
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
    df = load_financeiros_trimestrais(ticker, indicadores)
    if df.empty:
        return pd.DataFrame()
    df_wide = df.pivot(index="data", columns="indicador", values="valor").reset_index()
    df_wide = df_wide.sort_values("data")
    return df_wide


def get_available_financial_indicators(ticker: str):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT DISTINCT indicador FROM financeiros_trimestrais WHERE ticker = ? ORDER BY indicador",
        conn, params=(ticker,)
    )
    conn.close()
    return df["indicador"].tolist()


# ============================================================
# CÁLCULOS AVANÇADOS - ANÁLISE FUNDAMENTALISTA PROFUNDA
# ============================================================

def load_full_quarterly_data(ticker: str):
    """Carrega TODOS os dados financeiros trimestrais de um ticker em formato wide."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT data, indicador, valor FROM financeiros_trimestrais "
        "WHERE ticker = ? ORDER BY data ASC",
        conn, params=(ticker,)
    )
    conn.close()
    if df.empty:
        return pd.DataFrame()
    df["data"] = pd.to_datetime(df["data"])
    df_wide = df.pivot(index="data", columns="indicador", values="valor").reset_index()
    df_wide = df_wide.sort_values("data").reset_index(drop=True)
    return df_wide


def safe_divide(numerator, denominator):
    """Divisão segura que retorna NaN se denominador for 0 ou NaN."""
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    if numerator is None or pd.isna(numerator):
        return np.nan
    return numerator / denominator


def compute_receita_margens(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula Receita e Margens trimestrais (Bruta, EBITDA, Operacional, Liquida)."""
    result = df[["data"]].copy()

    if "Total Revenue" in df.columns:
        result["Receita Liquida"] = df["Total Revenue"]
    if "Gross Profit" in df.columns:
        result["Lucro Bruto"] = df["Gross Profit"]
    if "EBITDA" in df.columns:
        result["EBITDA"] = df["EBITDA"]
    if "Operating Income" in df.columns:
        result["Lucro Operacional"] = df["Operating Income"]
    if "Net Income" in df.columns:
        result["Lucro Liquido"] = df["Net Income"]

    # Margens (%)
    if "Total Revenue" in df.columns:
        rev = df["Total Revenue"]
        if "Gross Profit" in df.columns:
            result["Margem Bruta (%)"] = (df["Gross Profit"] / rev * 100).replace([np.inf, -np.inf], np.nan)
        if "EBITDA" in df.columns:
            result["Margem EBITDA (%)"] = (df["EBITDA"] / rev * 100).replace([np.inf, -np.inf], np.nan)
        if "Operating Income" in df.columns:
            result["Margem Operacional (%)"] = (df["Operating Income"] / rev * 100).replace([np.inf, -np.inf], np.nan)
        if "Net Income" in df.columns:
            result["Margem Liquida (%)"] = (df["Net Income"] / rev * 100).replace([np.inf, -np.inf], np.nan)

    # Crescimento YoY
    if "Total Revenue" in df.columns and len(df) > 4:
        result["Cresc. Receita YoY (%)"] = df["Total Revenue"].pct_change(periods=4) * 100
    if "Net Income" in df.columns and len(df) > 4:
        result["Cresc. Lucro YoY (%)"] = df["Net Income"].pct_change(periods=4) * 100
    if "EBITDA" in df.columns and len(df) > 4:
        result["Cresc. EBITDA YoY (%)"] = df["EBITDA"].pct_change(periods=4) * 100

    return result


def compute_fluxo_caixa(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas de Fluxo de Caixa."""
    result = df[["data"]].copy()

    if "Operating Cash Flow" in df.columns:
        result["FCO"] = df["Operating Cash Flow"]
    if "Capital Expenditure" in df.columns:
        result["Capex"] = df["Capital Expenditure"].abs() * -1  # negativo
    if "Free Cash Flow" in df.columns:
        result["FCF"] = df["Free Cash Flow"]
    elif "Operating Cash Flow" in df.columns and "Capital Expenditure" in df.columns:
        result["FCF"] = df["Operating Cash Flow"] + df["Capital Expenditure"]
    if "Investing Cash Flow" in df.columns:
        result["FCI"] = df["Investing Cash Flow"]
    if "Financing Cash Flow" in df.columns:
        result["FCF Financiamento"] = df["Financing Cash Flow"]

    # Conversão de caixa: FCO / EBITDA
    if "Operating Cash Flow" in df.columns and "EBITDA" in df.columns:
        result["FCO/EBITDA (%)"] = (df["Operating Cash Flow"] / df["EBITDA"] * 100).replace([np.inf, -np.inf], np.nan)

    # Conversão de caixa: FCF / Lucro Liquido
    if "Net Income" in df.columns:
        fcf_col = "Free Cash Flow" if "Free Cash Flow" in df.columns else None
        if fcf_col:
            result["FCF/Lucro Liq (%)"] = (df[fcf_col] / df["Net Income"] * 100).replace([np.inf, -np.inf], np.nan)

    # Capex / Receita
    if "Capital Expenditure" in df.columns and "Total Revenue" in df.columns:
        result["Capex/Receita (%)"] = (df["Capital Expenditure"].abs() / df["Total Revenue"] * 100).replace([np.inf, -np.inf], np.nan)

    return result


def compute_alavancagem(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas de Alavancagem e Endividamento."""
    result = df[["data"]].copy()

    # Divida Liquida / EBITDA (LTM)
    if "Net Debt" in df.columns and "EBITDA" in df.columns:
        ebitda_ltm = df["EBITDA"].rolling(window=4, min_periods=4).sum()
        result["Divida Liq/EBITDA"] = (df["Net Debt"] / ebitda_ltm).replace([np.inf, -np.inf], np.nan)

    # Divida Bruta / PL
    if "Total Debt" in df.columns and "Stockholders Equity" in df.columns:
        result["Divida Bruta/PL"] = (df["Total Debt"] / df["Stockholders Equity"]).replace([np.inf, -np.inf], np.nan)

    # Divida Liquida / PL
    if "Net Debt" in df.columns and "Stockholders Equity" in df.columns:
        result["Divida Liq/PL"] = (df["Net Debt"] / df["Stockholders Equity"]).replace([np.inf, -np.inf], np.nan)

    # Estrutura de Capital: Divida / (Divida + PL)
    if "Total Debt" in df.columns and "Stockholders Equity" in df.columns:
        total = df["Total Debt"] + df["Stockholders Equity"]
        result["Estrutura Capital (%)"] = (df["Total Debt"] / total * 100).replace([np.inf, -np.inf], np.nan)

    # Composicao da Divida
    if "Total Debt" in df.columns:
        result["Divida Bruta"] = df["Total Debt"]
    if "Net Debt" in df.columns:
        result["Divida Liquida"] = df["Net Debt"]
    if "Current Debt" in df.columns:
        result["Divida CP"] = df["Current Debt"]
    elif "Current Debt And Capital Lease Obligation" in df.columns:
        result["Divida CP"] = df["Current Debt And Capital Lease Obligation"]
    if "Long Term Debt" in df.columns:
        result["Divida LP"] = df["Long Term Debt"]
    elif "Long Term Debt And Capital Lease Obligation" in df.columns:
        result["Divida LP"] = df["Long Term Debt And Capital Lease Obligation"]

    # % CP e LP
    if "Divida CP" in result.columns and "Divida LP" in result.columns:
        total_debt = result["Divida CP"].abs() + result["Divida LP"].abs()
        result["% Divida CP"] = (result["Divida CP"].abs() / total_debt * 100).replace([np.inf, -np.inf], np.nan)
        result["% Divida LP"] = (result["Divida LP"].abs() / total_debt * 100).replace([np.inf, -np.inf], np.nan)

    return result


def compute_icr_cobertura(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ICR (Cobertura de Juros) e FCO/EBITDA."""
    result = df[["data"]].copy()

    # ICR = EBITDA / Despesas Financeiras
    interest_col = None
    for col in ["Interest Expense", "Interest Expense Non Operating"]:
        if col in df.columns:
            interest_col = col
            break

    if "EBITDA" in df.columns and interest_col:
        interest_abs = df[interest_col].abs()
        result["Cobertura de Juros (ICR)"] = (df["EBITDA"] / interest_abs).replace([np.inf, -np.inf], np.nan)

    # FCO / Despesa Financeira
    if "Operating Cash Flow" in df.columns and interest_col:
        interest_abs = df[interest_col].abs()
        result["FCO/Desp. Financeira"] = (df["Operating Cash Flow"] / interest_abs).replace([np.inf, -np.inf], np.nan)

    # EBIT / Despesa Financeira
    if "EBIT" in df.columns and interest_col:
        interest_abs = df[interest_col].abs()
        result["EBIT/Desp. Financeira"] = (df["EBIT"] / interest_abs).replace([np.inf, -np.inf], np.nan)

    return result


def compute_liquidez(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores de Liquidez."""
    result = df[["data"]].copy()

    # Liquidez Corrente = Ativo Circulante / Passivo Circulante
    if "Current Assets" in df.columns and "Current Liabilities" in df.columns:
        result["Liquidez Corrente"] = (df["Current Assets"] / df["Current Liabilities"]).replace([np.inf, -np.inf], np.nan)

    # Liquidez Seca = (Ativo Circulante - Estoques) / Passivo Circulante
    if "Current Assets" in df.columns and "Current Liabilities" in df.columns:
        inventario = df.get("Inventory", pd.Series(0, index=df.index)).fillna(0)
        result["Liquidez Seca"] = ((df["Current Assets"] - inventario) / df["Current Liabilities"]).replace([np.inf, -np.inf], np.nan)

    # Liquidez Imediata = Caixa / Passivo Circulante
    cash_col = None
    for col in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
        if col in df.columns:
            cash_col = col
            break
    if cash_col and "Current Liabilities" in df.columns:
        result["Liquidez Imediata"] = (df[cash_col] / df["Current Liabilities"]).replace([np.inf, -np.inf], np.nan)

    # Composicao: Caixa, Contas a Receber, Estoques
    if cash_col:
        result["Caixa e Equivalentes"] = df[cash_col]
    if "Accounts Receivable" in df.columns:
        result["Contas a Receber"] = df["Accounts Receivable"]
    if "Inventory" in df.columns:
        result["Estoques"] = df["Inventory"]
    if "Current Assets" in df.columns:
        result["Ativo Circulante"] = df["Current Assets"]
    if "Current Liabilities" in df.columns:
        result["Passivo Circulante"] = df["Current Liabilities"]

    return result


def compute_rentabilidade(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ROE, ROA, ROIC e análise DuPont."""
    result = df[["data"]].copy()

    # ROE = Lucro Liquido / PL
    if "Net Income" in df.columns and "Stockholders Equity" in df.columns:
        result["ROE (%)"] = (df["Net Income"] / df["Stockholders Equity"] * 100).replace([np.inf, -np.inf], np.nan)

    # ROA = Lucro Liquido / Ativo Total
    if "Net Income" in df.columns and "Total Assets" in df.columns:
        result["ROA (%)"] = (df["Net Income"] / df["Total Assets"] * 100).replace([np.inf, -np.inf], np.nan)

    # ROIC = NOPAT / Capital Investido
    # NOPAT = EBIT * (1 - taxa efetiva de impostos)
    if "EBIT" in df.columns and "Tax Provision" in df.columns and "Pretax Income" in df.columns:
        tax_rate = (df["Tax Provision"] / df["Pretax Income"]).replace([np.inf, -np.inf], np.nan).fillna(0.34)
        tax_rate = tax_rate.clip(0, 1)  # limitar entre 0 e 100%
        nopat = df["EBIT"] * (1 - tax_rate)

        # Capital Investido = PL + Divida Liquida
        invested = None
        if "Invested Capital" in df.columns:
            invested = df["Invested Capital"]
        elif "Stockholders Equity" in df.columns and "Net Debt" in df.columns:
            invested = df["Stockholders Equity"] + df["Net Debt"]
        elif "Stockholders Equity" in df.columns and "Total Debt" in df.columns:
            cash = df.get("Cash And Cash Equivalents", pd.Series(0, index=df.index)).fillna(0)
            invested = df["Stockholders Equity"] + df["Total Debt"] - cash

        if invested is not None:
            result["ROIC (%)"] = (nopat / invested * 100).replace([np.inf, -np.inf], np.nan)
            result["NOPAT"] = nopat

    # Análise DuPont (3 fatores)
    # ROE = Margem Liquida x Giro do Ativo x Alavancagem Financeira
    if "Net Income" in df.columns and "Total Revenue" in df.columns:
        result["DuPont: Margem Liq (%)"] = (df["Net Income"] / df["Total Revenue"] * 100).replace([np.inf, -np.inf], np.nan)
    if "Total Revenue" in df.columns and "Total Assets" in df.columns:
        result["DuPont: Giro Ativo"] = (df["Total Revenue"] / df["Total Assets"]).replace([np.inf, -np.inf], np.nan)
    if "Total Assets" in df.columns and "Stockholders Equity" in df.columns:
        result["DuPont: Alavancagem"] = (df["Total Assets"] / df["Stockholders Equity"]).replace([np.inf, -np.inf], np.nan)

    # Taxa efetiva de impostos
    if "Tax Provision" in df.columns and "Pretax Income" in df.columns:
        result["Taxa Efetiva IR (%)"] = (df["Tax Provision"] / df["Pretax Income"] * 100).replace([np.inf, -np.inf], np.nan)

    return result


def compute_divida_composicao(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula composição detalhada da dívida (CP vs LP)."""
    result = df[["data"]].copy()

    cash_col = None
    for col in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
        if col in df.columns:
            cash_col = col
            break

    if cash_col:
        result["Caixa e Equivalentes"] = df[cash_col]

    cp_col = "Current Debt" if "Current Debt" in df.columns else \
             ("Current Debt And Capital Lease Obligation" if "Current Debt And Capital Lease Obligation" in df.columns else None)
    lp_col = "Long Term Debt" if "Long Term Debt" in df.columns else \
             ("Long Term Debt And Capital Lease Obligation" if "Long Term Debt And Capital Lease Obligation" in df.columns else None)

    if cp_col:
        result["EFCP"] = df[cp_col]  # Endividamento Financeiro CP
    if lp_col:
        result["EFLP"] = df[lp_col]  # Endividamento Financeiro LP
    if "Total Debt" in df.columns:
        result["Divida Bruta"] = df["Total Debt"]
    if "Net Debt" in df.columns:
        result["Divida Liquida"] = df["Net Debt"]

    return result


def compute_balanco_patrimonial(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas do Balanço Patrimonial."""
    result = df[["data"]].copy()

    for col, label in [
        ("Total Assets", "Ativo Total"),
        ("Current Assets", "Ativo Circulante"),
        ("Total Non Current Assets", "Ativo Nao Circulante"),
        ("Total Liabilities Net Minority Interest", "Passivo Total"),
        ("Current Liabilities", "Passivo Circulante"),
        ("Total Non Current Liabilities Net Minority Interest", "Passivo Nao Circulante"),
        ("Stockholders Equity", "Patrimonio Liquido"),
        ("Net PPE", "Imobilizado"),
        ("Goodwill And Other Intangible Assets", "Goodwill + Intangiveis"),
        ("Retained Earnings", "Lucros Acumulados"),
        ("Working Capital", "Capital de Giro"),
    ]:
        if col in df.columns:
            result[label] = df[col]

    # Composição do ativo (%)
    if "Total Assets" in df.columns:
        ta = df["Total Assets"]
        if "Current Assets" in df.columns:
            result["% Ativo Circulante"] = (df["Current Assets"] / ta * 100).replace([np.inf, -np.inf], np.nan)
        if "Total Non Current Assets" in df.columns:
            result["% Ativo Nao Circulante"] = (df["Total Non Current Assets"] / ta * 100).replace([np.inf, -np.inf], np.nan)

    return result


def compute_valuation_historico(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Calcula múltiplos históricos de valuation: P/L, P/VP, EV/EBITDA, P/Receita."""
    df_cot = load_cotacoes(ticker)
    if df_cot.empty or df.empty:
        return pd.DataFrame()

    result = []
    for _, row in df.iterrows():
        data_tri = row["data"]
        # Pegar cotação mais próxima
        mask = (df_cot["data"] >= data_tri) & (df_cot["data"] <= data_tri + pd.Timedelta(days=10))
        cot = df_cot[mask]
        if cot.empty:
            mask2 = (df_cot["data"] <= data_tri) & (df_cot["data"] >= data_tri - pd.Timedelta(days=10))
            cot = df_cot[mask2]

        preco = cot["fechamento"].iloc[-1] if not cot.empty else None

        row_dict = {"data": data_tri, "Preco": preco}

        # LTM (Last Twelve Months) - soma dos últimos 4 trimestres
        idx = df.index[df["data"] == data_tri]
        if len(idx) > 0:
            i = idx[0]
            for col in ["Net Income", "Total Revenue", "EBITDA", "Free Cash Flow"]:
                if col in df.columns and i >= 3:
                    ltm = df[col].iloc[i-3:i+1].sum()
                    row_dict[f"{col}_LTM"] = ltm
                elif col in df.columns:
                    row_dict[f"{col}_LTM"] = np.nan

            # Valores pontuais do balanço
            for col in ["Stockholders Equity", "Total Assets", "Total Debt", "Net Debt"]:
                if col in df.columns:
                    row_dict[col] = row.get(col)

            # Shares outstanding
            for col in ["Diluted Average Shares", "Basic Average Shares"]:
                if col in df.columns and pd.notna(row.get(col)):
                    row_dict["shares"] = row[col]
                    break

        result.append(row_dict)

    df_result = pd.DataFrame(result)

    if df_result.empty or "Preco" not in df_result.columns:
        return df_result

    # Calcular múltiplos
    shares = df_result.get("shares")
    if shares is not None:
        market_cap = df_result["Preco"] * shares

        if "Net Income_LTM" in df_result.columns:
            df_result["P/L"] = (market_cap / df_result["Net Income_LTM"]).replace([np.inf, -np.inf], np.nan)

        if "Stockholders Equity" in df_result.columns:
            df_result["P/VP"] = (market_cap / df_result["Stockholders Equity"]).replace([np.inf, -np.inf], np.nan)

        if "Total Revenue_LTM" in df_result.columns:
            df_result["P/Receita"] = (market_cap / df_result["Total Revenue_LTM"]).replace([np.inf, -np.inf], np.nan)

        if "EBITDA_LTM" in df_result.columns:
            net_debt = df_result.get("Net Debt", pd.Series(0, index=df_result.index)).fillna(0)
            ev = market_cap + net_debt
            df_result["EV/EBITDA"] = (ev / df_result["EBITDA_LTM"]).replace([np.inf, -np.inf], np.nan)

        if "Free Cash Flow_LTM" in df_result.columns:
            df_result["P/FCF"] = (market_cap / df_result["Free Cash Flow_LTM"]).replace([np.inf, -np.inf], np.nan)

        # Earnings Yield e FCF Yield
        if "Net Income_LTM" in df_result.columns:
            df_result["Earnings Yield (%)"] = (df_result["Net Income_LTM"] / market_cap * 100).replace([np.inf, -np.inf], np.nan)
        if "Free Cash Flow_LTM" in df_result.columns:
            df_result["FCF Yield (%)"] = (df_result["Free Cash Flow_LTM"] / market_cap * 100).replace([np.inf, -np.inf], np.nan)

    return df_result


def compute_piotroski_score(df: pd.DataFrame) -> dict:
    """
    Calcula o Piotroski F-Score (0-9) com base nos últimos dados trimestrais.
    Retorna dict com score total e detalhamento de cada critério.
    """
    if df.empty or len(df) < 2:
        return {"score": None, "details": {}}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    details = {}

    # 1. ROA positivo (Lucro Liquido / Ativo Total > 0)
    if "Net Income" in df.columns and "Total Assets" in df.columns:
        roa = safe_divide(latest.get("Net Income"), latest.get("Total Assets"))
        passed = roa is not None and not np.isnan(roa) and roa > 0
        details["ROA Positivo"] = {"value": roa, "passed": passed}
        if passed:
            score += 1

    # 2. FCO positivo
    if "Operating Cash Flow" in df.columns:
        fco = latest.get("Operating Cash Flow")
        passed = fco is not None and not np.isnan(fco) and fco > 0
        details["FCO Positivo"] = {"value": fco, "passed": passed}
        if passed:
            score += 1

    # 3. ROA crescente (comparado ao trimestre anterior)
    if "Net Income" in df.columns and "Total Assets" in df.columns:
        roa_curr = safe_divide(latest.get("Net Income"), latest.get("Total Assets"))
        roa_prev = safe_divide(prev.get("Net Income"), prev.get("Total Assets"))
        passed = (roa_curr is not None and roa_prev is not None and
                  not np.isnan(roa_curr) and not np.isnan(roa_prev) and roa_curr > roa_prev)
        details["ROA Crescente"] = {"value": roa_curr, "passed": passed}
        if passed:
            score += 1

    # 4. FCO > Lucro Liquido (qualidade dos lucros)
    if "Operating Cash Flow" in df.columns and "Net Income" in df.columns:
        fco = latest.get("Operating Cash Flow")
        ni = latest.get("Net Income")
        passed = (fco is not None and ni is not None and
                  not np.isnan(fco) and not np.isnan(ni) and fco > ni)
        details["FCO > Lucro (Accruals)"] = {"value": f"FCO={fco}, NI={ni}", "passed": passed}
        if passed:
            score += 1

    # 5. Alavancagem decrescente (Divida/Ativo)
    if "Total Debt" in df.columns and "Total Assets" in df.columns:
        lev_curr = safe_divide(latest.get("Total Debt"), latest.get("Total Assets"))
        lev_prev = safe_divide(prev.get("Total Debt"), prev.get("Total Assets"))
        passed = (lev_curr is not None and lev_prev is not None and
                  not np.isnan(lev_curr) and not np.isnan(lev_prev) and lev_curr < lev_prev)
        details["Alavancagem Decrescente"] = {"value": lev_curr, "passed": passed}
        if passed:
            score += 1

    # 6. Liquidez Corrente crescente
    if "Current Assets" in df.columns and "Current Liabilities" in df.columns:
        liq_curr = safe_divide(latest.get("Current Assets"), latest.get("Current Liabilities"))
        liq_prev = safe_divide(prev.get("Current Assets"), prev.get("Current Liabilities"))
        passed = (liq_curr is not None and liq_prev is not None and
                  not np.isnan(liq_curr) and not np.isnan(liq_prev) and liq_curr > liq_prev)
        details["Liquidez Crescente"] = {"value": liq_curr, "passed": passed}
        if passed:
            score += 1

    # 7. Sem diluição (ações não aumentaram)
    shares_col = "Diluted Average Shares" if "Diluted Average Shares" in df.columns else \
                 ("Basic Average Shares" if "Basic Average Shares" in df.columns else None)
    if shares_col:
        shares_curr = latest.get(shares_col)
        shares_prev = prev.get(shares_col)
        passed = (shares_curr is not None and shares_prev is not None and
                  not np.isnan(shares_curr) and not np.isnan(shares_prev) and shares_curr <= shares_prev)
        details["Sem Diluicao"] = {"value": shares_curr, "passed": passed}
        if passed:
            score += 1

    # 8. Margem Bruta crescente
    if "Gross Profit" in df.columns and "Total Revenue" in df.columns:
        mg_curr = safe_divide(latest.get("Gross Profit"), latest.get("Total Revenue"))
        mg_prev = safe_divide(prev.get("Gross Profit"), prev.get("Total Revenue"))
        passed = (mg_curr is not None and mg_prev is not None and
                  not np.isnan(mg_curr) and not np.isnan(mg_prev) and mg_curr > mg_prev)
        details["Margem Bruta Crescente"] = {"value": mg_curr, "passed": passed}
        if passed:
            score += 1

    # 9. Giro do Ativo crescente
    if "Total Revenue" in df.columns and "Total Assets" in df.columns:
        giro_curr = safe_divide(latest.get("Total Revenue"), latest.get("Total Assets"))
        giro_prev = safe_divide(prev.get("Total Revenue"), prev.get("Total Assets"))
        passed = (giro_curr is not None and giro_prev is not None and
                  not np.isnan(giro_curr) and not np.isnan(giro_prev) and giro_curr > giro_prev)
        details["Giro Ativo Crescente"] = {"value": giro_curr, "passed": passed}
        if passed:
            score += 1

    return {"score": score, "details": details}


def compute_quarterly_multiples(ticker: str):
    """Calcula multiplos trimestrais historicos (versao legada)."""
    indicadores = ["Total Revenue", "Net Income", "EBITDA", "Stockholders Equity"]
    df_fin = load_financeiros_wide(ticker, indicadores)
    if df_fin.empty:
        return pd.DataFrame()

    df_cot = load_cotacoes(ticker)
    if df_cot.empty:
        return df_fin

    result_rows = []
    for _, row in df_fin.iterrows():
        data_tri = row["data"]
        mask = (df_cot["data"] >= data_tri) & (df_cot["data"] <= data_tri + pd.Timedelta(days=10))
        cot_close = df_cot[mask]
        if cot_close.empty:
            mask2 = (df_cot["data"] <= data_tri) & (df_cot["data"] >= data_tri - pd.Timedelta(days=10))
            cot_close = df_cot[mask2]

        preco = cot_close["fechamento"].iloc[-1] if not cot_close.empty else None
        row_dict = {"data": data_tri, "preco": preco}
        for ind in indicadores:
            row_dict[ind] = row.get(ind)
        result_rows.append(row_dict)

    df_result = pd.DataFrame(result_rows)
    return df_result


# ============================================================
# FUNÇÕES DE CÁLCULO GERAIS
# ============================================================

def compute_sector_averages(df_wide):
    if "setor" not in df_wide.columns:
        return pd.DataFrame()
    numeric_cols = ["trailingPE", "forwardPE", "priceToBook", "dividendYield",
                    "ev_ebitda", "returnOnEquity", "returnOnAssets"]
    available_cols = [c for c in numeric_cols if c in df_wide.columns]
    df_avg = df_wide.groupby("setor")[available_cols].mean().reset_index()
    return df_avg


def normalize_prices(df_multi):
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
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/D"
    return f"{value:.2f}%"


def format_number(value, decimals=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/D"
    return f"{value:,.{decimals}f}"


def get_last_price(ticker: str):
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
    return {"close": close, "change": change, "change_pct": change_pct, "date": df["data"].iloc[0]}


def get_last_update_time():
    conn = get_connection()
    cursor = conn.execute("SELECT MAX(atualizado_em) FROM fundamentalistas")
    result = cursor.fetchone()[0]
    conn.close()
    return result if result else "N/D"


def get_ticker_display(ticker: str):
    return ticker.replace(".SA", "")
