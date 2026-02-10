"""
Dashboard Ibovespa - Valuation & Multiplos
TAG Investimentos
Analise de multiplos e valuation das Top 20 acoes do Ibovespa.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils import (
    load_fundamentals_wide, load_cotacoes, load_cotacoes_multi,
    load_dividendos, load_dividendos_annual, compute_sector_averages,
    normalize_prices, format_brl, format_pct, format_number,
    get_last_price, get_last_update_time, get_ticker_display,
    load_financeiros_trimestrais, load_financeiros_wide,
    get_available_financial_indicators, compute_quarterly_multiples,
    IBOVESPA_TOP20, INDICATOR_LABELS, TICKER_NAMES, FINANCIAL_INDICATORS,
)

# ============================================================
# CONFIGURACAO DA PAGINA
# ============================================================

st.set_page_config(
    page_title="TAG Investimentos - Valuation & Multiplos",
    page_icon="https://taginvest.com.br/wp-content/uploads/2021/09/tag-logo-01.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CORES TAG INVESTIMENTOS
# ============================================================
TAG_VERMELHO = "#630D24"
TAG_VERMELHO_CLARO = "#8B1A3A"
TAG_VERMELHO_ESCURO = "#4A0A1B"
TAG_OFFWHITE = "#E6E4DB"
TAG_TEXTO = "#2C1A1A"
TAG_BRANCO = "#FFFFFF"

# Paleta de cores para graficos
TAG_COLORS = [
    TAG_VERMELHO,       # vermelho principal
    "#8B1A3A",          # vermelho claro
    "#B8860B",          # dourado escuro
    "#4A0A1B",          # bordô
    "#A0522D",          # sienna
    "#6B3A2A",          # marrom
    "#D4A574",          # areia
    "#C9B99A",          # bege
    "#7D6B5D",          # cinza quente
    "#3E2723",          # marrom escuro
]

PLOTLY_TEMPLATE = "plotly_white"

# ============================================================
# CSS CUSTOMIZADO TAG
# ============================================================

st.markdown(f"""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {TAG_VERMELHO};
    }}
    [data-testid="stSidebar"] * {{
        color: {TAG_OFFWHITE} !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: {TAG_OFFWHITE} !important;
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
        background-color: {TAG_VERMELHO_CLARO};
        border-radius: 4px;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: rgba(230, 228, 219, 0.3);
    }}

    /* Metric cards */
    [data-testid="stMetric"] {{
        background-color: {TAG_BRANCO};
        border: 1px solid rgba(99, 13, 36, 0.15);
        border-left: 4px solid {TAG_VERMELHO};
        padding: 15px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    [data-testid="stMetric"] label {{
        color: {TAG_VERMELHO} !important;
        font-weight: 600;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {TAG_TEXTO} !important;
    }}

    /* Titulos */
    h1 {{
        color: {TAG_VERMELHO} !important;
        border-bottom: 2px solid {TAG_VERMELHO};
        padding-bottom: 10px;
    }}
    h2, h3 {{
        color: {TAG_VERMELHO_ESCURO} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background-color: {TAG_VERMELHO};
        color: {TAG_OFFWHITE};
        border-radius: 4px 4px 0 0;
    }}
    .stTabs [data-baseweb="tab-list"] button {{
        color: {TAG_VERMELHO};
    }}

    /* Links e botoes */
    a {{
        color: {TAG_VERMELHO} !important;
    }}

    /* Dataframe header */
    .stDataFrame thead th {{
        background-color: {TAG_VERMELHO} !important;
        color: {TAG_OFFWHITE} !important;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        color: {TAG_VERMELHO} !important;
        font-weight: 600;
    }}

    /* Dividers */
    hr {{
        border-color: rgba(99, 13, 36, 0.2);
    }}

    /* Logo container */
    .tag-logo-container {{
        text-align: center;
        padding: 20px 10px 10px 10px;
    }}
    .tag-logo-container img {{
        max-width: 180px;
    }}
    .tag-sidebar-footer {{
        text-align: center;
        font-size: 0.75em;
        opacity: 0.7;
        padding: 10px;
        margin-top: 30px;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR COM LOGO TAG
# ============================================================

st.sidebar.markdown("""
<div class="tag-logo-container">
    <img src="https://taginvest.com.br/wp-content/uploads/2021/09/tag-logo-01.png"
         alt="TAG Investimentos" />
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

pagina = st.sidebar.radio(
    "Navegacao",
    ["Visao Geral", "Multiplos & Valuation", "Analise Individual", "Historico Financeiro", "Comparacao"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Atualizado em: {get_last_update_time()}")
st.sidebar.caption("Dados: Yahoo Finance")

st.sidebar.markdown("""
<div class="tag-sidebar-footer">
    TAG Investimentos<br>
    Dashboard de Valuation
</div>
""", unsafe_allow_html=True)


# ============================================================
# CACHE DE DADOS
# ============================================================

@st.cache_data(ttl=300)
def cached_fundamentals():
    return load_fundamentals_wide()


df_fund = cached_fundamentals()

if df_fund.empty:
    st.error("Banco de dados vazio. Execute o extrator primeiro: `python yahoo_finance_extractor.py`")
    st.stop()


# ============================================================
# PAGINA 1: VISAO GERAL
# ============================================================

def page_visao_geral():
    st.title("Visao Geral - Ibovespa Top 20")

    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acoes Monitoradas", len(df_fund))
    col2.metric(
        "Market Cap Medio",
        format_brl(df_fund["marketCap"].mean()) if "marketCap" in df_fund.columns else "N/D"
    )
    col3.metric(
        "P/L Mediano",
        format_number(df_fund["trailingPE"].median(), 1) if "trailingPE" in df_fund.columns else "N/D"
    )
    col4.metric(
        "DY Medio",
        format_pct(df_fund["dividendYield"].mean()) if "dividendYield" in df_fund.columns else "N/D"
    )

    st.markdown("---")

    # --- Tabela Resumo ---
    st.subheader("Tabela de Multiplos")

    cols_display = ["ticker", "nome", "setor"]
    col_map = {
        "trailingPE": "P/L",
        "priceToBook": "P/VP",
        "ev_ebitda": "EV/EBITDA",
        "dividendYield": "DY (%)",
        "priceToSalesTrailing12Months": "P/Receita",
        "returnOnEquity": "ROE",
        "profitMargins": "Margem Liq.",
        "marketCap": "Market Cap",
    }

    available = [c for c in col_map.keys() if c in df_fund.columns]
    df_table = df_fund[cols_display + available].copy()

    for pct_col in ["returnOnEquity", "profitMargins"]:
        if pct_col in df_table.columns:
            df_table[pct_col] = df_table[pct_col] * 100

    df_table = df_table.rename(columns=col_map)
    df_table = df_table.rename(columns={"ticker": "Ticker", "nome": "Nome", "setor": "Setor"})
    df_table = df_table.sort_values("P/L", na_position="last")

    st.dataframe(
        df_table,
        width="stretch",
        hide_index=True,
        height=600,
        column_config={
            "Market Cap": st.column_config.NumberColumn(format="%.0f"),
            "P/L": st.column_config.NumberColumn(format="%.1f"),
            "P/VP": st.column_config.NumberColumn(format="%.2f"),
            "EV/EBITDA": st.column_config.NumberColumn(format="%.1f"),
            "DY (%)": st.column_config.NumberColumn(format="%.2f%%"),
            "P/Receita": st.column_config.NumberColumn(format="%.2f"),
            "ROE": st.column_config.NumberColumn(format="%.1f%%"),
            "Margem Liq.": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    st.markdown("---")

    # --- Heatmap de Multiplos por Setor ---
    st.subheader("Heatmap de Multiplos")

    heatmap_cols = ["trailingPE", "priceToBook", "ev_ebitda", "dividendYield", "returnOnEquity"]
    heatmap_available = [c for c in heatmap_cols if c in df_fund.columns]

    if heatmap_available:
        df_heat = df_fund[["ticker", "setor"] + heatmap_available].copy()
        df_heat = df_heat.dropna(subset=heatmap_available, how="all")
        df_heat = df_heat.sort_values("setor")

        if "returnOnEquity" in df_heat.columns:
            df_heat["returnOnEquity"] = df_heat["returnOnEquity"] * 100

        df_zscore = df_heat[heatmap_available].copy()
        for col in heatmap_available:
            mean = df_zscore[col].mean()
            std = df_zscore[col].std()
            if std and std > 0:
                df_zscore[col] = (df_zscore[col] - mean) / std
            else:
                df_zscore[col] = 0

        for col in ["dividendYield", "returnOnEquity"]:
            if col in df_zscore.columns:
                df_zscore[col] = df_zscore[col] * -1

        labels_x = [INDICATOR_LABELS.get(c, c) for c in heatmap_available]
        tickers_y = df_heat["ticker"].apply(get_ticker_display).values

        # Colorscale TAG: vermelho (caro) -> branco -> verde (barato)
        tag_colorscale = [
            [0, "#2d6a4f"],
            [0.25, "#52b788"],
            [0.5, TAG_OFFWHITE],
            [0.75, TAG_VERMELHO_CLARO],
            [1, TAG_VERMELHO_ESCURO],
        ]

        fig_heat = go.Figure(data=go.Heatmap(
            z=df_zscore.values,
            x=labels_x,
            y=tickers_y,
            colorscale=tag_colorscale,
            customdata=df_heat[heatmap_available].values,
            texttemplate="%{customdata:.1f}",
            hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.2f}<extra></extra>",
            showscale=True,
        ))

        fig_heat.update_layout(
            height=max(400, len(df_heat) * 30),
            template=PLOTLY_TEMPLATE,
            margin=dict(l=80, r=20, t=20, b=40),
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_heat, width="stretch")


# ============================================================
# PAGINA 2: MULTIPLOS & VALUATION
# ============================================================

def page_multiplos():
    st.title("Multiplos & Valuation")

    multiplo_opcoes = {
        "P/L (Trailing)": "trailingPE",
        "P/L (Forward)": "forwardPE",
        "P/VP": "priceToBook",
        "EV/EBITDA": "ev_ebitda",
        "Dividend Yield (%)": "dividendYield",
        "P/Receita": "priceToSalesTrailing12Months",
    }

    multiplo_selecionado = st.selectbox(
        "Selecione o multiplo:",
        list(multiplo_opcoes.keys()),
    )
    col_name = multiplo_opcoes[multiplo_selecionado]

    if col_name not in df_fund.columns:
        st.warning(f"Dado '{multiplo_selecionado}' nao disponivel.")
        return

    # --- Bar Chart Horizontal ---
    st.subheader(f"{multiplo_selecionado} por Acao")

    df_bar = df_fund[["ticker", "setor", col_name]].dropna(subset=[col_name]).copy()
    df_bar["ticker_display"] = df_bar["ticker"].apply(get_ticker_display)
    df_bar = df_bar.sort_values(col_name, ascending=True)

    fig_bar = px.bar(
        df_bar,
        x=col_name,
        y="ticker_display",
        color="setor",
        orientation="h",
        text=col_name,
        color_discrete_sequence=TAG_COLORS,
        labels={col_name: multiplo_selecionado, "ticker_display": "Acao", "setor": "Setor"},
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(
        height=max(400, len(df_bar) * 35),
        template=PLOTLY_TEMPLATE,
        yaxis=dict(categoryorder="total ascending"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("---")

    # --- Scatter Plot + Medias Setoriais ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Scatter: Multiplo vs Qualidade")

        eixo_y_opcoes = {
            "ROE (%)": "returnOnEquity",
            "ROA (%)": "returnOnAssets",
            "Margem Liquida (%)": "profitMargins",
            "Margem Operacional (%)": "operatingMargins",
        }

        eixo_y_label = st.selectbox("Eixo Y:", list(eixo_y_opcoes.keys()))
        col_y = eixo_y_opcoes[eixo_y_label]

        if col_y in df_fund.columns:
            df_scatter = df_fund[["ticker", "setor", col_name, col_y, "marketCap"]].dropna().copy()
            df_scatter["ticker_display"] = df_scatter["ticker"].apply(get_ticker_display)
            df_scatter[col_y] = df_scatter[col_y] * 100

            fig_scatter = px.scatter(
                df_scatter,
                x=col_name,
                y=col_y,
                color="setor",
                text="ticker_display",
                size="marketCap",
                size_max=40,
                color_discrete_sequence=TAG_COLORS,
                labels={col_name: multiplo_selecionado, col_y: eixo_y_label, "setor": "Setor"},
            )
            fig_scatter.update_traces(textposition="top center", textfont_size=9)

            med_x = df_scatter[col_name].median()
            med_y = df_scatter[col_y].median()
            fig_scatter.add_hline(y=med_y, line_dash="dash", line_color="gray", opacity=0.4,
                                  annotation_text=f"Mediana: {med_y:.1f}%")
            fig_scatter.add_vline(x=med_x, line_dash="dash", line_color="gray", opacity=0.4,
                                  annotation_text=f"Mediana: {med_x:.1f}")

            fig_scatter.update_layout(
                height=500,
                template=PLOTLY_TEMPLATE,
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.info(f"Dado '{eixo_y_label}' nao disponivel.")

    with col_right:
        st.subheader("Medias Setoriais")

        df_sector = compute_sector_averages(df_fund)
        if not df_sector.empty and col_name in df_sector.columns:
            df_sector_sorted = df_sector[["setor", col_name]].dropna().sort_values(col_name)

            fig_sector = px.bar(
                df_sector_sorted,
                x="setor",
                y=col_name,
                text=col_name,
                color="setor",
                color_discrete_sequence=TAG_COLORS,
                labels={"setor": "Setor", col_name: multiplo_selecionado},
            )
            fig_sector.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig_sector.update_layout(
                height=500,
                template=PLOTLY_TEMPLATE,
                showlegend=False,
                xaxis_tickangle=-45,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_sector, width="stretch")
        else:
            st.info("Dados setoriais nao disponiveis.")


# ============================================================
# PAGINA 3: ANALISE INDIVIDUAL
# ============================================================

def page_analise_individual():
    st.title("Analise Individual")

    ticker_options = {
        f"{get_ticker_display(t)} - {TICKER_NAMES.get(t, t)}": t
        for t in sorted(df_fund["ticker"].tolist())
    }

    col_sel1, col_sel2 = st.columns([3, 2])
    with col_sel1:
        selected_display = st.selectbox("Selecione a acao:", list(ticker_options.keys()))
    ticker = ticker_options[selected_display]

    with col_sel2:
        periodo_map = {"1M": 30, "3M": 90, "6M": 180, "1A": 365, "2A": 730, "5A": 1825}
        periodo = st.radio("Periodo:", list(periodo_map.keys()), horizontal=True, index=3)

    start_date = (datetime.now() - timedelta(days=periodo_map[periodo])).strftime("%Y-%m-%d")

    # --- KPI Cards ---
    info = df_fund[df_fund["ticker"] == ticker]
    if info.empty:
        st.warning("Dados nao encontrados para esta acao.")
        return
    info = info.iloc[0]

    last = get_last_price(ticker)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Preco", f'R$ {last["close"]:.2f}', delta=f'{last["change_pct"]:.2f}%')
    c2.metric("P/L", format_number(info.get("trailingPE"), 1))
    c3.metric("P/VP", format_number(info.get("priceToBook"), 2))
    c4.metric("DY", format_pct(info.get("dividendYield")))

    roe_val = info.get("returnOnEquity")
    c5.metric("ROE", format_pct(roe_val * 100 if roe_val and not np.isnan(roe_val) else None))

    st.markdown("---")

    # --- Grafico de Preco + Volume ---
    st.subheader("Historico de Precos")

    chart_type = st.radio("Tipo de grafico:", ["Candlestick", "Linha"], horizontal=True)

    df_prices = load_cotacoes(ticker, start_date=start_date)

    if df_prices.empty:
        st.info("Sem dados de cotacao para o periodo selecionado.")
    else:
        fig_price = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
        )

        if chart_type == "Candlestick":
            fig_price.add_trace(go.Candlestick(
                x=df_prices["data"],
                open=df_prices["abertura"],
                high=df_prices["maxima"],
                low=df_prices["minima"],
                close=df_prices["fechamento"],
                name="OHLC",
                increasing_line_color="#2d6a4f",
                decreasing_line_color=TAG_VERMELHO,
            ), row=1, col=1)
        else:
            fig_price.add_trace(go.Scatter(
                x=df_prices["data"],
                y=df_prices["fechamento_ajustado"],
                mode="lines",
                name="Preco Ajustado",
                line=dict(color=TAG_VERMELHO, width=2),
            ), row=1, col=1)

        colors = ["#2d6a4f" if c >= o else TAG_VERMELHO
                  for c, o in zip(df_prices["fechamento"], df_prices["abertura"])]
        fig_price.add_trace(go.Bar(
            x=df_prices["data"],
            y=df_prices["volume"],
            marker_color=colors,
            name="Volume",
            showlegend=False,
        ), row=2, col=1)

        fig_price.update_layout(
            height=550,
            template=PLOTLY_TEMPLATE,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_price.update_yaxes(title_text="Preco (R$)", row=1, col=1)
        fig_price.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig_price, width="stretch")

    st.markdown("---")

    # --- Indicadores + Dividendos ---
    col_ind, col_div = st.columns(2)

    with col_ind:
        st.subheader("Indicadores Fundamentalistas")

        display_indicators = [
            "trailingPE", "forwardPE", "priceToBook", "ev_ebitda",
            "dividendYield", "priceToSalesTrailing12Months",
            "marketCap", "enterpriseValue", "ebitda",
            "profitMargins", "operatingMargins",
            "returnOnEquity", "returnOnAssets",
            "revenueGrowth", "earningsGrowth",
            "totalRevenue", "netIncomeToCommon",
            "totalDebt", "totalCash", "bookValue",
            "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]

        rows = []
        for ind in display_indicators:
            val = info.get(ind)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                label = INDICATOR_LABELS.get(ind, ind)

                if ind in ["marketCap", "enterpriseValue", "ebitda",
                           "totalRevenue", "netIncomeToCommon", "totalDebt",
                           "totalCash", "grossProfits"]:
                    formatted = format_brl(val)
                elif ind in ["profitMargins", "operatingMargins",
                             "returnOnEquity", "returnOnAssets",
                             "revenueGrowth", "earningsGrowth"]:
                    formatted = format_pct(val * 100)
                elif ind == "dividendYield":
                    formatted = format_pct(val)
                else:
                    formatted = format_number(val, 2)

                rows.append({"Indicador": label, "Valor": formatted})

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
                height=500,
            )

    with col_div:
        st.subheader("Dividendos Anuais")

        df_div = load_dividendos_annual(ticker)
        if df_div.empty:
            st.info("Sem historico de dividendos.")
        else:
            fig_div = px.bar(
                df_div,
                x="ano",
                y="total",
                text="total",
                labels={"ano": "Ano", "total": "Total (R$)"},
                color_discrete_sequence=[TAG_VERMELHO],
            )
            fig_div.update_traces(texttemplate="R$ %{text:.2f}", textposition="outside")
            fig_div.update_layout(
                height=400,
                template=PLOTLY_TEMPLATE,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_div, width="stretch")

        df_div_detail = load_dividendos(ticker)
        if not df_div_detail.empty:
            with st.expander("Ver historico detalhado"):
                df_show = df_div_detail[["data", "valor"]].copy()
                df_show["data"] = df_show["data"].dt.strftime("%d/%m/%Y")
                df_show.columns = ["Data", "Valor (R$)"]
                st.dataframe(df_show, width="stretch", hide_index=True)


# ============================================================
# PAGINA 4: COMPARACAO
# ============================================================

def page_comparacao():
    st.title("Comparacao de Acoes")

    display_options = {get_ticker_display(t): t for t in IBOVESPA_TOP20}

    selected_displays = st.multiselect(
        "Selecione 2 a 5 acoes para comparar:",
        list(display_options.keys()),
        default=["PETR4", "VALE3"],
        max_selections=5,
    )

    selected_tickers = [display_options[d] for d in selected_displays]

    if len(selected_tickers) < 2:
        st.warning("Selecione pelo menos 2 acoes para comparar.")
        return

    periodo_map = {"3M": 90, "6M": 180, "1A": 365, "2A": 730, "5A": 1825}
    periodo = st.radio("Periodo:", list(periodo_map.keys()), horizontal=True, index=2)
    start_date = (datetime.now() - timedelta(days=periodo_map[periodo])).strftime("%Y-%m-%d")

    st.markdown("---")

    # --- Performance Normalizada ---
    st.subheader("Performance Relativa (Base 100)")

    df_multi = load_cotacoes_multi(selected_tickers, start_date=start_date)

    if df_multi.empty:
        st.info("Sem dados de cotacao para o periodo selecionado.")
    else:
        df_norm = normalize_prices(df_multi)
        df_norm["ticker_display"] = df_norm["ticker"].apply(get_ticker_display)

        fig_perf = px.line(
            df_norm,
            x="data",
            y="preco_normalizado",
            color="ticker_display",
            color_discrete_sequence=TAG_COLORS,
            labels={"data": "Data", "preco_normalizado": "Retorno (Base 100)", "ticker_display": "Acao"},
        )
        fig_perf.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.3)
        fig_perf.update_layout(
            height=450,
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_perf, width="stretch")

    st.markdown("---")

    # --- Comparacao de Multiplos ---
    st.subheader("Comparacao de Multiplos")

    comparison_metrics = {
        "trailingPE": "P/L",
        "priceToBook": "P/VP",
        "ev_ebitda": "EV/EBITDA",
        "dividendYield": "DY (%)",
    }

    available_metrics = {k: v for k, v in comparison_metrics.items() if k in df_fund.columns}
    df_comp = df_fund[df_fund["ticker"].isin(selected_tickers)][
        ["ticker"] + list(available_metrics.keys())
    ].copy()
    df_comp["ticker"] = df_comp["ticker"].apply(get_ticker_display)

    df_melted = df_comp.melt(id_vars="ticker", var_name="multiplo", value_name="valor")
    df_melted["multiplo"] = df_melted["multiplo"].map(available_metrics)

    fig_comp = px.bar(
        df_melted.dropna(subset=["valor"]),
        x="multiplo",
        y="valor",
        color="ticker",
        barmode="group",
        text="valor",
        color_discrete_sequence=TAG_COLORS,
        labels={"multiplo": "Multiplo", "valor": "Valor", "ticker": "Acao"},
    )
    fig_comp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_comp.update_layout(
        height=450,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_comp, width="stretch")

    st.markdown("---")

    # --- Tabela Detalhada ---
    st.subheader("Tabela Comparativa Detalhada")

    detail_cols = {
        "nome": "Nome",
        "setor": "Setor",
        "trailingPE": "P/L",
        "forwardPE": "P/L (Fwd)",
        "priceToBook": "P/VP",
        "ev_ebitda": "EV/EBITDA",
        "dividendYield": "DY (%)",
        "returnOnEquity": "ROE",
        "profitMargins": "Margem Liq.",
        "operatingMargins": "Margem Op.",
        "marketCap": "Market Cap",
    }

    available_detail = {k: v for k, v in detail_cols.items() if k in df_fund.columns}
    df_detail = df_fund[df_fund["ticker"].isin(selected_tickers)][
        ["ticker"] + list(available_detail.keys())
    ].copy()

    for pct_col in ["returnOnEquity", "profitMargins", "operatingMargins"]:
        if pct_col in df_detail.columns:
            df_detail[pct_col] = df_detail[pct_col] * 100

    if "marketCap" in df_detail.columns:
        df_detail["marketCap"] = df_detail["marketCap"].apply(format_brl)

    df_detail["ticker"] = df_detail["ticker"].apply(get_ticker_display)
    df_detail = df_detail.set_index("ticker").T
    df_detail.index = [available_detail.get(idx, idx) for idx in df_detail.index]

    st.dataframe(df_detail, width="stretch")


# ============================================================
# PAGINA 5: HISTORICO FINANCEIRO
# ============================================================

def page_historico_financeiro():
    st.title("Historico Financeiro")
    st.caption("Evolucao trimestral de indicadores financeiros e multiplos")

    ticker_options = {
        f"{get_ticker_display(t)} - {TICKER_NAMES.get(t, t)}": t
        for t in sorted(df_fund["ticker"].tolist())
    }

    selected_display = st.selectbox(
        "Selecione a acao:", list(ticker_options.keys()), key="hist_ticker"
    )
    ticker = ticker_options[selected_display]

    available = get_available_financial_indicators(ticker)
    if not available:
        st.warning(
            "Sem dados financeiros trimestrais para esta acao. "
            "Execute o extrator novamente: `python yahoo_finance_extractor.py`"
        )
        return

    tab_dre, tab_multiplos, tab_balanco, tab_fluxo = st.tabs(
        ["DRE (Resultado)", "Multiplos Historicos", "Balanco", "Fluxo de Caixa"]
    )

    # ===================== ABA DRE =====================
    with tab_dre:
        st.subheader("Demonstracao de Resultado (DRE)")

        dre_indicators = [
            "Total Revenue", "Gross Profit", "Operating Income",
            "EBITDA", "EBIT", "Net Income", "Pretax Income",
        ]
        dre_available = [i for i in dre_indicators if i in available]

        if not dre_available:
            st.info("Sem dados de DRE disponiveis.")
        else:
            dre_selected = st.multiselect(
                "Indicadores DRE:",
                dre_available,
                default=dre_available[:4],
                format_func=lambda x: FINANCIAL_INDICATORS.get(x, x),
                key="dre_select",
            )

            if dre_selected:
                df_dre = load_financeiros_wide(ticker, dre_selected)
                if not df_dre.empty:
                    df_melted = df_dre.melt(
                        id_vars="data", var_name="indicador", value_name="valor"
                    )
                    df_melted["indicador_pt"] = df_melted["indicador"].map(
                        lambda x: FINANCIAL_INDICATORS.get(x, x)
                    )

                    fig_dre = px.bar(
                        df_melted,
                        x="data",
                        y="valor",
                        color="indicador_pt",
                        barmode="group",
                        color_discrete_sequence=TAG_COLORS,
                        labels={"data": "Trimestre", "valor": "Valor (R$)", "indicador_pt": "Indicador"},
                    )
                    fig_dre.update_layout(
                        height=500,
                        template=PLOTLY_TEMPLATE,
                        xaxis=dict(dtick="M3", tickformat="%b/%Y"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig_dre.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig_dre, width="stretch")

                    st.subheader("Evolucao Trimestral")
                    fig_line = px.line(
                        df_melted,
                        x="data",
                        y="valor",
                        color="indicador_pt",
                        markers=True,
                        color_discrete_sequence=TAG_COLORS,
                        labels={"data": "Trimestre", "valor": "Valor (R$)", "indicador_pt": "Indicador"},
                    )
                    fig_line.update_layout(
                        height=400,
                        template=PLOTLY_TEMPLATE,
                        xaxis=dict(dtick="M3", tickformat="%b/%Y"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig_line.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig_line, width="stretch")

                    with st.expander("Ver tabela de dados"):
                        df_show = df_dre.copy()
                        df_show["data"] = df_show["data"].dt.strftime("%d/%m/%Y")
                        for col in dre_selected:
                            if col in df_show.columns:
                                df_show[col] = df_show[col].apply(
                                    lambda v: format_brl(v) if pd.notna(v) else "N/D"
                                )
                        df_show = df_show.rename(
                            columns={**{"data": "Trimestre"},
                                     **{k: FINANCIAL_INDICATORS.get(k, k) for k in dre_selected}}
                        )
                        st.dataframe(df_show, width="stretch", hide_index=True)

    # ===================== ABA MULTIPLOS =====================
    with tab_multiplos:
        st.subheader("Multiplos Historicos Trimestrais")

        df_mult = compute_quarterly_multiples(ticker)

        if df_mult.empty:
            st.info("Sem dados para calcular multiplos historicos.")
        else:
            has_revenue = "Total Revenue" in df_mult.columns and df_mult["Total Revenue"].notna().any()
            has_income = "Net Income" in df_mult.columns and df_mult["Net Income"].notna().any()
            has_ebitda = "EBITDA" in df_mult.columns and df_mult["EBITDA"].notna().any()
            has_equity = "Stockholders Equity" in df_mult.columns and df_mult["Stockholders Equity"].notna().any()

            col1, col2 = st.columns(2)

            with col1:
                if has_revenue or has_income:
                    traces = []
                    if has_revenue:
                        traces.append(("Total Revenue", "Receita Total"))
                    if has_income:
                        traces.append(("Net Income", "Lucro Liquido"))

                    fig_rl = go.Figure()
                    colors_rl = [TAG_VERMELHO, "#B8860B"]
                    for i, (col, name) in enumerate(traces):
                        fig_rl.add_trace(go.Bar(
                            x=df_mult["data"],
                            y=df_mult[col],
                            name=name,
                            marker_color=colors_rl[i],
                        ))
                    fig_rl.update_layout(
                        title="Receita vs Lucro Liquido (Trimestral)",
                        height=400,
                        template=PLOTLY_TEMPLATE,
                        barmode="group",
                        xaxis=dict(dtick="M3", tickformat="%b/%Y"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig_rl.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig_rl, width="stretch")

            with col2:
                if has_ebitda:
                    fig_eb = go.Figure()
                    fig_eb.add_trace(go.Bar(
                        x=df_mult["data"],
                        y=df_mult["EBITDA"],
                        name="EBITDA",
                        marker_color=TAG_VERMELHO_CLARO,
                    ))
                    fig_eb.update_layout(
                        title="EBITDA (Trimestral)",
                        height=400,
                        template=PLOTLY_TEMPLATE,
                        xaxis=dict(dtick="M3", tickformat="%b/%Y"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig_eb.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig_eb, width="stretch")

            st.subheader("Margens Trimestrais")

            df_margins = df_mult[["data"]].copy()
            if has_revenue and has_income:
                df_margins["Margem Liquida (%)"] = (
                    df_mult["Net Income"] / df_mult["Total Revenue"] * 100
                )
            if has_revenue and has_ebitda:
                df_margins["Margem EBITDA (%)"] = (
                    df_mult["EBITDA"] / df_mult["Total Revenue"] * 100
                )

            margin_cols = [c for c in df_margins.columns if c != "data"]
            if margin_cols:
                df_mg_melted = df_margins.melt(
                    id_vars="data", var_name="margem", value_name="valor"
                )
                fig_mg = px.line(
                    df_mg_melted,
                    x="data",
                    y="valor",
                    color="margem",
                    markers=True,
                    color_discrete_sequence=[TAG_VERMELHO, "#B8860B"],
                    labels={"data": "Trimestre", "valor": "%", "margem": "Margem"},
                )
                fig_mg.update_layout(
                    height=400,
                    template=PLOTLY_TEMPLATE,
                    xaxis=dict(dtick="M3", tickformat="%b/%Y"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_mg, width="stretch")

            if has_income and has_equity:
                st.subheader("ROE Trimestral")
                df_mult["ROE (%)"] = (
                    df_mult["Net Income"] / df_mult["Stockholders Equity"] * 100
                )
                fig_roe = px.line(
                    df_mult,
                    x="data",
                    y="ROE (%)",
                    markers=True,
                    labels={"data": "Trimestre", "ROE (%)": "ROE (%)"},
                    color_discrete_sequence=[TAG_VERMELHO],
                )
                fig_roe.update_layout(
                    height=350,
                    template=PLOTLY_TEMPLATE,
                    xaxis=dict(dtick="M3", tickformat="%b/%Y"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_roe, width="stretch")

    # ===================== ABA BALANCO =====================
    with tab_balanco:
        st.subheader("Balanco Patrimonial")

        bal_indicators = [
            "Total Assets", "Total Liabilities Net Minority Interest",
            "Stockholders Equity", "Total Debt",
            "Cash And Cash Equivalents", "Net Debt", "Working Capital",
        ]
        bal_available = [i for i in bal_indicators if i in available]

        if not bal_available:
            st.info("Sem dados de balanco disponiveis.")
        else:
            bal_selected = st.multiselect(
                "Indicadores do Balanco:",
                bal_available,
                default=bal_available[:4],
                format_func=lambda x: FINANCIAL_INDICATORS.get(x, x),
                key="bal_select",
            )

            if bal_selected:
                df_bal = load_financeiros_wide(ticker, bal_selected)
                if not df_bal.empty:
                    df_bal_melted = df_bal.melt(
                        id_vars="data", var_name="indicador", value_name="valor"
                    )
                    df_bal_melted["indicador_pt"] = df_bal_melted["indicador"].map(
                        lambda x: FINANCIAL_INDICATORS.get(x, x)
                    )

                    fig_bal = px.bar(
                        df_bal_melted,
                        x="data",
                        y="valor",
                        color="indicador_pt",
                        barmode="group",
                        color_discrete_sequence=TAG_COLORS,
                        labels={"data": "Trimestre", "valor": "Valor (R$)", "indicador_pt": "Indicador"},
                    )
                    fig_bal.update_layout(
                        height=500,
                        template=PLOTLY_TEMPLATE,
                        xaxis=dict(dtick="M3", tickformat="%b/%Y"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig_bal.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig_bal, width="stretch")

                    with st.expander("Ver tabela de dados"):
                        df_show = df_bal.copy()
                        df_show["data"] = df_show["data"].dt.strftime("%d/%m/%Y")
                        for col in bal_selected:
                            if col in df_show.columns:
                                df_show[col] = df_show[col].apply(
                                    lambda v: format_brl(v) if pd.notna(v) else "N/D"
                                )
                        df_show = df_show.rename(
                            columns={**{"data": "Trimestre"},
                                     **{k: FINANCIAL_INDICATORS.get(k, k) for k in bal_selected}}
                        )
                        st.dataframe(df_show, width="stretch", hide_index=True)

    # ===================== ABA FLUXO DE CAIXA =====================
    with tab_fluxo:
        st.subheader("Fluxo de Caixa")

        cf_indicators = [
            "Free Cash Flow", "Operating Cash Flow", "Capital Expenditure",
        ]
        cf_available = [i for i in cf_indicators if i in available]

        if not cf_available:
            st.info("Sem dados de fluxo de caixa disponiveis.")
        else:
            cf_selected = st.multiselect(
                "Indicadores de Fluxo de Caixa:",
                cf_available,
                default=cf_available,
                format_func=lambda x: FINANCIAL_INDICATORS.get(x, x),
                key="cf_select",
            )

            if cf_selected:
                df_cf = load_financeiros_wide(ticker, cf_selected)
                if not df_cf.empty:
                    df_cf_melted = df_cf.melt(
                        id_vars="data", var_name="indicador", value_name="valor"
                    )
                    df_cf_melted["indicador_pt"] = df_cf_melted["indicador"].map(
                        lambda x: FINANCIAL_INDICATORS.get(x, x)
                    )

                    fig_cf = px.bar(
                        df_cf_melted,
                        x="data",
                        y="valor",
                        color="indicador_pt",
                        barmode="group",
                        color_discrete_sequence=TAG_COLORS,
                        labels={"data": "Trimestre", "valor": "Valor (R$)", "indicador_pt": "Indicador"},
                    )
                    fig_cf.update_layout(
                        height=500,
                        template=PLOTLY_TEMPLATE,
                        xaxis=dict(dtick="M3", tickformat="%b/%Y"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig_cf.update_yaxes(tickformat=",.0f")
                    st.plotly_chart(fig_cf, width="stretch")


# ============================================================
# ROTEAMENTO
# ============================================================

if pagina == "Visao Geral":
    page_visao_geral()
elif pagina == "Multiplos & Valuation":
    page_multiplos()
elif pagina == "Analise Individual":
    page_analise_individual()
elif pagina == "Historico Financeiro":
    page_historico_financeiro()
elif pagina == "Comparacao":
    page_comparacao()
