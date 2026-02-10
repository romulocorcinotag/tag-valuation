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
    load_full_quarterly_data, compute_receita_margens, compute_fluxo_caixa,
    compute_alavancagem, compute_icr_cobertura, compute_liquidez,
    compute_rentabilidade, compute_divida_composicao, compute_balanco_patrimonial,
    compute_valuation_historico, compute_piotroski_score,
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
TAG_VERDE = "#1B7A4A"
TAG_DOURADO = "#B8860B"

TAG_COLORS = [
    TAG_VERMELHO, "#8B1A3A", "#B8860B", "#4A0A1B", "#A0522D",
    "#6B3A2A", "#D4A574", "#C9B99A", "#7D6B5D", "#3E2723",
]

PLOTLY_TEMPLATE = "plotly_white"

# ============================================================
# CSS PROFISSIONAL
# ============================================================

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Reset global - fontes maiores e mais limpas */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* Main area */
    .main .block-container {{
        padding: 1.5rem 2rem;
        max-width: 100%;
    }}

    /* Sidebar profissional */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {TAG_VERMELHO} 0%, {TAG_VERMELHO_ESCURO} 100%);
        min-width: 260px;
    }}
    [data-testid="stSidebar"] * {{
        color: {TAG_OFFWHITE} !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: {TAG_OFFWHITE} !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 6px 8px !important;
        transition: all 0.2s ease;
    }}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
        background-color: rgba(255,255,255,0.12);
        border-radius: 6px;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: rgba(230, 228, 219, 0.2);
        margin: 12px 0;
    }}

    /* Metric cards modernos */
    [data-testid="stMetric"] {{
        background: {TAG_BRANCO};
        border: none;
        border-left: 4px solid {TAG_VERMELHO};
        padding: 18px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease;
    }}
    [data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }}
    [data-testid="stMetric"] label {{
        color: {TAG_VERMELHO} !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {TAG_TEXTO} !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }}

    /* Titulos profissionais */
    h1 {{
        color: {TAG_VERMELHO} !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        border-bottom: 3px solid {TAG_VERMELHO};
        padding-bottom: 12px;
        margin-bottom: 20px !important;
        letter-spacing: -0.01em;
    }}
    h2 {{
        color: {TAG_VERMELHO_ESCURO} !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        margin-top: 24px !important;
    }}
    h3 {{
        color: {TAG_VERMELHO_ESCURO} !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }}

    /* Tabs modernas */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        background-color: {TAG_VERMELHO} !important;
        color: {TAG_OFFWHITE} !important;
        border-radius: 8px 8px 0 0;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
    }}
    .stTabs [data-baseweb="tab-list"] button {{
        color: {TAG_VERMELHO} !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 8px 16px !important;
        border-radius: 8px 8px 0 0;
    }}
    .stTabs [data-baseweb="tab-list"] button:hover {{
        background-color: rgba(99, 13, 36, 0.08);
    }}

    /* Dataframes profissionais */
    .stDataFrame {{
        border-radius: 8px;
        overflow: hidden;
    }}
    .stDataFrame thead th {{
        background-color: {TAG_VERMELHO} !important;
        color: {TAG_OFFWHITE} !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 10px 12px !important;
    }}
    .stDataFrame td {{
        font-size: 0.9rem !important;
        padding: 8px 12px !important;
    }}

    /* Selectbox e inputs */
    .stSelectbox label, .stMultiSelect label, .stRadio > label {{
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: {TAG_TEXTO} !important;
    }}

    /* Caption e texto pequeno */
    .stCaption, [data-testid="stCaptionContainer"] {{
        font-size: 0.85rem !important;
        color: #666 !important;
    }}

    /* Links */
    a {{ color: {TAG_VERMELHO} !important; }}

    /* Expander */
    .streamlit-expanderHeader {{
        color: {TAG_VERMELHO} !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}

    /* Dividers */
    hr {{ border-color: rgba(99, 13, 36, 0.12); margin: 20px 0; }}

    /* Sidebar logo */
    .tag-logo-container {{
        text-align: center;
        padding: 24px 16px 8px 16px;
    }}
    .tag-logo-container img {{
        max-width: 170px;
        filter: brightness(1.1);
    }}
    .tag-sidebar-footer {{
        text-align: center;
        font-size: 0.72rem;
        opacity: 0.6;
        padding: 12px;
        margin-top: 30px;
        letter-spacing: 0.02em;
    }}

    /* Custom card component */
    .metric-card {{
        background: {TAG_BRANCO};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }}
    .metric-card:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }}
    .metric-card .label {{
        font-size: 0.78rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    .metric-card .value {{
        font-size: 1.5rem;
        font-weight: 700;
        line-height: 1.2;
    }}

    /* Score badge */
    .score-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.03em;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
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
    ["Visao Geral", "Multiplos & Valuation", "Analise Individual",
     "Analise Fundamentalista", "Score & Rating", "Comparacao"],
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
# CACHE E DADOS
# ============================================================

@st.cache_data(ttl=300)
def cached_fundamentals():
    return load_fundamentals_wide()

df_fund = cached_fundamentals()

if df_fund.empty:
    st.error("Banco de dados vazio. Execute o extrator primeiro: `python yahoo_finance_extractor.py`")
    st.stop()


# ============================================================
# HELPERS PARA GRAFICOS PROFISSIONAIS
# ============================================================

def _chart_layout(fig, height=450, title=None, show_legend=True):
    """Aplica layout profissional padrao a qualquer grafico."""
    layout_args = dict(
        height=height,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=50 if title else 20, b=50),
        font=dict(family="Inter, sans-serif", size=13, color=TAG_TEXTO),
        xaxis=dict(
            dtick="M3", tickformat="%b/%Y",
            gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ) if show_legend else dict(visible=False),
    )
    if title:
        layout_args["title"] = dict(text=title, font=dict(size=15, color=TAG_VERMELHO_ESCURO), x=0)
    fig.update_layout(**layout_args)
    return fig


def _melt_for_chart(df, exclude_cols=None):
    """Converte DataFrame wide para long."""
    if exclude_cols is None:
        exclude_cols = ["data"]
    value_cols = [c for c in df.columns if c not in exclude_cols]
    if not value_cols:
        return pd.DataFrame()
    return df.melt(id_vars="data", value_vars=value_cols,
                   var_name="indicador", value_name="valor")


def _make_chart(df_melted, chart_type, height, y_label, colors, y_format=None,
                barmode="group", title=None, show_legend=True):
    """Helper para criar graficos padronizados."""
    if chart_type == "bar":
        fig = px.bar(df_melted, x="data", y="valor", color="indicador",
                     barmode=barmode, color_discrete_sequence=colors,
                     labels={"data": "", "valor": y_label, "indicador": ""})
    else:
        fig = px.line(df_melted, x="data", y="valor", color="indicador",
                      markers=True, color_discrete_sequence=colors,
                      labels={"data": "", "valor": y_label, "indicador": ""})
    _chart_layout(fig, height, title, show_legend)
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    return fig


def _metric_card(label, value, suffix="", color=None, border_color=None):
    """Renderiza um metric card customizado em HTML."""
    if color is None:
        color = TAG_TEXTO
    if border_color is None:
        border_color = TAG_VERMELHO
    return f"""
    <div class="metric-card" style="border-left: 4px solid {border_color};">
        <div class="label">{label}</div>
        <div class="value" style="color: {color};">{value}{suffix}</div>
    </div>
    """


# ============================================================
# PAGINA 1: VISAO GERAL
# ============================================================

def page_visao_geral():
    st.title("Visao Geral - Ibovespa Top 20")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acoes Monitoradas", len(df_fund))
    col2.metric("Market Cap Medio",
                format_brl(df_fund["marketCap"].mean()) if "marketCap" in df_fund.columns else "N/D")
    col3.metric("P/L Mediano",
                format_number(df_fund["trailingPE"].median(), 1) if "trailingPE" in df_fund.columns else "N/D")
    col4.metric("DY Medio",
                format_pct(df_fund["dividendYield"].mean()) if "dividendYield" in df_fund.columns else "N/D")

    st.markdown("---")

    # Tabela Resumo
    st.subheader("Tabela de Multiplos")

    cols_display = ["ticker", "nome", "setor"]
    col_map = {
        "trailingPE": "P/L", "priceToBook": "P/VP", "ev_ebitda": "EV/EBITDA",
        "dividendYield": "DY (%)", "priceToSalesTrailing12Months": "P/Receita",
        "returnOnEquity": "ROE", "profitMargins": "Margem Liq.", "marketCap": "Market Cap",
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
        df_table, width="stretch", hide_index=True, height=600,
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

    # Heatmap
    st.subheader("Heatmap de Multiplos")
    heatmap_cols = ["trailingPE", "priceToBook", "ev_ebitda", "dividendYield", "returnOnEquity"]
    heatmap_available = [c for c in heatmap_cols if c in df_fund.columns]

    if heatmap_available:
        df_heat = df_fund[["ticker", "setor"] + heatmap_available].copy()
        df_heat = df_heat.dropna(subset=heatmap_available, how="all").sort_values("setor")
        if "returnOnEquity" in df_heat.columns:
            df_heat["returnOnEquity"] = df_heat["returnOnEquity"] * 100

        df_zscore = df_heat[heatmap_available].copy()
        for col in heatmap_available:
            mean = df_zscore[col].mean()
            std = df_zscore[col].std()
            df_zscore[col] = ((df_zscore[col] - mean) / std) if std and std > 0 else 0
        for col in ["dividendYield", "returnOnEquity"]:
            if col in df_zscore.columns:
                df_zscore[col] = df_zscore[col] * -1

        labels_x = [INDICATOR_LABELS.get(c, c) for c in heatmap_available]
        tickers_y = df_heat["ticker"].apply(get_ticker_display).values

        real_values = df_heat[heatmap_available].values
        text_matrix = [[f"{val:.1f}" if pd.notna(val) else "" for val in row] for row in real_values]

        tag_colorscale = [
            [0, "#1B7A4A"], [0.25, "#52b788"], [0.5, TAG_OFFWHITE],
            [0.75, TAG_VERMELHO_CLARO], [1, TAG_VERMELHO_ESCURO],
        ]

        fig_heat = go.Figure(data=go.Heatmap(
            z=df_zscore.values, x=labels_x, y=tickers_y,
            colorscale=tag_colorscale, customdata=real_values,
            text=text_matrix, texttemplate="%{text}",
            textfont=dict(size=12, color=TAG_TEXTO, family="Inter, sans-serif"),
            hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.2f}<extra></extra>",
            showscale=True,
        ))
        fig_heat.update_layout(
            height=max(450, len(df_heat) * 32),
            template=PLOTLY_TEMPLATE,
            margin=dict(l=90, r=20, t=20, b=50),
            yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
            xaxis=dict(tickfont=dict(size=12)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig_heat, width="stretch")


# ============================================================
# PAGINA 2: MULTIPLOS & VALUATION
# ============================================================

def page_multiplos():
    st.title("Multiplos & Valuation")

    multiplo_opcoes = {
        "P/L (Trailing)": "trailingPE", "P/L (Forward)": "forwardPE",
        "P/VP": "priceToBook", "EV/EBITDA": "ev_ebitda",
        "Dividend Yield (%)": "dividendYield", "P/Receita": "priceToSalesTrailing12Months",
    }
    multiplo_selecionado = st.selectbox("Selecione o multiplo:", list(multiplo_opcoes.keys()))
    col_name = multiplo_opcoes[multiplo_selecionado]

    if col_name not in df_fund.columns:
        st.warning(f"Dado '{multiplo_selecionado}' nao disponivel.")
        return

    st.subheader(f"{multiplo_selecionado} por Acao")

    df_bar = df_fund[["ticker", "setor", col_name]].dropna(subset=[col_name]).copy()
    df_bar["ticker_display"] = df_bar["ticker"].apply(get_ticker_display)
    df_bar = df_bar.sort_values(col_name, ascending=True)

    fig_bar = px.bar(
        df_bar, x=col_name, y="ticker_display", color="setor", orientation="h",
        text=col_name, color_discrete_sequence=TAG_COLORS,
        labels={col_name: multiplo_selecionado, "ticker_display": "", "setor": "Setor"},
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                          textfont=dict(size=11))
    fig_bar.update_layout(
        height=max(450, len(df_bar) * 35), template=PLOTLY_TEMPLATE,
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=12)),
        xaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Scatter: Multiplo vs Qualidade")
        eixo_y_opcoes = {
            "ROE (%)": "returnOnEquity", "ROA (%)": "returnOnAssets",
            "Margem Liquida (%)": "profitMargins", "Margem Operacional (%)": "operatingMargins",
        }
        eixo_y_label = st.selectbox("Eixo Y:", list(eixo_y_opcoes.keys()))
        col_y = eixo_y_opcoes[eixo_y_label]

        if col_y in df_fund.columns:
            df_scatter = df_fund[["ticker", "setor", col_name, col_y, "marketCap"]].dropna().copy()
            df_scatter["ticker_display"] = df_scatter["ticker"].apply(get_ticker_display)
            df_scatter[col_y] = df_scatter[col_y] * 100

            fig_scatter = px.scatter(
                df_scatter, x=col_name, y=col_y, color="setor", text="ticker_display",
                size="marketCap", size_max=40, color_discrete_sequence=TAG_COLORS,
                labels={col_name: multiplo_selecionado, col_y: eixo_y_label, "setor": "Setor"},
            )
            fig_scatter.update_traces(textposition="top center", textfont=dict(size=10))
            med_x = df_scatter[col_name].median()
            med_y = df_scatter[col_y].median()
            fig_scatter.add_hline(y=med_y, line_dash="dash", line_color="gray", opacity=0.4,
                                  annotation_text=f"Med: {med_y:.1f}%", annotation_font_size=10)
            fig_scatter.add_vline(x=med_x, line_dash="dash", line_color="gray", opacity=0.4,
                                  annotation_text=f"Med: {med_x:.1f}", annotation_font_size=10)
            _chart_layout(fig_scatter, 500)
            st.plotly_chart(fig_scatter, width="stretch")

    with col_right:
        st.subheader("Medias Setoriais")
        df_sector = compute_sector_averages(df_fund)
        if not df_sector.empty and col_name in df_sector.columns:
            df_sector_sorted = df_sector[["setor", col_name]].dropna().sort_values(col_name)
            fig_sector = px.bar(
                df_sector_sorted, x="setor", y=col_name, text=col_name,
                color="setor", color_discrete_sequence=TAG_COLORS,
                labels={"setor": "", col_name: multiplo_selecionado},
            )
            fig_sector.update_traces(texttemplate="%{text:.2f}", textposition="outside",
                                     textfont=dict(size=11))
            fig_sector.update_layout(
                height=500, template=PLOTLY_TEMPLATE, showlegend=False,
                xaxis_tickangle=-45, xaxis=dict(tickfont=dict(size=10)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig_sector, width="stretch")


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

    # Grafico de Preco + Volume
    st.subheader("Historico de Precos")
    chart_type = st.radio("Tipo:", ["Candlestick", "Linha"], horizontal=True)
    df_prices = load_cotacoes(ticker, start_date=start_date)

    if not df_prices.empty:
        fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.05, row_heights=[0.7, 0.3])
        if chart_type == "Candlestick":
            fig_price.add_trace(go.Candlestick(
                x=df_prices["data"], open=df_prices["abertura"], high=df_prices["maxima"],
                low=df_prices["minima"], close=df_prices["fechamento"], name="OHLC",
                increasing_line_color=TAG_VERDE, decreasing_line_color=TAG_VERMELHO,
            ), row=1, col=1)
        else:
            fig_price.add_trace(go.Scatter(
                x=df_prices["data"], y=df_prices["fechamento_ajustado"],
                mode="lines", name="Preco", line=dict(color=TAG_VERMELHO, width=2),
            ), row=1, col=1)

        colors = [TAG_VERDE if c >= o else TAG_VERMELHO
                  for c, o in zip(df_prices["fechamento"], df_prices["abertura"])]
        fig_price.add_trace(go.Bar(
            x=df_prices["data"], y=df_prices["volume"],
            marker_color=colors, name="Volume", showlegend=False,
        ), row=2, col=1)

        fig_price.update_layout(
            height=550, template=PLOTLY_TEMPLATE, xaxis_rangeslider_visible=False,
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
        )
        fig_price.update_yaxes(title_text="Preco (R$)", row=1, col=1, tickfont=dict(size=11))
        fig_price.update_yaxes(title_text="Volume", row=2, col=1, tickfont=dict(size=10))
        st.plotly_chart(fig_price, width="stretch")

    st.markdown("---")

    col_ind, col_div = st.columns(2)

    with col_ind:
        st.subheader("Indicadores Fundamentalistas")
        display_indicators = [
            "trailingPE", "forwardPE", "priceToBook", "ev_ebitda",
            "dividendYield", "priceToSalesTrailing12Months",
            "marketCap", "enterpriseValue", "ebitda",
            "profitMargins", "operatingMargins", "returnOnEquity", "returnOnAssets",
            "revenueGrowth", "earningsGrowth", "totalRevenue", "netIncomeToCommon",
            "totalDebt", "totalCash", "bookValue", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        ]
        rows = []
        for ind in display_indicators:
            val = info.get(ind)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                label = INDICATOR_LABELS.get(ind, ind)
                if ind in ["marketCap", "enterpriseValue", "ebitda", "totalRevenue",
                           "netIncomeToCommon", "totalDebt", "totalCash"]:
                    formatted = format_brl(val)
                elif ind in ["profitMargins", "operatingMargins", "returnOnEquity",
                             "returnOnAssets", "revenueGrowth", "earningsGrowth"]:
                    formatted = format_pct(val * 100)
                elif ind == "dividendYield":
                    formatted = format_pct(val)
                else:
                    formatted = format_number(val, 2)
                rows.append({"Indicador": label, "Valor": formatted})

        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=500)

    with col_div:
        st.subheader("Dividendos Anuais")
        df_div = load_dividendos_annual(ticker)
        if df_div.empty:
            st.info("Sem historico de dividendos.")
        else:
            fig_div = px.bar(df_div, x="ano", y="total", text="total",
                             labels={"ano": "Ano", "total": "Total (R$)"},
                             color_discrete_sequence=[TAG_VERMELHO])
            fig_div.update_traces(texttemplate="R$ %{text:.2f}", textposition="outside",
                                  textfont=dict(size=11))
            _chart_layout(fig_div, 400)
            st.plotly_chart(fig_div, width="stretch")

        df_div_detail = load_dividendos(ticker)
        if not df_div_detail.empty:
            with st.expander("Ver historico detalhado"):
                df_show = df_div_detail[["data", "valor"]].copy()
                df_show["data"] = df_show["data"].dt.strftime("%d/%m/%Y")
                df_show.columns = ["Data", "Valor (R$)"]
                st.dataframe(df_show, width="stretch", hide_index=True)


# ============================================================
# PAGINA 4: ANALISE FUNDAMENTALISTA PROFUNDA
# ============================================================

def page_analise_fundamentalista():
    st.title("Analise Fundamentalista")
    st.caption("Highlights financeiros trimestrais: DRE, Fluxo de Caixa, Alavancagem, Liquidez, Rentabilidade e Valuation")

    ticker_options = {
        f"{get_ticker_display(t)} - {TICKER_NAMES.get(t, t)}": t
        for t in sorted(df_fund["ticker"].tolist())
    }
    selected_display = st.selectbox("Selecione a acao:", list(ticker_options.keys()), key="fund_ticker")
    ticker = ticker_options[selected_display]

    df_full = load_full_quarterly_data(ticker)
    if df_full.empty:
        st.warning("Sem dados financeiros trimestrais. Execute o extrator: `python yahoo_finance_extractor.py`")
        return

    tab_receita, tab_fluxo, tab_alav, tab_icr, tab_liq, tab_divida, tab_rent, tab_val = st.tabs([
        "Receita e Margens", "Fluxos de Caixa", "Alavancagem", "ICR e Cobertura",
        "Liquidez", "Divida CP e LP", "Rentabilidade", "Valuation Hist.",
    ])

    # ===================== ABA 1: RECEITA E MARGENS =====================
    with tab_receita:
        df_rm = compute_receita_margens(df_full)
        abs_cols = [c for c in ["Receita Liquida", "Lucro Bruto", "EBITDA", "Lucro Operacional", "Lucro Liquido"]
                    if c in df_rm.columns]
        margin_cols = [c for c in ["Margem Bruta (%)", "Margem EBITDA (%)", "Margem Operacional (%)", "Margem Liquida (%)"]
                       if c in df_rm.columns]

        if abs_cols:
            col1, col2 = st.columns(2)
            with col1:
                fig_rev = make_subplots(specs=[[{"secondary_y": True}]])
                if "Receita Liquida" in df_rm.columns:
                    fig_rev.add_trace(go.Bar(
                        x=df_rm["data"], y=df_rm["Receita Liquida"],
                        name="Receita", marker_color="rgba(99, 13, 36, 0.25)", opacity=0.8,
                    ), secondary_y=False)
                margin_colors = [TAG_VERMELHO, TAG_DOURADO, TAG_VERDE, "#4A0A1B"]
                for i, mc in enumerate(margin_cols):
                    fig_rev.add_trace(go.Scatter(
                        x=df_rm["data"], y=df_rm[mc], name=mc.replace(" (%)", ""),
                        mode="lines+markers", line=dict(color=margin_colors[i % len(margin_colors)], width=2.5),
                        marker=dict(size=6),
                    ), secondary_y=True)
                _chart_layout(fig_rev, 450, "Receita e Margens")
                fig_rev.update_yaxes(title_text="R$", tickformat=",.0f", secondary_y=False, titlefont=dict(size=11))
                fig_rev.update_yaxes(title_text="%", secondary_y=True, titlefont=dict(size=11))
                st.plotly_chart(fig_rev, width="stretch")

            with col2:
                df_abs = _melt_for_chart(df_rm[["data"] + abs_cols])
                if not df_abs.empty:
                    fig_abs = _make_chart(df_abs, "bar", 450, "R$", TAG_COLORS, ",.0f", title="Receita vs Lucros")
                    st.plotly_chart(fig_abs, width="stretch")

        growth_cols = [c for c in ["Cresc. Receita YoY (%)", "Cresc. Lucro YoY (%)", "Cresc. EBITDA YoY (%)"]
                       if c in df_rm.columns]
        if growth_cols:
            st.markdown("---")
            df_growth = _melt_for_chart(df_rm[["data"] + growth_cols])
            if not df_growth.empty:
                fig_g = _make_chart(df_growth, "line", 350, "%", TAG_COLORS, title="Crescimento YoY (%)")
                fig_g.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
                st.plotly_chart(fig_g, width="stretch")

    # ===================== ABA 2: FLUXOS DE CAIXA =====================
    with tab_fluxo:
        df_fc = compute_fluxo_caixa(df_full)
        col1, col2 = st.columns(2)
        with col1:
            flow_cols = [c for c in ["FCO", "Capex", "FCF"] if c in df_fc.columns]
            if flow_cols:
                df_flow = _melt_for_chart(df_fc[["data"] + flow_cols])
                fig_f = _make_chart(df_flow, "bar", 420, "R$", [TAG_VERMELHO, TAG_DOURADO, TAG_VERDE],
                                    ",.0f", title="FCO, Capex e FCF")
                fig_f.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
                st.plotly_chart(fig_f, width="stretch")

        with col2:
            all_flows = [c for c in ["FCO", "FCI", "FCF Financiamento"] if c in df_fc.columns]
            if all_flows:
                df_flow2 = _melt_for_chart(df_fc[["data"] + all_flows])
                fig_f2 = _make_chart(df_flow2, "bar", 420, "R$", [TAG_VERMELHO, TAG_VERDE, TAG_DOURADO],
                                     ",.0f", title="FCO, FCI e Financiamento")
                fig_f2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
                st.plotly_chart(fig_f2, width="stretch")

        conv_cols = [c for c in ["FCO/EBITDA (%)", "FCF/Lucro Liq (%)", "Capex/Receita (%)"] if c in df_fc.columns]
        if conv_cols:
            st.markdown("---")
            df_conv = _melt_for_chart(df_fc[["data"] + conv_cols])
            fig_cv = _make_chart(df_conv, "line", 350, "%", TAG_COLORS, title="Conversao de Caixa")
            fig_cv.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.3, annotation_text="100%")
            st.plotly_chart(fig_cv, width="stretch")

    # ===================== ABA 3: ALAVANCAGEM =====================
    with tab_alav:
        df_alav = compute_alavancagem(df_full)
        col1, col2 = st.columns(2)
        with col1:
            if "Divida Liq/EBITDA" in df_alav.columns:
                fig_lev = go.Figure()
                vals = df_alav["Divida Liq/EBITDA"]
                fig_lev.add_trace(go.Bar(
                    x=df_alav["data"], y=vals, name="Div.Liq/EBITDA",
                    marker_color=[TAG_VERMELHO if v and v > 3 else TAG_DOURADO if v and v > 2 else TAG_VERDE for v in vals],
                    text=vals.apply(lambda x: f"{x:.2f}x" if pd.notna(x) else ""),
                    textposition="outside", textfont=dict(size=11, family="Inter"),
                ))
                fig_lev.add_hline(y=3, line_dash="dash", line_color=TAG_VERMELHO, opacity=0.5,
                                  annotation_text="Alerta 3x", annotation_font=dict(size=10))
                _chart_layout(fig_lev, 420, "Divida Liquida / EBITDA", False)
                st.plotly_chart(fig_lev, width="stretch")

        with col2:
            ec_cols = [c for c in ["Estrutura Capital (%)", "Divida Bruta/PL", "Divida Liq/PL"] if c in df_alav.columns]
            if ec_cols:
                df_ec = _melt_for_chart(df_alav[["data"] + ec_cols])
                fig_ec = _make_chart(df_ec, "line", 420, "", TAG_COLORS, title="Estrutura de Capital")
                st.plotly_chart(fig_ec, width="stretch")

    # ===================== ABA 4: ICR E COBERTURA =====================
    with tab_icr:
        df_icr = compute_icr_cobertura(df_full)
        icr_cols = [c for c in ["Cobertura de Juros (ICR)", "FCO/Desp. Financeira", "EBIT/Desp. Financeira"]
                    if c in df_icr.columns]
        if icr_cols:
            fig_icr = go.Figure()
            colors_icr = [TAG_VERMELHO, TAG_DOURADO, TAG_VERDE]
            for i, col in enumerate(icr_cols):
                fig_icr.add_trace(go.Scatter(
                    x=df_icr["data"], y=df_icr[col], name=col.replace(" (ICR)", ""),
                    mode="lines+markers+text", line=dict(color=colors_icr[i], width=2.5),
                    text=df_icr[col].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else ""),
                    textposition="top center", textfont=dict(size=10),
                    marker=dict(size=7),
                ))
            fig_icr.add_hline(y=1, line_dash="dash", line_color=TAG_VERMELHO, opacity=0.5,
                              annotation_text="Critico (1x)", annotation_font=dict(size=10))
            fig_icr.add_hline(y=2, line_dash="dot", line_color=TAG_DOURADO, opacity=0.3,
                              annotation_text="Atencao (2x)", annotation_font=dict(size=10))
            _chart_layout(fig_icr, 450, "Cobertura de Juros e FCO")
            st.plotly_chart(fig_icr, width="stretch")

            if "Cobertura de Juros (ICR)" in df_icr.columns:
                last_icr = df_icr["Cobertura de Juros (ICR)"].dropna()
                if not last_icr.empty:
                    v = last_icr.iloc[-1]
                    if v > 3:
                        st.success(f"ICR = {v:.2f}x — Cobertura confortavel")
                    elif v > 1.5:
                        st.warning(f"ICR = {v:.2f}x — Cobertura moderada")
                    else:
                        st.error(f"ICR = {v:.2f}x — Cobertura insuficiente")
        else:
            st.info("Sem dados de despesas financeiras disponiveis.")

    # ===================== ABA 5: LIQUIDEZ =====================
    with tab_liq:
        df_liq = compute_liquidez(df_full)
        col1, col2 = st.columns(2)
        with col1:
            liq_cols = [c for c in ["Liquidez Corrente", "Liquidez Seca", "Liquidez Imediata"] if c in df_liq.columns]
            if liq_cols:
                fig_liq = go.Figure()
                liq_colors = [TAG_VERMELHO, TAG_DOURADO, TAG_VERDE]
                for i, col in enumerate(liq_cols):
                    fig_liq.add_trace(go.Scatter(
                        x=df_liq["data"], y=df_liq[col], name=col,
                        mode="lines+markers", line=dict(color=liq_colors[i], width=2.5),
                        marker=dict(size=6),
                    ))
                fig_liq.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.4, annotation_text="Ref. 1x")
                _chart_layout(fig_liq, 420, "Indices de Liquidez")
                st.plotly_chart(fig_liq, width="stretch")

        with col2:
            comp_cols = [c for c in ["Caixa e Equivalentes", "Contas a Receber", "Estoques"] if c in df_liq.columns]
            if comp_cols:
                fig_comp = go.Figure()
                comp_colors = [TAG_VERDE, TAG_DOURADO, TAG_VERMELHO]
                for i, col in enumerate(comp_cols):
                    fig_comp.add_trace(go.Bar(
                        x=df_liq["data"], y=df_liq[col], name=col, marker_color=comp_colors[i],
                    ))
                _chart_layout(fig_comp, 420, "Composicao Ativo Circulante")
                fig_comp.update_layout(barmode="stack")
                fig_comp.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig_comp, width="stretch")

    # ===================== ABA 6: DIVIDA CP E LP =====================
    with tab_divida:
        df_div = compute_divida_composicao(df_full)
        col1, col2 = st.columns(2)
        with col1:
            stack_cols = [c for c in ["Caixa e Equivalentes", "EFCP", "EFLP"] if c in df_div.columns]
            if stack_cols:
                fig_d = go.Figure()
                d_colors = [TAG_VERDE, TAG_DOURADO, TAG_VERMELHO]
                for i, col in enumerate(stack_cols):
                    fig_d.add_trace(go.Bar(
                        x=df_div["data"], y=df_div[col], name=col, marker_color=d_colors[i],
                    ))
                _chart_layout(fig_d, 420, "Liquidez vs Divida (CP + LP)")
                fig_d.update_layout(barmode="stack")
                fig_d.update_yaxes(tickformat=",.0f")
                st.plotly_chart(fig_d, width="stretch")

        with col2:
            if "EFCP" in df_div.columns and "EFLP" in df_div.columns:
                df_pct = df_div[["data", "EFCP", "EFLP"]].dropna()
                if not df_pct.empty:
                    total = df_pct["EFCP"].abs() + df_pct["EFLP"].abs()
                    df_pct["% CP"] = (df_pct["EFCP"].abs() / total * 100)
                    df_pct["% LP"] = (df_pct["EFLP"].abs() / total * 100)
                    fig_pct = go.Figure()
                    fig_pct.add_trace(go.Bar(
                        x=df_pct["data"], y=df_pct["% CP"], name="EFCP", marker_color=TAG_DOURADO,
                        text=df_pct["% CP"].apply(lambda x: f"{x:.0f}%"), textposition="inside",
                        textfont=dict(size=12, color="white"),
                    ))
                    fig_pct.add_trace(go.Bar(
                        x=df_pct["data"], y=df_pct["% LP"], name="EFLP", marker_color=TAG_VERMELHO,
                        text=df_pct["% LP"].apply(lambda x: f"{x:.0f}%"), textposition="inside",
                        textfont=dict(size=12, color="white"),
                    ))
                    _chart_layout(fig_pct, 420, "Composicao da Divida (% CP vs LP)")
                    fig_pct.update_layout(barmode="stack", yaxis=dict(range=[0, 100], ticksuffix="%"))
                    st.plotly_chart(fig_pct, width="stretch")

        debt_cols = [c for c in ["Divida Bruta", "Divida Liquida"] if c in df_div.columns]
        if debt_cols:
            st.markdown("---")
            df_debt = _melt_for_chart(df_div[["data"] + debt_cols])
            fig_debt = _make_chart(df_debt, "line", 350, "R$", [TAG_VERMELHO, TAG_DOURADO], ",.0f",
                                   title="Evolucao da Divida")
            st.plotly_chart(fig_debt, width="stretch")

    # ===================== ABA 7: RENTABILIDADE =====================
    with tab_rent:
        df_rent = compute_rentabilidade(df_full)
        col1, col2 = st.columns(2)
        with col1:
            ret_cols = [c for c in ["ROE (%)", "ROA (%)", "ROIC (%)"] if c in df_rent.columns]
            if ret_cols:
                df_ret = _melt_for_chart(df_rent[["data"] + ret_cols])
                fig_ret = _make_chart(df_ret, "line", 420, "%", [TAG_VERMELHO, TAG_DOURADO, TAG_VERDE],
                                      title="ROE, ROA e ROIC")
                fig_ret.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)
                st.plotly_chart(fig_ret, width="stretch")

        with col2:
            dupont_cols = [c for c in ["DuPont: Margem Liq (%)", "DuPont: Giro Ativo", "DuPont: Alavancagem"]
                          if c in df_rent.columns]
            if dupont_cols:
                fig_dp = make_subplots(specs=[[{"secondary_y": True}]])
                if "DuPont: Margem Liq (%)" in df_rent.columns:
                    fig_dp.add_trace(go.Bar(
                        x=df_rent["data"], y=df_rent["DuPont: Margem Liq (%)"],
                        name="Margem Liq (%)", marker_color="rgba(99, 13, 36, 0.3)",
                    ), secondary_y=False)
                if "DuPont: Giro Ativo" in df_rent.columns:
                    fig_dp.add_trace(go.Scatter(
                        x=df_rent["data"], y=df_rent["DuPont: Giro Ativo"],
                        name="Giro Ativo", mode="lines+markers",
                        line=dict(color=TAG_DOURADO, width=2.5), marker=dict(size=6),
                    ), secondary_y=True)
                if "DuPont: Alavancagem" in df_rent.columns:
                    fig_dp.add_trace(go.Scatter(
                        x=df_rent["data"], y=df_rent["DuPont: Alavancagem"],
                        name="Alavancagem", mode="lines+markers",
                        line=dict(color=TAG_VERDE, width=2.5), marker=dict(size=6),
                    ), secondary_y=True)
                _chart_layout(fig_dp, 420, "Analise DuPont (3 fatores)")
                fig_dp.update_yaxes(title_text="Margem (%)", secondary_y=False, titlefont=dict(size=11))
                fig_dp.update_yaxes(title_text="Giro / Alav. (x)", secondary_y=True, titlefont=dict(size=11))
                st.plotly_chart(fig_dp, width="stretch")

        if "Taxa Efetiva IR (%)" in df_rent.columns:
            st.markdown("---")
            fig_tax = go.Figure()
            fig_tax.add_trace(go.Bar(
                x=df_rent["data"], y=df_rent["Taxa Efetiva IR (%)"],
                marker_color=[TAG_VERMELHO if v and v > 34 else TAG_DOURADO if v and v > 25 else TAG_VERDE
                              for v in df_rent["Taxa Efetiva IR (%)"]],
                text=df_rent["Taxa Efetiva IR (%)"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else ""),
                textposition="outside", textfont=dict(size=11), name="Taxa IR",
            ))
            fig_tax.add_hline(y=34, line_dash="dash", line_color="gray", opacity=0.4,
                              annotation_text="Nominal 34%", annotation_font=dict(size=10))
            _chart_layout(fig_tax, 320, "Taxa Efetiva de IR", False)
            st.plotly_chart(fig_tax, width="stretch")

    # ===================== ABA 8: VALUATION HISTORICO =====================
    with tab_val:
        df_val = compute_valuation_historico(df_full, ticker)
        if df_val.empty:
            st.info("Sem dados suficientes para calcular multiplos historicos.")
        else:
            mult_cols = [c for c in ["P/L", "P/VP", "EV/EBITDA", "P/Receita", "P/FCF"] if c in df_val.columns]
            if mult_cols:
                # 2 colunas, 3 graficos cada
                ncols = 2
                cols = st.columns(ncols)
                for idx, mc in enumerate(mult_cols):
                    vals = df_val[mc].dropna()
                    if vals.empty:
                        continue
                    with cols[idx % ncols]:
                        fig_m = go.Figure()
                        color = TAG_VERMELHO if idx % 2 == 0 else TAG_VERDE
                        fig_m.add_trace(go.Scatter(
                            x=df_val["data"], y=df_val[mc], mode="lines+markers", name=mc,
                            line=dict(color=color, width=2.5), marker=dict(size=5),
                            fill="tozeroy", fillcolor=f"rgba({','.join(str(int(color[i:i+2], 16)) for i in (1,3,5))}, 0.08)",
                        ))
                        median_val = vals.median()
                        fig_m.add_hline(y=median_val, line_dash="dash", line_color=TAG_DOURADO,
                                        opacity=0.6, annotation_text=f"Mediana: {median_val:.1f}",
                                        annotation_font=dict(size=10))
                        _chart_layout(fig_m, 300, mc, False)
                        st.plotly_chart(fig_m, width="stretch")

                yield_cols = [c for c in ["Earnings Yield (%)", "FCF Yield (%)"] if c in df_val.columns]
                if yield_cols:
                    st.markdown("---")
                    df_yield = _melt_for_chart(df_val[["data"] + yield_cols])
                    if not df_yield.empty:
                        fig_y = _make_chart(df_yield, "line", 300, "%", [TAG_VERMELHO, TAG_VERDE],
                                            title="Earnings Yield e FCF Yield")
                        st.plotly_chart(fig_y, width="stretch")


# ============================================================
# PAGINA 5: SCORE & RATING
# ============================================================

def page_score_rating():
    st.title("Score & Rating Fundamentalista")
    st.caption("Analise quantitativa da qualidade e saude financeira")

    ticker_options = {
        f"{get_ticker_display(t)} - {TICKER_NAMES.get(t, t)}": t
        for t in sorted(df_fund["ticker"].tolist())
    }
    selected_display = st.selectbox("Selecione a acao:", list(ticker_options.keys()), key="score_ticker")
    ticker = ticker_options[selected_display]

    df_full = load_full_quarterly_data(ticker)
    if df_full.empty:
        st.warning("Sem dados financeiros trimestrais.")
        return

    # --- PIOTROSKI F-SCORE ---
    st.subheader("Piotroski F-Score")
    st.caption("Score de 0 a 9: lucratividade, alavancagem e eficiencia operacional")

    piotroski = compute_piotroski_score(df_full)
    score = piotroski["score"]
    details = piotroski["details"]

    if score is not None:
        col_score, col_gauge = st.columns([1, 2])
        with col_score:
            if score >= 7:
                color, label = TAG_VERDE, "FORTE"
            elif score >= 4:
                color, label = TAG_DOURADO, "NEUTRO"
            else:
                color, label = TAG_VERMELHO, "FRACO"

            st.markdown(f"""
            <div style="text-align: center; padding: 24px 0;">
                <div style="font-size: 4.5rem; font-weight: 700; color: {color}; line-height: 1;
                            font-family: Inter, sans-serif;">{score}/9</div>
                <div style="margin-top: 8px;">
                    <span class="score-badge" style="background: {color}; color: white;">{label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 9], "tickwidth": 1, "tickfont": dict(size=12)},
                    "bar": {"color": color, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 3], "color": "rgba(99, 13, 36, 0.12)"},
                        {"range": [3, 6], "color": "rgba(184, 134, 11, 0.12)"},
                        {"range": [6, 9], "color": "rgba(27, 122, 74, 0.12)"},
                    ],
                    "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": score},
                },
                number={"font": {"size": 36, "family": "Inter", "color": color}},
                title={"text": "Piotroski F-Score", "font": {"size": 14, "family": "Inter"}},
            ))
            fig_gauge.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig_gauge, width="stretch")

        # Detalhamento
        st.markdown("---")
        st.subheader("Detalhamento dos Criterios")

        categories = {
            "Lucratividade": ["ROA Positivo", "FCO Positivo", "ROA Crescente", "FCO > Lucro (Accruals)"],
            "Alavancagem / Liquidez": ["Alavancagem Decrescente", "Liquidez Crescente", "Sem Diluicao"],
            "Eficiencia Operacional": ["Margem Bruta Crescente", "Giro Ativo Crescente"],
        }

        for cat_name, criteria in categories.items():
            st.markdown(f"**{cat_name}**")
            for criterion in criteria:
                if criterion in details:
                    d = details[criterion]
                    icon = "✅" if d["passed"] else "❌"
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{icon}&nbsp;&nbsp;{criterion}")
            st.markdown("")
    else:
        st.info("Dados insuficientes (minimo 2 trimestres).")

    st.markdown("---")

    # --- RESUMO SAUDE FINANCEIRA ---
    st.subheader("Resumo de Saude Financeira")

    df_rm = compute_receita_margens(df_full)
    df_rent = compute_rentabilidade(df_full)
    df_alav = compute_alavancagem(df_full)
    df_liq = compute_liquidez(df_full)
    df_icr_data = compute_icr_cobertura(df_full)

    def _last(df, col):
        if col in df.columns:
            v = df[col].dropna()
            return v.iloc[-1] if not v.empty else None
        return None

    metrics = [
        ("Margem Liquida", _last(df_rm, "Margem Liquida (%)"), "%", 10, 5, False),
        ("Margem EBITDA", _last(df_rm, "Margem EBITDA (%)"), "%", 20, 10, False),
        ("ROE", _last(df_rent, "ROE (%)"), "%", 15, 8, False),
        ("ROIC", _last(df_rent, "ROIC (%)"), "%", 10, 5, False),
        ("Div.Liq/EBITDA", _last(df_alav, "Divida Liq/EBITDA"), "x", 2, 3, True),
        ("Liquidez Corr.", _last(df_liq, "Liquidez Corrente"), "x", 1.5, 1, False),
        ("ICR", _last(df_icr_data, "Cobertura de Juros (ICR)"), "x", 3, 1.5, False),
        ("ROA", _last(df_rent, "ROA (%)"), "%", 5, 2, False),
    ]

    cols = st.columns(4)
    for i, (name, value, suffix, good, warn, invert) in enumerate(metrics):
        with cols[i % 4]:
            if value is not None and not np.isnan(value):
                if invert:
                    color = TAG_VERDE if value < good else (TAG_DOURADO if value < warn else TAG_VERMELHO)
                else:
                    color = TAG_VERDE if value > good else (TAG_DOURADO if value > warn else TAG_VERMELHO)
                st.markdown(_metric_card(name, f"{value:.2f}", suffix, color, color), unsafe_allow_html=True)
            else:
                st.markdown(_metric_card(name, "N/D", "", "#999", "#ccc"), unsafe_allow_html=True)

    st.markdown("---")

    # --- BALANCO PATRIMONIAL ---
    st.subheader("Balanco Patrimonial")
    df_bal = compute_balanco_patrimonial(df_full)

    bal_cols = [c for c in ["Ativo Total", "Passivo Total", "Patrimonio Liquido"] if c in df_bal.columns]
    if bal_cols:
        df_bal_m = _melt_for_chart(df_bal[["data"] + bal_cols])
        fig_bal = _make_chart(df_bal_m, "bar", 400, "R$", TAG_COLORS, ",.0f",
                              title="Ativo, Passivo e Patrimonio Liquido")
        st.plotly_chart(fig_bal, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        pct_cols = [c for c in ["% Ativo Circulante", "% Ativo Nao Circulante"] if c in df_bal.columns]
        if pct_cols:
            fig_c = go.Figure()
            if "% Ativo Circulante" in df_bal.columns:
                fig_c.add_trace(go.Bar(x=df_bal["data"], y=df_bal["% Ativo Circulante"],
                                       name="Circulante", marker_color=TAG_DOURADO))
            if "% Ativo Nao Circulante" in df_bal.columns:
                fig_c.add_trace(go.Bar(x=df_bal["data"], y=df_bal["% Ativo Nao Circulante"],
                                       name="Nao Circulante", marker_color=TAG_VERMELHO))
            _chart_layout(fig_c, 350, "Composicao do Ativo (%)")
            fig_c.update_layout(barmode="stack", yaxis=dict(range=[0, 100], ticksuffix="%"))
            st.plotly_chart(fig_c, width="stretch")

    with col2:
        wc_cols = [c for c in ["Capital de Giro", "Lucros Acumulados"] if c in df_bal.columns]
        if wc_cols:
            df_wc = _melt_for_chart(df_bal[["data"] + wc_cols])
            fig_wc = _make_chart(df_wc, "line", 350, "R$", [TAG_VERMELHO, TAG_VERDE], ",.0f",
                                 title="Capital de Giro e Lucros Acum.")
            st.plotly_chart(fig_wc, width="stretch")


# ============================================================
# PAGINA 6: COMPARACAO
# ============================================================

def page_comparacao():
    st.title("Comparacao de Acoes")

    display_options = {get_ticker_display(t): t for t in IBOVESPA_TOP20}
    selected_displays = st.multiselect(
        "Selecione 2 a 5 acoes:", list(display_options.keys()),
        default=["PETR4", "VALE3"], max_selections=5,
    )
    selected_tickers = [display_options[d] for d in selected_displays]

    if len(selected_tickers) < 2:
        st.warning("Selecione pelo menos 2 acoes.")
        return

    periodo_map = {"3M": 90, "6M": 180, "1A": 365, "2A": 730, "5A": 1825}
    periodo = st.radio("Periodo:", list(periodo_map.keys()), horizontal=True, index=2)
    start_date = (datetime.now() - timedelta(days=periodo_map[periodo])).strftime("%Y-%m-%d")

    st.markdown("---")

    # Performance Normalizada
    st.subheader("Performance Relativa (Base 100)")
    df_multi = load_cotacoes_multi(selected_tickers, start_date=start_date)
    if not df_multi.empty:
        df_norm = normalize_prices(df_multi)
        df_norm["ticker_display"] = df_norm["ticker"].apply(get_ticker_display)
        fig_perf = px.line(
            df_norm, x="data", y="preco_normalizado", color="ticker_display",
            color_discrete_sequence=TAG_COLORS,
            labels={"data": "", "preco_normalizado": "Retorno (Base 100)", "ticker_display": ""},
        )
        fig_perf.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.3)
        _chart_layout(fig_perf, 450)
        st.plotly_chart(fig_perf, width="stretch")

    st.markdown("---")

    # Comparacao de Multiplos
    st.subheader("Comparacao de Multiplos")
    comparison_metrics = {"trailingPE": "P/L", "priceToBook": "P/VP", "ev_ebitda": "EV/EBITDA", "dividendYield": "DY (%)"}
    available_metrics = {k: v for k, v in comparison_metrics.items() if k in df_fund.columns}
    df_comp = df_fund[df_fund["ticker"].isin(selected_tickers)][["ticker"] + list(available_metrics.keys())].copy()
    df_comp["ticker"] = df_comp["ticker"].apply(get_ticker_display)

    df_melted = df_comp.melt(id_vars="ticker", var_name="multiplo", value_name="valor")
    df_melted["multiplo"] = df_melted["multiplo"].map(available_metrics)

    fig_comp = px.bar(
        df_melted.dropna(subset=["valor"]), x="multiplo", y="valor", color="ticker",
        barmode="group", text="valor", color_discrete_sequence=TAG_COLORS,
        labels={"multiplo": "", "valor": "", "ticker": ""},
    )
    fig_comp.update_traces(texttemplate="%{text:.2f}", textposition="outside", textfont=dict(size=11))
    _chart_layout(fig_comp, 450)
    st.plotly_chart(fig_comp, width="stretch")

    st.markdown("---")

    # Tabela Detalhada
    st.subheader("Tabela Comparativa")
    detail_cols = {
        "nome": "Nome", "setor": "Setor", "trailingPE": "P/L", "forwardPE": "P/L (Fwd)",
        "priceToBook": "P/VP", "ev_ebitda": "EV/EBITDA", "dividendYield": "DY (%)",
        "returnOnEquity": "ROE", "profitMargins": "Margem Liq.", "marketCap": "Market Cap",
    }
    available_detail = {k: v for k, v in detail_cols.items() if k in df_fund.columns}
    df_detail = df_fund[df_fund["ticker"].isin(selected_tickers)][["ticker"] + list(available_detail.keys())].copy()
    for pct_col in ["returnOnEquity", "profitMargins"]:
        if pct_col in df_detail.columns:
            df_detail[pct_col] = df_detail[pct_col] * 100
    if "marketCap" in df_detail.columns:
        df_detail["marketCap"] = df_detail["marketCap"].apply(format_brl)
    df_detail["ticker"] = df_detail["ticker"].apply(get_ticker_display)
    df_detail = df_detail.set_index("ticker").T
    df_detail.index = [available_detail.get(idx, idx) for idx in df_detail.index]
    st.dataframe(df_detail, width="stretch")


# ============================================================
# ROTEAMENTO
# ============================================================

if pagina == "Visao Geral":
    page_visao_geral()
elif pagina == "Multiplos & Valuation":
    page_multiplos()
elif pagina == "Analise Individual":
    page_analise_individual()
elif pagina == "Analise Fundamentalista":
    page_analise_fundamentalista()
elif pagina == "Score & Rating":
    page_score_rating()
elif pagina == "Comparacao":
    page_comparacao()
