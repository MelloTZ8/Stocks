import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import feedparser
import time
from datetime import datetime, timedelta
import pandas_datareader.data as web

# --- 1. PAGE CONFIG & STATE ---
st.set_page_config(page_title="Macro Dash Master", layout="wide")

# Cloud-Safe Session State Init
if "pg" not in st.session_state:
    st.session_state.pg = 0

# --- 2. ASSET BUCKETS & MAPS (Dash 1.0) ---
categories = {
    "B: Digital Assets": {'IBIT': '₿ IBIT', 'MSTR': '₿ MSTR'},
    "C: Tech": {'MSFT': '💻 MSFT', 'AAPL': '💻 AAPL', 'GOOGL': '💻 GOOGL'},
    "D: Semis": {'NVDA': '📟 NVDA', 'AMD': '📟 AMD', 'AVGO': '📟 AVGO'},
    "A: Indicators": {'TIP': '🟦 TIP', 'TLT': '🟦 TLT', 'DX-Y.NYB': '🟦 DXY', '^TNX': '🟦 10Y', '^VIX': '💀 VIX'},
    "E: Energy/Inf": {'XLE': '🔥 XLE', 'XOM': '🔥 XOM', 'DBC': '🔥 DBC'},
    "F: Financials": {'KBE': '🏦 KBE', 'JPM': '🏦 JPM', 'BRK-B': '🏦 BRK'},
    "G: Defensive": {'SCHD': '🛡️ SCHD', 'WMT': '🛡️ WMT', 'PG': '🛡️ PG'},
}

color_map = {
    '🟦 TIP': '#0000FF', '🟦 TLT': '#4169E1', '🟦 DXY': '#00BFFF', '🟦 10Y': '#191970', '💀 VIX': '#708090',
    '₿ IBIT': '#FF8C00', '₿ MSTR': '#E67E22', '💻 MSFT': '#00FFFF', '💻 AAPL': '#20B2AA', '💻 GOOGL': '#48D1CC',
    '📟 NVDA': '#FFD700', '📟 AMD': '#DAA520', '📟 AVGO': '#BDB76B', '🔥 XLE': '#FF0000', '🔥 XOM': '#DC143C', 
    '🔥 DBC': '#8B0000', '🟡 GLD': '#FFD700', '🏦 KBE': '#800080', '🏦 JPM': '#9370DB', '🏦 BRK': '#4B0082',
    '🛡️ SCHD': '#008000', '🛡️ WMT': '#32CD32', '🛡️ PG': '#228B22',
}

category_emojis = {
    "Indicators": "📉", "Digital Assets": "₿", "Tech": "💻", 
    "Semis": "📟", "Energy/Inf": "🔥", "Financials": "🏦", "Defensive": "🛡️"
}

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("📐 Correlation Decoder")
with st.sidebar.expander("Detailed Quartile Interpretation", expanded=True):
    st.markdown("""
    **🔴 Positive (Synergy)**
    * **1.00:** Perfect Lockstep
    * **0.75 - 0.99:** Strong Synergy
    * **0.50 - 0.74:** Moderate Driver
    * **0.25 - 0.49:** Weak/Positive Drift
    * **0.00 - 0.24:** Noise/Neutral

    **🔵 Negative (Seesaw)**
    * **-1.00:** Perfect Inverse
    * **-0.75 to -0.99:** Strong Inverse
    * **-0.50 to -0.74:** Strong Hedge
    * **-0.25 to -0.49:** Weak Divergence
    * **0.00 to -0.24:** Noise/Neutral
    """)

st.sidebar.divider()
st.sidebar.header("🕒 Dashboard Controls")
day_offset = st.sidebar.slider("Snapshot Day (Market Analysis)", 0, 252, 0)
comp_lookback = st.sidebar.slider("Comparison Start (Market Analysis)", 2, 252, 60)
window_size = st.sidebar.slider("Correlation Sensitivity", 5, 60, 21)

st.sidebar.header("🗞️ Terminal Feed")
active_feeds = {
    "CNBC Alerts": st.sidebar.checkbox("CNBC Breaking", True),
    "MarketWatch": st.sidebar.checkbox("MarketPulse", True),
    "Yahoo": st.sidebar.checkbox("Yahoo Finance", True),
    "FT": st.sidebar.checkbox("Financial Times", True)
}

# --- 4. DATA LOADING (Cloud-Hardened) ---
@st.cache_data(ttl=3600)
def load_market_data(ticker_list):
    for attempt in range(3):
        try:
            df = yf.download(ticker_list + ['GLD'], period="3y", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df = df['Close']
            if not df.empty:
                if '^TNX' in df.columns: df['^TNX'] = df['^TNX'] / 10
                return df.dropna()
        except: time.sleep(2)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_bond_data():
    end = datetime.today().date()
    start = datetime(1994, 1, 1).date()
    tickers = ['DGS3MO', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
    df = web.DataReader(tickers, 'fred', start, end)
    df.columns = ['3-Month', '2-Year', '5-Year', '10-Year', '30-Year']
    df.dropna(inplace=True)
    
    # Calculate Spreads
    df['10Y_3M_Spread'] = df['10-Year'] - df['3-Month']
    df['10Y_2Y_Spread'] = df['10-Year'] - df['2-Year']
    df['30Y_5Y_Spread'] = df['30-Year'] - df['5-Year']
    return df

# Initialize Data
all_tickers = list(set([t for c in categories.values() for t in c.keys()] + ['GLD']))
raw_data = load_market_data(all_tickers)
rename_map = {t: l for cat in categories.values() for t, l in cat.items()}
rename_map['GLD'] = '🟡 GLD'
bond_df = load_bond_data()

if raw_data.empty or bond_df.empty:
    st.error("📡 API rate limit hit or unavailable. Please wait a few moments and refresh.")
    st.stop()

# --- 5. UTILITIES ---
def get_perf_and_trend(series, days):
    try:
        start_val = series.iloc[-days]; end_val = series.iloc[-1]
        pct = ((end_val - start_val) / start_val) * 100
        return f"{pct:.2f}%", "🟢" if pct > 0 else "🔴"
    except: return "0.00%", "⚪"

def draw_gauge(title, val, col, key):
    color = f"rgb(255, {int(255*(1-val))}, {int(255*(1-val))})" if val > 0 else f"rgb({int(255*(1+val))}, {int(255*(1+val))}, 255)"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14}}, number={'font': {'size': 24}},
        gauge={'axis': {'range': [-1, 1], 'tickvals': [-1, -0.5, 0, 0.5, 1]}, 'bar': {'color': color}, 'bgcolor': "#262626", 'borderwidth': 1}))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=50, b=20), paper_bgcolor="#0E1117", font={'color': "white"})
    col.plotly_chart(fig, use_container_width=True, key=key)

# --- 6. PROCESSING DASH 1.0 ---
target_idx = (len(raw_data) - 1) - day_offset
snapshot_date = raw_data.index[target_idx]
full_window = raw_data.iloc[:target_idx + 1]
corr_window = full_window.iloc[-window_size:].rename(columns=rename_map)
ordered_labels = [rename_map[t] for cat in categories.values() for t in cat.keys()] + ['🟡 GLD']
corr_matrix = corr_window[ordered_labels].corr()

# --- 7. TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "📈 Macro Bond Watch", "📰 Terminal Feed"])

# ==========================================
# TAB 1: MARKET ANALYSIS
# ==========================================
with tab1:
    st.subheader(f"1. Performance Snapshot: {snapshot_date.strftime('%Y-%m-%d')}")
    
    with st.expander("⚙️ Filter Chart Assets by Category", expanded=False):
        selected_chart_labels = []
        cols = st.columns(len(categories))
        for i, (cat, assets) in enumerate(categories.items()):
            with cols[i]:
                st.markdown(f"**{cat.split(': ')[1]}**")
                for t, l in assets.items():
                    if st.checkbox(l, value=True, key=f"toggle_{t}"):
                        selected_chart_labels.append(l)

    if not selected_chart_labels:
        st.warning("Please select at least one asset to display the chart.")
    else:
        perf_window = full_window.iloc[max(0, len(full_window)-comp_lookback):].rename(columns=rename_map)
        norm_data = (perf_window[selected_chart_labels] / perf_window[selected_chart_labels].iloc[0]) * 100
        fig_line = px.line(norm_data, log_y=True, template="plotly_dark", color_discrete_map=color_map)
        fig_line.update_layout(hovermode='closest')
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("2. Market DNA Heatmap 🔥")
    fig_corr = px.imshow(corr_matrix.mask(np.triu(np.ones_like(corr_matrix, dtype=bool))), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1], template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()
    st.subheader("3. Macro Command Center: Gauges")
    anchors = [
        ('TLT', "📉 Bonds"), ('^TNX', "🏛️ 10Y Yield"), ('DX-Y.NYB', "💵 US Dollar"), 
        ('MSTR', "₿ Crypto"), ('MSFT', "💻 Tech"), ('NVDA', "📟 Semis"), 
        ('XLE', "🔥 Energy"), ('JPM', "🏦 Banks"), ('WMT', "🛡️ Defensive"), ('^VIX', "💀 Volatility")
    ]

    for ticker, row_title in anchors:
        st.markdown(f"### {row_title} vs Sector Averages")
        t_col, g_col = st.columns([2, 8])
        anchor_s = full_window[ticker]
        
        p_data = {
            "Period": ["1D", "1M", "3M", "6M"],
            "Perf %": [get_perf_and_trend(anchor_s, d)[0] for d in [2, 21, 63, 126]],
            "Trend": [get_perf_and_trend(anchor_s, d)[1] for d in [2, 21, 63, 126]]
        }
        
        t_col.dataframe(pd.DataFrame(p_data), hide_index=True)

        active_cats = [c for c in categories.keys() if ticker not in categories[c] and c != "A: Indicators"]
        gauge_cols = g_col.columns(len(active_cats) + 1)
        anchor_lbl = rename_map[ticker]
        section_audit = []

        for i, cat_name in enumerate(active_cats):
            cat_ticks = [rename_map[tk] for tk in categories[cat_name].keys()]
            indiv_vals = [corr_matrix.loc[anchor_lbl, lbl] for lbl in cat_ticks]
            avg_val = np.mean(indiv_vals)
            short_name = cat_name.split(': ')[1]
            emoji = category_emojis.get(short_name, "")
            draw_gauge(f"{emoji} {short_name}", avg_val, gauge_cols[i], f"g_{ticker}_{cat_name}_{day_offset}")
            section_audit.append({"Sector": short_name, "Values": ", ".join([f"{v:.2f}" for v in indiv_vals]), "Avg": f"{avg_val:.2f}"})

        draw_gauge("🟡 Gold", corr_matrix.loc[anchor_lbl, '🟡 GLD'], gauge_cols[-1], f"gold_{ticker}_{day_offset}")
        with st.expander(f"🧮 {row_title} Math Audit"): st.dataframe(pd.DataFrame(section_audit), hide_index=True)

    st.divider()
    y_chg = full_window['^TNX'].diff(5).iloc[-1]
    d_chg = full_window['DX-Y.NYB'].diff(5).iloc[-1]
    regime = "STAGFLATION (Risk-Off)" if y_chg > 0 and d_chg > 0 else "REFLATION (Sector Rotation)" if y_chg > 0 else "DEFLATION (Risk-Off)" if d_chg > 0 else "GOLDILOCKS (Risk-On)"
    
    c_reg, c_nar = st.columns([1, 2])
    c_reg.metric("Detected Regime", regime, delta=f"Y:{y_chg:.2f} | D:{d_chg:.2f}", delta_color="inverse")
    
    with c_nar:
        st.subheader("📜 Macro States & Dependencies")
        st.markdown(r"""
        * **Goldilocks (Risk-On):** Yields $\downarrow$ + Dollar $\downarrow \rightarrow$ Broad Expansion. Capital flows into Tech, Crypto, and high-beta risk assets.
        * **Stagflation (Risk-Off / Squeeze):** Yields $\uparrow$ + Dollar $\uparrow \rightarrow$ Liquidity Squeeze. Valuations compress across the board; cash is king.
        * **Reflation (Sector Rotation):** Yields $\uparrow$ + Dollar $\downarrow \rightarrow$ Growth under pressure. Capital rotates into Value, Energy, and Financials.
        * **Deflation (Risk-Off / Flight to Safety):** Yields $\downarrow$ + Dollar $\uparrow \rightarrow$ Growth shock. Capital hides in Treasury Bonds, Gold, and Defensive stalwarts.
        """)

    st.divider()
    st.subheader("📝 Analysis Summary & Thesis")
    st.markdown(r"""
    <div style="font-size:18px; line-height:1.6;">
    <ul>
        <li><b>The Liquidity Squeeze (TLT vs Tech):</b> High negative correlations indicate a valuation/discount-rate squeeze. If they move together, it is a broad liquidity event.</li>
        <li><b>The Reflation Trade (Energy vs Banks):</b> Look for high positive synergy; Energy driving yields typically bolsters Bank margins.</li>
        <li><b>The Wrecking Ball (DXY):</b> When the Dollar correlates positively with all assets, we are in a 'Dollar Milkshake' flight to safety.</li>
        <li><b>The Digital Hedge (VIX vs Crypto):</b> Inverse correlation to VIX signals 'Digital Gold'. Moving with VIX signals 'High-Beta Risk'.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("🏁 Footnote: Gauge Ingredients & Manual")
    st.markdown(r"""
    ### 🛠️ Sector Ingredients (The Recipe)
    Each gauge averages the **individual correlation coefficients** from the Heatmap for the following assets:
    * **📉 Indicators:** TIP, TLT, DXY, 10Y Yield, VIX
    * **₿ Digital Assets:** IBIT, MSTR
    * **💻 Tech:** MSFT, AAPL, GOOGL
    * **📟 Semis:** NVDA, AMD, AVGO
    * **🔥 Energy/Inf:** XLE, XOM, DBC
    * **🏦 Financials:** KBE, JPM, BRK-B
    * **🛡️ Defensive:** SCHD, WMT, PG
    
    ### 💡 How to Read the Dials
    **Example:** Looking at **📉 TLT (Bonds)** vs **💻 Tech**:
    1. **Left Swing (Blue) [-1.0 to -0.5]:** Bonds down, Tech down. Rates are the primary killer.
    2. **Right Swing (Red) [+0.5 to +1.0]:** Bonds up, Tech up. Market is pricing in a "Pivot" or growth bid.
    """)

# ==========================================
# TAB 2: MACRO BOND WATCH
# ==========================================
with tab2:
    st.header("Macro Bond Watch")
    st.markdown("Tracking the Treasury curve and spread dynamics to front-run regime shifts.")

    # Interactive Controls - Date Slider (Full Width Top)
    min_date = bond_df.index.min().to_pydatetime()
    max_date = bond_df.index.max().to_pydatetime()

    start_date, end_date = st.slider("Select Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="MMM YYYY")

    # Streamlit Tab Workaround (Horizontal Radio)
    mbw_view = st.radio(
        "Select Chart View",
        ["📊 Combined Dashboard", "📈 Yield Spectrum", "📉 Spread Regimes", "📉 MACD Momentum"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # 1. Create a placeholder container so the Charts render ABOVE the MACD toggle
    mbw_chart_container = st.container()

    # 2. Place the MACD toggle BELOW the charts placeholder
    macd_target = st.selectbox(
        "Select Spread for MACD Analysis",
        options=['10Y_2Y_Spread', '10Y_3M_Spread', '30Y_5Y_Spread'],
        format_func=lambda x: {'10Y_2Y_Spread': '2s10s (10Y - 2Y)', '10Y_3M_Spread': '3m10s (10Y - 3M)', '30Y_5Y_Spread': '5s30s (30Y - 5Y)'}[x]
    )

    macd_label = {'10Y_2Y_Spread': '2s10s (10Y - 2Y)', '10Y_3M_Spread': '3m10s (10Y - 3M)', '30Y_5Y_Spread': '5s30s (30Y - 5Y)'}[macd_target]

    # Filter & Calc MACD
    bond_filtered = bond_df[(bond_df.index >= start_date) & (bond_df.index <= end_date)].copy()
    bond_filtered['MACD_Line'] = bond_filtered[macd_target].ewm(span=12, adjust=False).mean() - bond_filtered[macd_target].ewm(span=26, adjust=False).mean()
    bond_filtered['Signal_Line'] = bond_filtered['MACD_Line'].ewm(span=9, adjust=False).mean()
    bond_filtered['MACD_Hist'] = bond_filtered['MACD_Line'] - bond_filtered['Signal_Line']
    bond_filtered['Hist_Color'] = ['#00FF00' if val > 0 else '#FF0000' for val in bond_filtered['MACD_Hist']]

    yield_cols = ['3-Month', '2-Year', '5-Year', '10-Year', '30-Year']
    bond_colors = ['#FF0000', '#FF4500', '#FFD700', '#00BFFF', '#0000FF']

    # --- Generate MBW Figures ---
    # 1. Yields Only
    fig_yields = go.Figure()
    for col, color in zip(yield_cols, bond_colors):
        fig_yields.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered[col], mode='lines', name=col, line=dict(color=color, width=2)))
    fig_yields.update_layout(title='Absolute Treasury Yields', template='plotly_dark', hovermode='closest', height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"))
    fig_yields.update_yaxes(title_text="Yield (%)")

    # 2. Spreads Only
    fig_spreads = go.Figure()
    fig_spreads.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['10Y_3M_Spread'], mode='lines', name='10Y - 3M Spread', line=dict(color='#00FF00', width=2)))
    fig_spreads.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['10Y_2Y_Spread'], mode='lines', name='10Y - 2Y Spread (2s10s)', line=dict(color='#FFFF00', width=2)))
    fig_spreads.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['30Y_5Y_Spread'], mode='lines', name='30Y - 5Y Spread (5s30s)', line=dict(color='#8A2BE2', width=2)))
    fig_spreads.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5)
    fig_spreads.update_layout(title='Spread Regimes: 2s10s, 3m10s & 5s30s', template='plotly_dark', hovermode='closest', height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"))
    fig_spreads.update_yaxes(title_text="Spread")

    # 3. MACD Only
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Bar(x=bond_filtered.index, y=bond_filtered['MACD_Hist'], marker_color=bond_filtered['Hist_Color'], name='MACD Histogram'))
    fig_macd.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['MACD_Line'], mode='lines', name='MACD Line', line=dict(color='#00BFFF', width=2)))
    fig_macd.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='#FFA500', width=2)))
    fig_macd.add_hline(y=0, line_color="white", line_width=1)
    fig_macd.update_layout(title=f'MACD: {macd_label}', template='plotly_dark', hovermode='closest', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"))
    fig_macd.update_yaxes(title_text="MACD")

    # 4. Combined
    fig_combined = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=('1. Absolute Treasury Yields', '2. Spread Regimes: 2s10s, 3m10s & 5s30s', f'MACD: {macd_label}'), row_heights=[0.5, 0.3, 0.2])
    for col, color in zip(yield_cols, bond_colors): fig_combined.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered[col], mode='lines', name=col, line=dict(color=color, width=2), legend="legend"), row=1, col=1)
    fig_combined.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['10Y_3M_Spread'], mode='lines', name='10Y - 3M Spread', line=dict(color='#00FF00', width=2), legend="legend2"), row=2, col=1)
    fig_combined.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['10Y_2Y_Spread'], mode='lines', name='10Y - 2Y Spread (2s10s)', line=dict(color='#FFFF00', width=2), legend="legend2"), row=2, col=1)
    fig_combined.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['30Y_5Y_Spread'], mode='lines', name='30Y - 5Y Spread (5s30s)', line=dict(color='#8A2BE2', width=2), legend="legend2"), row=2, col=1)
    fig_combined.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5, row=2, col=1)
    fig_combined.add_trace(go.Bar(x=bond_filtered.index, y=bond_filtered['MACD_Hist'], marker_color=bond_filtered['Hist_Color'], name='MACD Histogram', showlegend=False), row=3, col=1)
    fig_combined.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['MACD_Line'], mode='lines', name='MACD Line', line=dict(color='#00BFFF', width=1.5), showlegend=False), row=3, col=1)
    fig_combined.add_trace(go.Scatter(x=bond_filtered.index, y=bond_filtered['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='#FFA500', width=1.5), showlegend=False), row=3, col=1)
    fig_combined.add_hline(y=0, line_color="white", line_width=1, row=3, col=1)
    fig_combined.update_layout(template='plotly_dark', hovermode='closest', height=900, margin=dict(b=40, t=60), legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"), legend2=dict(orientation="h", yanchor="bottom", y=0.45, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"))
    fig_combined.update_yaxes(title_text="Yield (%)", row=1, col=1)
    fig_combined.update_yaxes(title_text="Spread", row=2, col=1)
    fig_combined.update_yaxes(title_text="MACD", row=3, col=1)

    # Render Chosen View INSIDE the container
    with mbw_chart_container:
        if "Combined" in mbw_view: st.plotly_chart(fig_combined, use_container_width=True)
        elif "Yield Spectrum" in mbw_view: st.plotly_chart(fig_yields, use_container_width=True)
        elif "Spread Regimes" in mbw_view: st.plotly_chart(fig_spreads, use_container_width=True)
        elif "MACD" in mbw_view: st.plotly_chart(fig_macd, use_container_width=True)

    # Regime Logic
    st.markdown("---")
    st.subheader("Live Market Regime")
    latest_bond = bond_filtered.iloc[-1]
    val_2s10s, val_3m10s, macd_hist = latest_bond['10Y_2Y_Spread'], latest_bond['10Y_3M_Spread'], latest_bond['MACD_Hist']

    if val_2s10s < 0 and val_3m10s < 0:
        if macd_hist > 0: regime, regime_color = "INVERTED BUT STEEPENING (Approaching De-Inversion)", "#FFA500"
        else: regime, regime_color = "DEEPLY INVERTED (Recession Warning)", "#FF0000"
    elif val_2s10s > 0 and val_3m10s > 0: regime, regime_color = "NORMAL / POSITIVE CURVE", "#00FF00"
    elif (val_2s10s > 0 and val_3m10s < 0) or (val_2s10s < 0 and val_3m10s > 0): regime, regime_color = "DE-INVERTING (The 'Event' Zone)", "#FFFF00"
    else: regime, regime_color = "TRANSITIONING", "#FFFFFF"

    st.markdown(f"**Latest Data Date:** {latest_bond.name.strftime('%B %d, %Y')}")
    st.markdown(f"**Current Status:** <span style='color:{regime_color}; font-size: 22px; font-weight: bold;'>{regime}</span>", unsafe_allow_html=True)

    cA, cB, cC = st.columns(3)
    cA.metric("2s10s Spread", f"{val_2s10s:.2f}%")
    cB.metric("3m10s Spread", f"{val_3m10s:.2f}%")
    cC.metric(f"MACD Momentum ({macd_target[:6]})", "Bullish (Steepening)" if macd_hist > 0 else "Bearish (Flattening/Inverting)")

    # Playbook
    st.markdown("---")
    st.subheader("The Gundlach Playbook: Reading the Regimes")
    st.markdown("""
    * **1. The Warning (Inversion):** When the spreads drop below the zero line, the bond market is pricing in structural economic sickness. It expects that current Federal Reserve policy will break something, eventually forcing emergency rate cuts. *A recession rarely starts while the curve is deeply inverted.*
    * **2. The Event (Bull Steepener):** The danger zone is the "De-Inversion." When the yellow (2s10s) or green (3m10s) lines violently cross back *above* zero, short-term yields are collapsing because the Fed is panicking. Historically, the exact moment the curve un-inverts is when job losses spike and the recession officially lands.
    * **3. The Vigilante Premium (5s30s):** The purple line is driven by the free market. The Fed controls the short end, but they can't control the 30-year. If the 5s30s explodes higher, bond vigilantes are demanding a massive premium to hold long-term debt because they fear runaway inflation or reckless government deficit spending.
    * **4. Using the MACD:** The MACD measures the momentum of your selected spread. When the blue MACD line crosses up through the orange Signal line, the curve is steepening (bullish for long bonds). When it crosses down, the flattening/inverting momentum is accelerating. 

    ### The Macro Reason: Current Reality vs. Future Guess
    The 3-month/10-year spread is actually considered the "purer" recession signal. Here is why:
    * **The 3-Month Yield is "Right Now":** The 13-week Treasury yield tracks the Federal Funds Rate almost perfectly. It tells you exactly how tight monetary policy is today.
    * **The 2-Year Yield is a "Guess":** The 2-year yield is highly sensitive to what the market anticipates the Fed will do over the next 24 months. If the market thinks the Fed is going to cut rates next year, the 2-year yield will drop long before the Fed actually makes a move.

    When you look at the 3m10s, you are comparing the actual, current restrictive cost of capital (3-month) against the market's long-term expectation for growth and inflation (10-year). In fact, the Federal Reserve Bank of New York has a famous, proprietary recession probability model, and they explicitly use the 3m10s spread, not the 2s10s, because historically it produces fewer false positives.
    """)

# ==========================================
# TAB 3: TERMINAL FEED
# ==========================================
with tab3:
    st.header("🗞️ Macro Terminal Feed")
    urls = {
        "CNBC Alerts": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=401&id=10000664",
        "MarketWatch": "http://feeds.marketwatch.com/marketwatch/marketpulse/",
        "Yahoo": "https://finance.yahoo.com/news/rssindex", 
        "FT": "https://www.ft.com/?format=rss"
    }
    
    all_news = []
    limit = datetime.now() - timedelta(days=14)
    for name, url in urls.items():
        if active_feeds[name]:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                try:
                    parsed_time = getattr(entry, 'published_parsed', getattr(entry, 'updated_parsed', None))
                    if parsed_time:
                        dt = datetime.fromtimestamp(time.mktime(parsed_time))
                        if dt > limit: 
                            all_news.append({"Date": dt, "Source": name, "Title": entry.title, "Link": entry.link})
                except Exception: 
                    continue
    
    all_news = sorted(all_news, key=lambda x: x['Date'], reverse=True)
    
    start = st.session_state.pg * 50
    for item in all_news[start : start + 50]:
        st.write(f"**{item['Date'].strftime('%m/%d %H:%M')}** | `{item['Source']}` : [{item['Title']}]({item['Link']})")
    
    c1, c2 = st.columns(2)
    if st.session_state.pg > 0:
        if c1.button("⬅️ Previous"): 
            st.session_state.pg -= 1
            st.rerun()
    if len(all_news) > start + 50:
        if c2.button("Next ➡️"): 
            st.session_state.pg += 1
            st.rerun()