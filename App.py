import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import feedparser
import time
from datetime import datetime, timedelta

# --- 1. PAGE CONFIG & STATE ---
st.set_page_config(page_title="Macro Snapshot v32.3", layout="wide")

# Cloud-Safe Session State Init
if "pg" not in st.session_state: 
    st.session_state.pg = 0

# --- 2. ASSET BUCKETS ---
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

# --- 3. SIDEBAR (Restored Full Legend) ---
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
day_offset = st.sidebar.slider("Snapshot Day", 0, 252, 0)
comp_lookback = st.sidebar.slider("Comparison Start", 2, 252, 60)
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

all_tickers = list(set([t for c in categories.values() for t in c.keys()] + ['GLD']))
raw_data = load_market_data(all_tickers)
rename_map = {t: l for cat in categories.values() for t, l in cat.items()}
rename_map['GLD'] = '🟡 GLD'

if raw_data.empty:
    st.error("📡 Yahoo Finance API rate limit hit or unavailable. Please try again in a few minutes.")
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

# --- 6. PROCESSING ---
target_idx = (len(raw_data) - 1) - day_offset
snapshot_date = raw_data.index[target_idx]
full_window = raw_data.iloc[:target_idx + 1]
corr_window = full_window.iloc[-window_size:].rename(columns=rename_map)
ordered_labels = [rename_map[t] for cat in categories.values() for t in cat.keys()] + ['🟡 GLD']
corr_matrix = corr_window[ordered_labels].corr()

# --- 7. TABS ---
tab1, tab2 = st.tabs(["📊 Market Analysis", "📰 Terminal Feed"])

with tab1:
    # 1. Charts with Grouped Toggles
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

    # 2. Heatmap
    st.subheader("2. Market DNA Heatmap (Indicators Centered)")
    fig_corr = px.imshow(corr_matrix.mask(np.triu(np.ones_like(corr_matrix, dtype=bool))), text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1], template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    # 3. Command Center
    st.divider()
    st.subheader("3. Macro Command Center: Distilled Dials")
    anchors = [
        ('TLT', "📉 TLT (Bonds)"), ('^TNX', "🏛️ 10Y Yield"), ('DX-Y.NYB', "💵 DXY (Dollar)"), 
        ('MSTR', "₿ Crypto (MSTR)"), ('MSFT', "💻 Tech (MSFT)"), ('NVDA', "📟 Semis (NVDA)"), 
        ('XLE', "🔥 Energy (XLE)"), ('JPM', "🏦 Banks (JPM)"), ('WMT', "🛡️ Defensive (WMT)"), ('^VIX', "💀 VIX (Fear)")
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
        
        # CHANGED: Swapped st.table for st.dataframe with hide_index=True
        t_col.dataframe(pd.DataFrame(p_data), hide_index=True)

        active_cats = [c for c in categories.keys() if ticker not in categories[c]]
        gauge_cols = g_col.columns(len(active_cats) + 1)
        anchor_lbl = rename_map[ticker]
        section_audit = []

        for i, cat_name in enumerate(active_cats):
            cat_ticks = [rename_map[tk] for tk in categories[cat_name].keys()]
            indiv_vals = [corr_matrix.loc[anchor_lbl, lbl] for lbl in cat_ticks]
            avg_val = np.mean(indiv_vals)
            short_name = cat_name.split(': ')[1]
            draw_gauge(short_name, avg_val, gauge_cols[i], f"g_{ticker}_{cat_name}_{day_offset}")
            section_audit.append({"Sector": short_name, "Values": ", ".join([f"{v:.2f}" for v in indiv_vals]), "Avg": f"{avg_val:.2f}"})

        draw_gauge("🟡 Gold", corr_matrix.loc[anchor_lbl, '🟡 GLD'], gauge_cols[-1], f"gold_{ticker}_{day_offset}")
        with st.expander(f"🧮 {row_title} Math Audit"): st.dataframe(pd.DataFrame(section_audit), hide_index=True)

    # 4. REGIME & EXPANDED NARRATIVES
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

    # 5. RESTORED FOOTER (Analysis & Recipe)
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

with tab2:
    st.header("🗞️ Macro Terminal Feed")
    urls = {"CNBC Alerts": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=401&id=10000664",
            "MarketWatch": "http://feeds.marketwatch.com/marketwatch/marketpulse/",
            "Yahoo": "https://finance.yahoo.com/news/rssindex", "FT": "https://www.ft.com/?format=rss"}
    
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