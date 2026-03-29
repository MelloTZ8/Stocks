import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import feedparser
import time

# --- 1. PAGE CONFIG & ASSET BUCKETS ---
st.set_page_config(page_title="Macro Snapshot v31.0", layout="wide")

categories = {
    "A: Indicators": {'TIP': '🟦 TIP', 'TLT': '🟦 TLT', 'DX-Y.NYB': '🟦 DXY', '^TNX': '🟦 10Y', '^VIX': '💀 VIX'},
    "B: Digital Assets": {'IBIT': '₿ IBIT', 'MSTR': '₿ MSTR'},
    "C: Tech": {'MSFT': '💻 MSFT', 'AAPL': '💻 AAPL', 'GOOGL': '💻 GOOGL'},
    "D: Semis": {'NVDA': '📟 NVDA', 'AMD': '📟 AMD', 'AVGO': '📟 AVGO'},
    "E: Energy/Inf": {'XLE': '🔥 XLE', 'XOM': '🔥 XOM', 'DBC': '🔥 DBC'},
    "F: Financials": {'KBE': '🏦 KBE', 'JPM': '🏦 JPM', 'BRK-B': '🏦 BRK'},
    "G: Defensive": {'SCHD': '🛡️ SCHD', 'WMT': '🛡️ WMT', 'PG': '🛡️ PG'},
}

category_emojis = {
    "Indicators": "📉", "Digital Assets": "₿", "Tech": "💻", 
    "Semis": "📟", "Energy/Inf": "🔥", "Financials": "🏦", "Defensive": "🛡️"
}

color_map = {
    '🟦 TIP': '#0000FF', '🟦 TLT': '#4169E1', '🟦 DXY': '#00BFFF', '🟦 10Y': '#191970', '💀 VIX': '#708090',
    '₿ IBIT': '#FF8C00', '₿ MSTR': '#E67E22', '💻 MSFT': '#00FFFF', '💻 AAPL': '#20B2AA', '💻 GOOGL': '#48D1CC',
    '📟 NVDA': '#FFD700', '📟 AMD': '#DAA520', '📟 AVGO': '#BDB76B', '🔥 XLE': '#FF0000', '🔥 XOM': '#DC143C', 
    '🔥 DBC': '#8B0000', '🟡 GLD': '#FFD700', '🏦 KBE': '#800080', '🏦 JPM': '#9370DB', '🏦 BRK': '#4B0082',
    '🛡️ SCHD': '#008000', '🛡️ WMT': '#32CD32', '🛡️ PG': '#228B22',
}

# --- 2. SIDEBAR: RESTORED DETAILED LEGEND ---
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
window_size = st.sidebar.slider("Correlation Sensitivity", 5, 60, 15)

rename_map = {t: l for cat in categories.values() for t, l in cat.items()}
rename_map['GLD'] = '🟡 GLD'
selected_labels = st.sidebar.multiselect("Select Assets:", options=list(rename_map.values()), default=list(rename_map.values()))

# --- 3. DATA LOADING (Cloud-Hardened) ---
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

if raw_data.empty:
    st.error("📡 Connection to Yahoo Finance failed. Please refresh.")
    st.stop()

# --- 4. DATA PROCESSING ---
target_idx = (len(raw_data) - 1) - day_offset
snapshot_date = raw_data.index[target_idx]
full_window = raw_data.iloc[:target_idx + 1]
perf_window = full_window.iloc[max(0, len(full_window)-comp_lookback):].rename(columns=rename_map)
corr_window = full_window.iloc[-window_size:].rename(columns=rename_map)
corr_matrix = corr_window.corr()

# --- UTILITY ---
def get_perf_and_trend(series, days):
    try:
        start_val = series.iloc[-days]; end_val = series.iloc[-1]
        pct = ((end_val - start_val) / start_val) * 100
        return f"{pct:.2f}%", "🟢" if pct > 0 else "🔴"
    except: return "0.00%", "⚪"

# --- 5. TABS ---
tab1, tab2 = st.tabs(["📊 Market Analysis", "📰 Live Economics & News"])

with tab1:
    # 1. Charts (Restored Hover & Color Families)
    st.subheader(f"1. Performance Snapshot: {snapshot_date.strftime('%Y-%m-%d')}")
    norm_data = (perf_window[selected_labels] / perf_window[selected_labels].iloc[0]) * 100
    fig_line = px.line(norm_data, log_y=True, template="plotly_dark", color_discrete_map=color_map)
    fig_line.update_layout(height=500, hovermode='closest')
    st.plotly_chart(fig_line, use_container_width=True, theme=None)

    st.subheader("2. Market DNA Heatmap")
    fig_corr = px.imshow(corr_matrix.mask(np.triu(np.ones_like(corr_matrix, dtype=bool))), 
                         text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1], template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True, theme=None)

    st.divider()
    st.subheader("3. Macro Command Center: Distilled Dials")

    def get_corr_color(val):
        if val > 0: return f"rgb(255, {int(255*(1-val))}, {int(255*(1-val))})"
        return f"rgb({int(255*(1+val))}, {int(255*(1+val))}, 255)"

    def draw_gauge(title, val, col, key):
        quartile_ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            title={'text': title, 'font': {'size': 14, 'color': 'white'}},
            number={'font': {'size': 24, 'color': 'white'}},
            gauge={
                'axis': {'range': [-1, 1], 'tickvals': quartile_ticks, 'tickfont': {'size': 9}, 'tickcolor': 'white'},
                'bar': {'color': get_corr_color(val)},
                'bgcolor': "#262626", 'borderwidth': 1, 'bordercolor': "white"
            }))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=20), paper_bgcolor="#0E1117", font={'color': "white"})
        col.plotly_chart(fig, use_container_width=True, key=key, theme=None)

    # RESTORED: Gauge Segments and Row Titles
    anchors = [
        ('TLT', "📉 TLT (Bonds)"), ('^TNX', "🏛️ 10Y Yield"), ('DX-Y.NYB', "💵 DXY (Dollar)"), 
        ('MSTR', "₿ Crypto (MSTR)"), ('MSFT', "💻 Tech (MSFT)"), ('NVDA', "📟 Semis (NVDA)"), 
        ('XLE', "🔥 Energy (XLE)"), ('JPM', "🏦 Banks (JPM)"), ('WMT', "🛡️ Defensive (WMT)"), ('^VIX', "💀 VIX (Fear)")
    ]

    for ticker, row_title in anchors:
        st.markdown(f"### {row_title} vs Sector Averages")
        t_col, g_col = st.columns([2, 8])
        
        # Perf Table
        anchor_s = full_window[ticker]
        p_df = pd.DataFrame({
            "Period": ["1D", "1M", "3M", "6M"],
            "Perf %": [get_perf_and_trend(anchor_s, d)[0] for d in [2,21,63,126]],
            "Trend": [get_perf_and_trend(anchor_s, d)[1] for d in [2,21,63,126]]
        })
        t_col.table(p_df)

        # Distilled Dials Logic
        active_cats = [c for c in categories.keys() if c != "A: Indicators" and ticker not in categories[c]]
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
            section_audit.append({"Sector": short_name, "Heatmap Values": ", ".join([f"{v:.2f}" for v in indiv_vals]), "Distilled Avg": f"{avg_val:.2f}"})

        # Standalone Gold
        gold_corr = corr_matrix.loc[anchor_lbl, '🟡 GLD']
        draw_gauge("🟡 Gold", gold_corr, gauge_cols[-1], f"gold_{ticker}_{day_offset}")

        with st.expander(f"🧮 {row_title} Math Audit"):
            st.dataframe(pd.DataFrame(section_audit), hide_index=True)

    st.divider()
    # RESTORED: Analysis Summary Thesis
    st.subheader("📝 Analysis Summary & Thesis")
    st.markdown(f"""
    <div style="font-size:24px; line-height:1.6;">
    <ul>
        <li><b>The Liquidity Squeeze (TLT vs Tech):</b> High negative correlations indicate a valuation/discount-rate squeeze. If they move together, it is a broad liquidity event.</li>
        <li><b>The Reflation Trade (Energy vs Banks):</b> Look for high positive synergy; Energy driving yields typically bolsters Bank margins.</li>
        <li><b>The Wrecking Ball (DXY):</b> When the Dollar correlates positively with all assets, we are in a 'Dollar Milkshake' flight to safety.</li>
        <li><b>The Digital Hedge (VIX vs Crypto):</b> Inverse correlation to VIX signals 'Digital Gold'. Moving with VIX signals 'High-Beta Risk'.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    # RESTORED: Footnote Recipes & Example
    st.subheader("🏁 Footnote: Gauge Ingredients & Manual")
    st.markdown("""
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
    st.header("🗞️ Intelligence Feed")
    keyword = st.text_input("Highlight Keywords:", "Fed")
    feed_sources = {
        "Bloomberg Wealth": "https://feeds.bloomberg.com/wealth/news.rss",
        "Bloomberg Tech": "https://feeds.bloomberg.com/technology/news.rss",
        "CNBC Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
        "WSJ Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"
    }
    news_items = []
    for source, url in feed_sources.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            news_items.append({"Source": source, "Date": getattr(entry, 'published', "Recent"), "Headline": entry.title, "Link": entry.link})
    for row in news_items[:30]:
        display_text = f"**[{row['Headline']}]({row['Link']})**"
        if keyword.lower() in row['Headline'].lower():
            st.error(f"🚨 {row['Source']} | {row['Date']} | {display_text}")
        else:
            st.write(f"`{row['Source']}` | {row['Date']} | {display_text}")
        st.divider()