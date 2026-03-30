import streamlit as st
import pandas_datareader.data as web
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import datetime

# 1. Streamlit Page Config
st.set_page_config(page_title="Macro Bond Watch", layout="wide")

st.title("Macro Bond Watch")
st.markdown("Tracking the Treasury curve and spread dynamics to front-run regime shifts.")

# 2. Fetch Data (Lookback extended to 1994 via FRED)
@st.cache_data
def load_data():
    end = datetime.date.today()
    start = datetime.date(1994, 1, 1) # Hardcoded start date to capture 30+ years of macro history
    
    tickers = ['DGS3MO', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
    df = web.DataReader(tickers, 'fred', start, end)
    df.columns = ['3-Month', '2-Year', '5-Year', '10-Year', '30-Year']
    
    df.dropna(inplace=True)
    
    # Calculate Spreads
    df['10Y_3M_Spread'] = df['10-Year'] - df['3-Month']
    df['10Y_2Y_Spread'] = df['10-Year'] - df['2-Year']
    df['30Y_5Y_Spread'] = df['30-Year'] - df['5-Year']
    
    return df

df = load_data()

# 3. Interactive Controls - Date Slider (Top)
min_date = df.index.min().to_pydatetime()
max_date = df.index.max().to_pydatetime()

start_date, end_date = st.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="MMM YYYY"
)

# 4. Interactive Controls - MACD Toggle
macd_target = st.selectbox(
    "Select Spread for MACD Analysis",
    options=['10Y_2Y_Spread', '10Y_3M_Spread', '30Y_5Y_Spread'],
    format_func=lambda x: {
        '10Y_2Y_Spread': '2s10s (10Y - 2Y)', 
        '10Y_3M_Spread': '3m10s (10Y - 3M)', 
        '30Y_5Y_Spread': '5s30s (30Y - 5Y)'
    }[x]
)

macd_label = {
    '10Y_2Y_Spread': '2s10s (10Y - 2Y)', 
    '10Y_3M_Spread': '3m10s (10Y - 3M)', 
    '30Y_5Y_Spread': '5s30s (30Y - 5Y)'
}[macd_target]

# Filter dataframe based on date slider
df_filtered = df[(df.index >= start_date) & (df.index <= end_date)].copy()

# 5. Standard 12/26/9 MACD Calculation
df_filtered['MACD_Line'] = df_filtered[macd_target].ewm(span=12, adjust=False).mean() - df_filtered[macd_target].ewm(span=26, adjust=False).mean()
df_filtered['Signal_Line'] = df_filtered['MACD_Line'].ewm(span=9, adjust=False).mean()
df_filtered['MACD_Hist'] = df_filtered['MACD_Line'] - df_filtered['Signal_Line']

# NEON Colors for the MACD Histogram
df_filtered['Hist_Color'] = ['#00FF00' if val > 0 else '#FF0000' for val in df_filtered['MACD_Hist']]

# ==========================================
# CHART BUILDING SECTION
# ==========================================
yield_cols = ['3-Month', '2-Year', '5-Year', '10-Year', '30-Year']
colors = ['#FF0000', '#FF4500', '#FFD700', '#00BFFF', '#0000FF']

# --- A. Build COMBINED View (Original 3-Pane) ---
fig_combined = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=('1. Absolute Treasury Yields', '2. Spread Regimes: 2s10s, 3m10s & 5s30s', f'MACD: {macd_label}'),
    row_heights=[0.5, 0.3, 0.2]
)

for col, color in zip(yield_cols, colors):
    fig_combined.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[col], mode='lines', name=col, line=dict(color=color, width=2), legend="legend"), row=1, col=1)

fig_combined.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['10Y_3M_Spread'], mode='lines', name='10Y - 3M Spread', line=dict(color='#00FF00', width=2), legend="legend2"), row=2, col=1)
fig_combined.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['10Y_2Y_Spread'], mode='lines', name='10Y - 2Y Spread (2s10s)', line=dict(color='#FFFF00', width=2), legend="legend2"), row=2, col=1)
fig_combined.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['30Y_5Y_Spread'], mode='lines', name='30Y - 5Y Spread (5s30s)', line=dict(color='#8A2BE2', width=2), legend="legend2"), row=2, col=1)
fig_combined.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5, row=2, col=1)

fig_combined.add_trace(go.Bar(x=df_filtered.index, y=df_filtered['MACD_Hist'], marker_color=df_filtered['Hist_Color'], name='MACD Histogram', showlegend=False), row=3, col=1)
fig_combined.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MACD_Line'], mode='lines', name='MACD Line', line=dict(color='#00BFFF', width=1.5), showlegend=False), row=3, col=1)
fig_combined.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='#FFA500', width=1.5), showlegend=False), row=3, col=1)
fig_combined.add_hline(y=0, line_color="white", line_width=1, row=3, col=1)

fig_combined.update_layout(
    template='plotly_dark', hovermode='closest', height=900, margin=dict(b=40, t=60),
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"),
    legend2=dict(orientation="h", yanchor="bottom", y=0.45, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)")
)
fig_combined.update_yaxes(title_text="Yield (%)", row=1, col=1)
fig_combined.update_yaxes(title_text="Spread", row=2, col=1)
fig_combined.update_yaxes(title_text="MACD", row=3, col=1)

# --- B. Build INDIVIDUAL Views (For Tabs) ---
# 1. Yields Only
fig_yields = go.Figure()
for col, color in zip(yield_cols, colors):
    fig_yields.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered[col], mode='lines', name=col, line=dict(color=color, width=2)))
fig_yields.update_layout(title='Absolute Treasury Yields', template='plotly_dark', hovermode='closest', height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"))
fig_yields.update_yaxes(title_text="Yield (%)")

# 2. Spreads Only
fig_spreads = go.Figure()
fig_spreads.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['10Y_3M_Spread'], mode='lines', name='10Y - 3M Spread', line=dict(color='#00FF00', width=2)))
fig_spreads.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['10Y_2Y_Spread'], mode='lines', name='10Y - 2Y Spread (2s10s)', line=dict(color='#FFFF00', width=2)))
fig_spreads.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['30Y_5Y_Spread'], mode='lines', name='30Y - 5Y Spread (5s30s)', line=dict(color='#8A2BE2', width=2)))
fig_spreads.add_hline(y=0, line_dash="dash", line_color="white", line_width=1.5)
fig_spreads.update_layout(title='Spread Regimes: 2s10s, 3m10s & 5s30s', template='plotly_dark', hovermode='closest', height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"))
fig_spreads.update_yaxes(title_text="Spread")

# 3. MACD Only
fig_macd = go.Figure()
fig_macd.add_trace(go.Bar(x=df_filtered.index, y=df_filtered['MACD_Hist'], marker_color=df_filtered['Hist_Color'], name='MACD Histogram'))
fig_macd.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['MACD_Line'], mode='lines', name='MACD Line', line=dict(color='#00BFFF', width=2)))
fig_macd.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='#FFA500', width=2)))
fig_macd.add_hline(y=0, line_color="white", line_width=1)
fig_macd.update_layout(title=f'MACD: {macd_label}', template='plotly_dark', hovermode='closest', height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)"))
fig_macd.update_yaxes(title_text="MACD")

# ==========================================
# STREAMLIT UI RENDER (TABS)
# ==========================================
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Combined Dashboard", "📈 1. Yield Spectrum", "📉 2. Spread Regimes", "📉 3. MACD Momentum"])

with tab1:
    st.plotly_chart(fig_combined, use_container_width=True)
with tab2:
    st.plotly_chart(fig_yields, use_container_width=True)
with tab3:
    st.plotly_chart(fig_spreads, use_container_width=True)
with tab4:
    st.plotly_chart(fig_macd, use_container_width=True)

# ==========================================
# CURRENT REGIME DETECTOR
# ==========================================
st.markdown("---")
st.subheader("Live Market Regime")

latest_data = df_filtered.iloc[-1]
val_2s10s = latest_data['10Y_2Y_Spread']
val_3m10s = latest_data['10Y_3M_Spread']
macd_hist = latest_data['MACD_Hist']

if val_2s10s < 0 and val_3m10s < 0:
    if macd_hist > 0:
        regime = "INVERTED BUT STEEPENING (Approaching De-Inversion)"
        regime_color = "#FFA500" 
    else:
        regime = "DEEPLY INVERTED (Recession Warning)"
        regime_color = "#FF0000" 
elif val_2s10s > 0 and val_3m10s > 0:
    regime = "NORMAL / POSITIVE CURVE"
    regime_color = "#00FF00" 
elif (val_2s10s > 0 and val_3m10s < 0) or (val_2s10s < 0 and val_3m10s > 0):
    regime = "DE-INVERTING (The 'Event' Zone)"
    regime_color = "#FFFF00" 
else:
    regime = "TRANSITIONING"
    regime_color = "#FFFFFF"

st.markdown(f"**Latest Data Date:** {latest_data.name.strftime('%B %d, %Y')}")
st.markdown(f"**Current Status:** <span style='color:{regime_color}; font-size: 22px; font-weight: bold;'>{regime}</span>", unsafe_allow_html=True)

colA, colB, colC = st.columns(3)
colA.metric("2s10s Spread", f"{val_2s10s:.2f}%")
colB.metric("3m10s Spread", f"{val_3m10s:.2f}%")
momentum_state = "Bullish (Steepening)" if macd_hist > 0 else "Bearish (Flattening/Inverting)"
colC.metric(f"MACD Momentum ({macd_target[:6]})", momentum_state)

# ==========================================
# GUNDLACH SUMMARY SECTION
# ==========================================
st.markdown("---")
st.subheader("The Gundlach Playbook: Reading the Regimes")
st.markdown("""
* **1. The Warning (Inversion):** When the spreads in Chart 2 drop below the zero line, the bond market is pricing in structural economic sickness. It expects that current Federal Reserve policy will break something, eventually forcing emergency rate cuts. *A recession rarely starts while the curve is deeply inverted.*
* **2. The Event (Bull Steepener):** The danger zone is the "De-Inversion." When the yellow (2s10s) or green (3m10s) lines violently cross back *above* zero, short-term yields are collapsing because the Fed is panicking. Historically, the exact moment the curve un-inverts is when job losses spike and the recession officially lands.
* **3. The Vigilante Premium (5s30s):** The purple line is driven by the free market. The Fed controls the short end, but they can't control the 30-year. If the 5s30s explodes higher, bond vigilantes are demanding a massive premium to hold long-term debt because they fear runaway inflation or reckless government deficit spending.
* **4. Using the MACD:** The MACD measures the momentum of your selected spread. When the blue MACD line crosses up through the orange Signal line, the curve is steepening (bullish for long bonds). When it crosses down, the flattening/inverting momentum is accelerating. 

### The Macro Reason: Current Reality vs. Future Guess
The 3-month/10-year spread is actually considered the "purer" recession signal. Here is why:

* **The 3-Month Yield is "Right Now":** The 13-week Treasury yield tracks the Federal Funds Rate almost perfectly. It tells you exactly how tight monetary policy is today.
* **The 2-Year Yield is a "Guess":** The 2-year yield is highly sensitive to what the market anticipates the Fed will do over the next 24 months. If the market thinks the Fed is going to cut rates next year, the 2-year yield will drop long before the Fed actually makes a move.

When you look at the 3m10s, you are comparing the actual, current restrictive cost of capital (3-month) against the market's long-term expectation for growth and inflation (10-year). In fact, the Federal Reserve Bank of New York has a famous, proprietary recession probability model, and they explicitly use the 3m10s spread, not the 2s10s, because historically it produces fewer false positives.
""")