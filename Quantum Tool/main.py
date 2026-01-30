import streamlit as st
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from qiskit_machine_learning.algorithms import VQR
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.primitives import StatevectorEstimator as Estimator 
from qiskit_algorithms.optimizers import COBYLA
from streamlit_autorefresh import st_autorefresh
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import os

# --- 1. CORE CONFIG & HIGH-CONTRAST CSS ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()
st.set_page_config(page_title="ME Quantitative Analyst", page_icon="🏹", layout="wide")

if 'access_granted' not in st.session_state:
    st.session_state.access_granted = False

# CSS: News Ticker, High-Contrast Inputs, and restored Layout
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Roboto+Mono:wght@500&display=swap');
    
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.95)), 
                    url("https://images.unsplash.com/photo-1512453979798-5ea266f8880c?q=80&w=2070");
        background-size: cover; background-attachment: fixed;
    }

    /* --- GLOBAL NEWS TICKER --- */
    .ticker-wrap {
        position: fixed; top: 0; left: 0; width: 100%; height: 35px;
        background: rgba(0, 232, 255, 0.15); backdrop-filter: blur(10px);
        overflow: hidden; z-index: 9999; border-bottom: 1px solid #00E8FF;
    }
    .ticker {
        display: inline-block; height: 35px; line-height: 35px;
        white-space: nowrap; padding-right: 100%;
        animation: ticker 60s linear infinite;
        font-family: 'Roboto Mono', monospace; color: #00E8FF; font-size: 14px;
    }
    @keyframes ticker {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-100%, 0, 0); }
    }

    /* --- INPUT VISIBILITY (PASSWORD & TICKERS) --- */
    input {
        color: #FFFFFF !important;
        background-color: #0a0e14 !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    div[data-baseweb="input"] {
        background-color: #0a0e14 !important;
        border: 2px solid #00E8FF !important;
    }
    ::placeholder { color: #888888 !important; opacity: 1; }

    /* Containers */
    div[data-testid="metric-container"], .stDataFrame, .stPlotlyChart {
        background: rgba(10, 15, 25, 0.98) !important;
        border: 1px solid #4a5568 !important; border-radius: 12px; padding: 15px;
    }

    h1, h2, h3, p, label, span, .stMarkdown { 
        color: #FFFFFF !important; font-family: 'Inter', sans-serif; 
    }
    
    .stButton>button { 
        background: #00E8FF !important; color: black !important; 
        font-weight: 800; border-radius: 8px; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINES ---

def get_ticker_html():
    """Generates scrolling text for the top ticker."""
    try:
        trending = yf.Ticker("SPY").news[:5]
        headlines = "  •  ".join([n['title'] for n in trending])
        return f'<div class="ticker-wrap"><div class="ticker">{headlines}</div></div>'
    except:
        return '<div class="ticker-wrap"><div class="ticker">ESTABLISHING QUANTUM DATA LINK... MARKET FEED OFFLINE</div></div>'

def get_sentiment(ticker):
    try:
        data = yf.Ticker(ticker)
        news = data.news
        if not news: return 0
        scores = [sia.polarity_scores(a.get('title', ''))['compound'] for a in news[:5]]
        return np.mean(scores) if scores else 0
    except: return 0

@st.cache_data(ttl=600)
def quantum_engine(symbol, lookback):
    hist = yf.Ticker(symbol).history(period="1y")
    if hist.empty: return None
    df = hist[['Close']].copy()
    df['Vol'] = df['Close'].pct_change().rolling(5).std()
    df['Mom'] = df['Close'].pct_change(5)
    df = df.dropna()
    X_scaler, y_scaler = MinMaxScaler(feature_range=(0, np.pi)), MinMaxScaler(feature_range=(-1, 1))
    X = X_scaler.fit_transform(df[['Vol', 'Mom']])
    y = y_scaler.fit_transform(df['Close'].values.reshape(-1, 1)).flatten()
    vqr = VQR(feature_map=zz_feature_map(2), ansatz=real_amplitudes(2), optimizer=COBYLA(maxiter=30), estimator=Estimator())
    vqr.fit(X[-lookback:], y[-lookback:])
    preds = y_scaler.inverse_transform(vqr.predict(X).reshape(-1, 1))
    return df, preds, float(1 - np.abs(y[-1] - vqr.predict(X[-1:])[0]))

# --- 3. THE WALL ---
if not st.session_state.access_granted:
    st.markdown(get_ticker_html(), unsafe_allow_html=True)
    st.title("🏹 ME Quantitative Analyst")
    st.subheader("High-Contrast Quantum Terminal")
    col1, col2 = st.columns(2)
    with col2:
        token = st.text_input("Enter Private Token", type="password", placeholder="Token Required")
        if st.button("Unlock Terminal"):
            if token == "QUANT2026":
                st.session_state.access_granted = True
                st.rerun()
            else: st.error("Access Denied.")
    with col1:
        st.write("Authorized Analyst Portal. Synchronizing wave-functions for Canary Wharf Hub.")

# --- 4. THE TERMINAL ---
else:
    st.markdown(get_ticker_html(), unsafe_allow_html=True)
    st.logo("https://img.icons8.com/nolan/128/quantum-computing.png", size="large")
    st_autorefresh(interval=300000, key="global_sync")
    
    SECTORS = {
        "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "Semis": ["AMD", "TSM", "AVGO", "ARM", "INTC"],
        "Finance": ["GS", "JPM", "MS", "V", "MA"]
    }

    # Restored Journal and Admin Tabs
    tab_sweep, tab_core, tab_journal, tab_admin = st.tabs(["🔥 Global Sweep", "⚛️ Quantum Core", "📓 Trade Journal", "👤 Admin"])

    with tab_sweep:
        st.subheader("🚀 Global Narrative & Quantum Sweep")
        selected = st.multiselect("Active Asset Hubs", list(SECTORS.keys()), default=["Magnificent 7"])
        if st.button("🚀 Execute Market Scan"):
            scan_list = []
            for s in selected: scan_list.extend(SECTORS[s])
            results = []
            prog = st.progress(0, text="Analyzing Market Gravity...")
            for i, t in enumerate(scan_list):
                try:
                    q_res = quantum_engine(t, 40)
                    if q_res:
                        df, p, conf = q_res
                        price, target = df['Close'].iloc[-1], p[-1][0]
                        gap = ((target-price)/price)*100
                        results.append({"Ticker": t, "Gap %": round(gap, 2), "Status": "Bullish" if gap > 0 else "Bearish"})
                except: continue
                prog.progress((i+1)/len(scan_list))
            st.dataframe(pd.DataFrame(results), use_container_width=True)

    with tab_core:
        with st.sidebar:
            st.header("🎛️ Terminal Config")
            ticker_input = st.text_input("Asset Ticker", value="NVDA").upper()
            sync_trigger = st.button("🔄 Synchronize Asset")
            look = st.slider("Lookback Window", 20, 150, 60)
            if st.button("🔒 Secure Session"):
                st.session_state.access_granted = False
                st.rerun()
        
        if sync_trigger or ticker_input:
            try:
                df, p, c = quantum_engine(ticker_input, look)
                sent = get_sentiment(ticker_input)
                c1, c2, c3 = st.columns(3)
                c1.metric("Live Price", f"${df['Close'].iloc[-1]:.2f}")
                c2.metric("Quantum Gap", f"{((p[-1][0]-df['Close'].iloc[-1])/df['Close'].iloc[-1])*100:.1f}%")
                c3.metric("State Confidence", f"{c*100:.1f}%")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Market Price', line=dict(color='#00d4ff', width=2)))
                fig.add_trace(go.Scatter(x=df.index, y=p.flatten(), name='Quantum Target', line=dict(color='#ffaa00', dash='dot')))
                fig.update_layout(template='plotly_dark', height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#FFFFFF"))
                st.plotly_chart(fig, use_container_width=True)
            except: st.error("Syncing...")

    with tab_journal:
        st.subheader("📓 Trade Journal Ledger")
        JOURNAL_FILE = "trading_journal.csv"
        if not os.path.exists(JOURNAL_FILE):
            pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Price', 'Notes']).to_csv(JOURNAL_FILE, index=False)
        j_df = pd.read_csv(JOURNAL_FILE)
        edited_j = st.data_editor(j_df, use_container_width=True, num_rows="dynamic")
        if st.button("Save Journal"):
            edited_j.to_csv(JOURNAL_FILE, index=False)
            st.toast("Journal Saved.")

    with tab_admin:
        if os.path.exists("waitlist_database.csv"):
            st.dataframe(pd.read_csv("waitlist_database.csv"), use_container_width=True)