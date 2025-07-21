import streamlit as st
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

st.set_page_config(page_title="Prediksi Harga Crypto dari CoinGecko", layout="wide")
st.title("📉 Prediksi Harga Crypto dari CoinGecko")

@st.cache_data
def get_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "1",
        "interval": "hourly"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        prices = data["prices"]
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["price"].shift(1)
        df["close"] = df["price"]
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)
        df = df.dropna().reset_index(drop=True)
        df["target"] = df["close"].shift(-1)
        return df.dropna()
    except:
        return None

df = get_data()

if df is None or len(df) < 10:
    st.error("Gagal mengambil data dari CoinGecko atau data terlalu sedikit.")
    st.stop()

X = df[["open", "high", "low"]]
y = df["target"]

if len(X) > 5:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.success(f"Model Akurasi: {acc:.2f}")
else:
    st.warning("Data terlalu sedikit untuk pelatihan model.")
    st.stop()

# Prediksi harga
last_data = X.tail(1)
prediction = model.predict(last_data)
st.success(f"🎯 Prediksi Harga Selanjutnya: ${prediction[0]:,.2f}")

# Tampilan DataFrame
st.subheader("📊 Data Terbaru")
st.dataframe(df.tail(), use_container_width=True)

# Grafik Candlestick
fig = go.Figure(data=[
    go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='BTC/USD'
    )
])
fig.update_layout(title="Grafik Harga Bitcoin (Candlestick)", xaxis_title="Waktu", yaxis_title="Harga USD")
st.plotly_chart(fig, use_container_width=True)
