from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from newsapi import NewsApiClient
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # allows the HTML file to talk to this server

# Load your trained model once at startup
model = pickle.load(open("models/market_model.pkl", "rb"))

# Your NewsAPI key
NEWS_API_KEY = "ececdf9eec604b70b624b6d1d75f87f4"

def compute_features(ticker="^NSEI"):
    data = yf.download(ticker, period="1y")
    data.columns = data.columns.get_level_values(0)
    data["Returns"] = data["Close"].pct_change()
    data["Return_Lag1"] = data["Returns"].shift(1)
    data["Return_Lag2"] = data["Returns"].shift(2)
    data["Return_Lag3"] = data["Returns"].shift(3)
    data["MA10"]  = data["Close"].rolling(10).mean()
    data["MA50"]  = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["Volatility"] = data["Returns"].rolling(10).std()
    data["Momentum"]   = data["Close"] - data["Close"].shift(10)
    delta = data["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + gain / loss))
    data["Geo_Index"] = 0.0
    data.dropna(inplace=True)
    return data

def get_geo_index(keyword):
    newsapi   = NewsApiClient(api_key=NEWS_API_KEY)
    sentiment = pipeline("sentiment-analysis")
    articles  = newsapi.get_everything(q=keyword, language="en", sort_by="publishedAt")
    scores = []
    for a in articles["articles"][:20]:
        res   = sentiment(a["title"][:512])[0]
        score = res["score"] if res["label"] == "POSITIVE" else -res["score"]
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

@app.route("/api/market-data")
def market_data():
    data  = compute_features()
    last  = data.iloc[-1]
    closes = data["Close"].tolist()
    dates  = [str(d.date()) for d in data.index]
    return jsonify({
        "dates":      dates[-60:],
        "closes":     closes[-60:],
        "rsi":        data["RSI"].tolist()[-60:],
        "ma10":       data["MA10"].tolist()[-60:],
        "ma50":       data["MA50"].tolist()[-60:],
        "latest": {
            "close":      float(last["Close"]),
            "rsi":        float(last["RSI"]),
            "momentum":   float(last["Momentum"]),
            "volatility": float(last["Volatility"]),
            "ma10":       float(last["MA10"]),
            "ma50":       float(last["MA50"]),
        }
    })

@app.route("/api/predict/<keyword>")
def predict(keyword):
    data = compute_features()
    geo  = get_geo_index(keyword)
    features = ["Returns","Return_Lag1","Return_Lag2","Return_Lag3",
                "MA10","MA50","MA200","Volatility","Momentum","RSI","Geo_Index"]
    latest = data[features].iloc[-1:].copy()
    latest["Geo_Index"] = float(geo)
    prob   = float(model.predict_proba(latest)[0][1])
    signal = ("Strong Bullish" if prob > 0.7 else
              "Bullish"        if prob > 0.6 else
              "Weak Bullish"   if prob > 0.5 else "Bearish")
    return jsonify({ "geo_index": geo, "bull_prob": prob, "signal": signal })

if __name__ == "__main__":
    app.run(debug=True, port=5000)