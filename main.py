import os
import pandas as pd
import yfinance as yf
import joblib
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import pytz

# -----------------------
# Config
# -----------------------
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

SIGNALS_FILE = "signals_log.csv"
NIFTY50 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","HINDUNILVR.NS",
    "SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS",
    "BAJFINANCE.NS","HCLTECH.NS","WIPRO.NS","SUNPHARMA.NS","ULTRACEMCO.NS","POWERGRID.NS",
    "NTPC.NS","ONGC.NS","BAJAJFINSV.NS","TITAN.NS","MARUTI.NS","NESTLEIND.NS","ADANIGREEN.NS",
    "ADANIPORTS.NS","ADANIENT.NS","TECHM.NS","GRASIM.NS","JSWSTEEL.NS","COALINDIA.NS",
    "HDFCLIFE.NS","BRITANNIA.NS","DIVISLAB.NS","HEROMOTOCO.NS","CIPLA.NS","TATAMOTORS.NS",
    "TATASTEEL.NS","M&M.NS","DRREDDY.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","HINDALCO.NS",
    "APOLLOHOSP.NS","SBILIFE.NS","SHREECEM.NS","BPCL.NS","IOC.NS","INDUSINDBK.NS"
]

# Load ML model + scaler
MODEL_PATH = "ml_model.pkl"
SCALER_PATH = "scaler.pkl"

# -----------------------
# Email sender
# -----------------------
def send_email(subject, body):
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)



# -----------------------
# Train model if missing
# -----------------------
def train_model_if_missing():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return

    print("Training new ML model (first run)...")
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier

    data_all = []
    for ticker in NIFTY50:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            df = df.dropna()
            df["return_next_day"] = df["Close"].pct_change().shift(-1)
            df["target"] = np.where(df["return_next_day"] > 0.03, 1, 
                             np.where(df["return_next_day"] < -0.03, -1, 0))
            df = df.dropna()
            X = df[["Open", "High", "Low", "Close", "Volume"]]
            y = df["target"]
            data_all.append((X, y))
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    if not data_all:
        print("No data for training.")
        return

    X_full = pd.concat([d[0] for d in data_all])
    y_full = pd.concat([d[1] for d in data_all])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    model = LGBMClassifier(n_estimators=300, learning_rate=0.05)
    model.fit(X_scaled, y_full)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Model training complete.")


# -----------------------
# Prediction + logging
# -----------------------
def run_predictions():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Model/scaler not found, skipping predictions")
        return

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    rows = []

    for ticker in NIFTY50:
        try:
            df = yf.download(ticker, period="2d", interval="5m", progress=False)
            df = df.dropna()
            if df.empty: 
                continue

            latest = df.iloc[-1]
            X = scaler.transform([[latest["Open"], latest["High"], latest["Low"], latest["Close"], latest["Volume"]]])
            pred = model.predict(X)[0]

            if pred != 0:  # 1 = BUY, -1 = SELL
                signal = "BUY" if pred == 1 else "SELL"
                rows.append({
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "signal": signal,
                    "price": latest["Close"]
                })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    if rows:
        df_new = pd.DataFrame(rows)
        if os.path.exists(SIGNALS_FILE):
            df_old = pd.read_csv(SIGNALS_FILE)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(SIGNALS_FILE, index=False)
        print(f"Logged {len(rows)} signals")

# -----------------------
# Daily summary
# -----------------------
def send_daily_report():
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    today = now.strftime("%Y-%m-%d")

    if not os.path.exists(SIGNALS_FILE):
        return

    df = pd.read_csv(SIGNALS_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_today = df[df["timestamp"].dt.strftime("%Y-%m-%d") == today]

    if df_today.empty:
        body = "No signals generated today."
    else:
        lines = []
        for _, row in df_today.iterrows():
            lines.append(f"{row['timestamp']} | {row['ticker']} | {row['signal']} @ {row['price']:.2f}")
        body = "\n".join(lines)

    send_email(f"NIFTY50 Daily Report - {today}", body)
    print("Daily report sent")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    train_model_if_missing()
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    if now.hour == 15 and now.minute >= 30:
        send_daily_report()
    else:
        run_predictions()

