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
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    if now.hour == 15 and now.minute >= 30:  # At/after 3:30 PM IST
        send_daily_report()
    else:
        run_predictions()
