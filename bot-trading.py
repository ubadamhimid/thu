import requests
import json
import websocket
import threading
import time
import numpy as np
import pandas as pd
import ta
import hmac
import hashlib
import base64
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# إعداد Flask API
app = Flask(__name__)

# إعدادات KuCoin API
KUCOIN_API_KEY = "67c0ee0d02d39e00012a8fc6"
KUCOIN_API_SECRET = "0ac4fa7a-3ba3-45bb-ab59-8c73ed4371c4"
KUCOIN_API_PASSPHRASE = "Voetbal10!"

# إعدادات Telegram
TELEGRAM_BOT_TOKEN = "7305418909:AAGOeDSbhc7ugfjyIlzGJm4M_Acpb07cKFk"
TELEGRAM_CHAT_ID = "1638104695"



# جلب جميع العملات المتاحة على KuCoin

def get_all_symbols():
    url = "https://api.kucoin.com/api/v1/symbols"
    response = requests.get(url)
    data = response.json()
    symbols = [item['symbol'] for item in data['data'] if item['symbol'].endswith("-USDT")]
    return symbols[:10]  # الحد من العملات لتقليل الضغط


SYMBOLS = get_all_symbols()
market_data = {symbol: {"price": [], "volume": []} for symbol in SYMBOLS}
signals = []


# إرسال إشعارات إلى Telegram

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=data)


# تحليل البيانات وجمع الميزات الإضافية

def collect_historical_data():
    dataset = []
    for symbol in SYMBOLS:
        if len(market_data[symbol]["price"]) < 50:
            continue

        close_prices = pd.Series(market_data[symbol]["price"])
        volume_data = pd.Series(market_data[symbol]["volume"])

        atr = ta.volatility.AverageTrueRange(high=close_prices, low=close_prices, close=close_prices,
                                             window=14).average_true_range().iloc[-1]
        vwap = (close_prices * volume_data).sum() / volume_data.sum()

        bb_indicator = ta.volatility.BollingerBands(close_prices, window=20)
        upper_band = bb_indicator.bollinger_hband().iloc[-1]
        lower_band = bb_indicator.bollinger_lband().iloc[-1]

        rsi = ta.momentum.RSIIndicator(close_prices, window=14).rsi().iloc[-1]
        macd_indicator = ta.trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9)
        macd = macd_indicator.macd().iloc[-1]
        macdsignal = macd_indicator.macd_signal().iloc[-1]

        price_change = ((close_prices.iloc[-1] / close_prices.iloc[-5]) - 1) * 100  # تحليل آخر 5 شمعات بدلاً من البداية
        volume_spike = (volume_data.iloc[-1] / volume_data.mean()) * 100 if volume_data.mean() > 0 else 0

        ema_9 = ta.trend.ema_indicator(close_prices, window=9).iloc[-1]
        ema_21 = ta.trend.ema_indicator(close_prices, window=21).iloc[-1]
        ema_crossover = 1 if ema_9 > ema_21 else 0

        dataset.append(
            [upper_band, lower_band, rsi, macd, macdsignal, price_change, volume_spike, atr, vwap, ema_crossover])
    return np.array(dataset)


# تدريب النموذج المحسن

def train_ai_model():
    data = collect_historical_data()
    if len(data) < 100:
        return None
    labels = [1 if row[5] > 10 and row[9] == 1 else 0 for row in data]  # إشارات أكثر دقة
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    return model


# API لجلب الإشارات المخزنة
@app.route("/api/signals", methods=["GET"])
def get_signals():
    return jsonify({"signals": signals})


# الكشف في الوقت الحقيقي

def detect_pump_with_ai(model):
    while True:
        for symbol in SYMBOLS:
            if len(market_data[symbol]["price"]) < 50:
                continue

            close_prices = pd.Series(market_data[symbol]["price"])
            volume_data = pd.Series(market_data[symbol]["volume"])
            atr = ta.volatility.AverageTrueRange(high=close_prices, low=close_prices, close=close_prices,
                                                 window=14).average_true_range().iloc[-1]
            vwap = (close_prices * volume_data).sum() / volume_data.sum()

            rsi = ta.momentum.RSIIndicator(close_prices, window=14).rsi().iloc[-1]
            macd_indicator = ta.trend.MACD(close_prices, window_slow=26, window_fast=12, window_sign=9)
            macd = macd_indicator.macd().iloc[-1]
            macdsignal = macd_indicator.macd_signal().iloc[-1]
            price_change = ((close_prices.iloc[-1] / close_prices.iloc[-5]) - 1) * 100
            volume_spike = (volume_data.iloc[-1] / volume_data.mean()) * 100 if volume_data.mean() > 0 else 0

            ema_9 = ta.trend.ema_indicator(close_prices, window=9).iloc[-1]
            ema_21 = ta.trend.ema_indicator(close_prices, window=21).iloc[-1]
            ema_crossover = 1 if ema_9 > ema_21 else 0

            prediction_data = np.array(
                [[upper_band, lower_band, rsi, macd, macdsignal, price_change, volume_spike, atr, vwap, ema_crossover]])
            prediction = model.predict(prediction_data)

            if prediction[0] == 1:
                signal_message = f"🚀 تنبيه: {symbol} قد يشهد ارتفاعًا قويًا خلال الدقائق القادمة!"
                send_telegram_message(signal_message)
                signals.append({"symbol": symbol, "message": signal_message})
        time.sleep(120)  # تحديث كل دقيقتين بدلاً من 5 دقائق


# تشغيل النموذج وتحليل البيانات في الوقت الحقيقي
if __name__ == "__main__":
    ai_model = train_ai_model()
    if ai_model:
        threading.Thread(target=detect_pump_with_ai, args=(ai_model,)).start()
    app.run(host="0.0.0.0", port=5000, debug=True)