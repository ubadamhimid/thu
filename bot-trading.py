import ccxt
import pandas as pd
import time
import requests
import random
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator

# --- إعداد مفاتيح API ---
api_key = "67c5f96745e41a00016725dc"
api_secret = "18bfe368-9bdb-4e91-abf6-c41eae7ef8fe"
api_passphrase = "Voetbal10!"
telegram_bot_token = "7305418909:AAGOeDSbhc7ugfjyIlzGJm4M_Acpb07cKFk"
telegram_chat_id = "1638104695"

# --- الاتصال بمنصة KuCoin ---
exchange = ccxt.kucoin({
    'apiKey': api_key,
    'secret': api_secret,
    'password': api_passphrase,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# --- جلب جميع أزواج USDT ---
def get_all_usdt_pairs():
    tickers = exchange.fetch_tickers()
    usdt_pairs = [symbol for symbol in tickers if symbol.endswith("/USDT")]
    return usdt_pairs

# --- إعدادات التداول ---
trading_pairs = get_all_usdt_pairs()

def send_telegram_message(message):
    """إرسال إشعار نصي إلى تيليجرام"""
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    requests.post(url, data=payload)

def send_telegram_photo(photo_path):
    """إرسال صورة إلى تيليجرام"""
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendPhoto"
    files = {"photo": open(photo_path, "rb")}
    data = {"chat_id": telegram_chat_id}
    requests.post(url, files=files, data=data)

def fetch_market_data(symbol, timeframe='1h', limit=100):
    """جلب بيانات السوق مع التعامل مع الأخطاء"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        error_message = f"⚠ حدث خطأ أثناء جلب بيانات {symbol}: {e}"
        print(error_message)
        send_telegram_message(error_message)
        return None

def apply_indicators(df):
    """إضافة المؤشرات الفنية"""
    df['SMA50'] = SMAIndicator(df['close'], window=50).sma_indicator()
    df['SMA200'] = SMAIndicator(df['close'], window=200).sma_indicator()
    df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
    df['Resistance'] = df['high'].rolling(window=20).max()
    df['Support'] = df['low'].rolling(window=20).min()
    return df

def generate_trade_signal(df):
    """تحديد إشارات الدخول والخروج وإضافة الهدف والستوب"""
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    target_price = None
    stop_loss_price = None

    if latest['SMA50'] > latest['SMA200'] and previous['RSI'] < 30 and latest['RSI'] > 30:
        target_price = latest['close'] * 1.05  # هدف 5%
        stop_loss_price = latest['close'] * 0.95  # ستوب 5%
        return 'BUY', target_price, stop_loss_price
    elif latest['SMA50'] < latest['SMA200'] and previous['RSI'] > 70 and latest['RSI'] < 70:
        target_price = latest['close'] * 0.95  # هدف 5%
        stop_loss_price = latest['close'] * 1.05  # ستوب 5%
        return 'SELL', target_price, stop_loss_price
    return None, None, None

def plot_and_send_chart(df, symbol):
    """رسم المخطط البياني وإرساله إلى تيليجرام"""
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    plt.plot(df['timestamp'], df['SMA50'], label='SMA50', color='orange')
    plt.plot(df['timestamp'], df['SMA200'], label='SMA200', color='red')
    plt.title(f"{symbol} Price Chart")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig("chart.png")
    plt.close()
    send_telegram_photo("chart.png")

def main():
    """تشغيل البوت بشكل دوري لجميع العملات USDT دون تنفيذ الصفقات"""
    while True:
        try:
            for symbol in trading_pairs:
                df = fetch_market_data(symbol)
                if df is None or df.empty:
                    print(f"⚠ تخطي {symbol} بسبب عدم توفر بيانات السوق")
                    continue
                df = apply_indicators(df)
                signal, target, stop_loss = generate_trade_signal(df)

                if signal:
                    message = f"🔔 إشارة تداول {signal} لزوج {symbol}\n" \
                              f"📊 السعر الحالي: {df['close'].iloc[-1]}\n" \
                              f"🎯 الهدف: {target}\n" \
                              f"🛑 وقف الخسارة: {stop_loss}\n" \
                              f"📈 SMA50: {df['SMA50'].iloc[-1]}\n" \
                              f"📉 SMA200: {df['SMA200'].iloc[-1]}\n" \
                              f"💹 RSI: {df['RSI'].iloc[-1]}"
                    send_telegram_message(message)
                    plot_and_send_chart(df, symbol)

            time.sleep(random.randint(30, 60))
        except Exception as e:
            error_message = f"⚠ حدث خطأ: {e}"
            print(error_message)
            send_telegram_message(error_message)
            time.sleep(60)

if __name__ == "__main__":
    main()
