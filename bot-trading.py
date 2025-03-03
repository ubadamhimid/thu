import requests
import time
import pandas as pd
import numpy as np
import joblib
import asyncio
import threading
from sklearn.linear_model import LogisticRegression
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

# إعدادات KuCoin API
KUCOIN_API_URL = "https://api.kucoin.com/api/v1/market/allTickers"

# إعدادات تيليجرام
TELEGRAM_BOT_TOKEN = "7305418909:AAGOeDSbhc7ugfjyIlzGJm4M_Acpb07cKFk"
CHAT_ID = "1638104695"
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# تشغيل asyncio في Thread منفصل
def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async_loop = asyncio.new_event_loop()
t = threading.Thread(target=start_async_loop, args=(async_loop,), daemon=True)
t.start()

# جلب بيانات السوق
def get_market_data():
    response = requests.get(KUCOIN_API_URL)
    if response.status_code != 200:
        return []
    data = response.json()
    return data.get("data", {}).get("ticker", [])

# تحليل العملات

def get_top_coins(data):
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['changeRate'] = pd.to_numeric(df['changeRate'], errors='coerce').fillna(0) * 100
    df['volValue'] = pd.to_numeric(df['volValue'], errors='coerce').fillna(0)
    top_50 = df.nlargest(50, ['changeRate', 'volValue'])
    return top_50[['symbol', 'changeRate', 'last', 'high', 'low', 'volValue']]

# تحليل الصفقات

def analyze_coin(symbol, last_price, high_price, low_price, volume):
    resistance = float(high_price)
    support = float(low_price)
    take_profit = float(last_price) * 1.03
    stop_loss = float(last_price) * 0.97
    return resistance, support, take_profit, stop_loss

# تتبع الصفقات
open_trades = {}
trade_stats = {"wins": 0, "losses": 0, "total": 0}

def update_data():
    global open_trades, trade_stats
    try:
        market_data = get_market_data()
        if not market_data:
            return
        top_50_coins = get_top_coins(market_data)
        if top_50_coins.empty:
            return
        
        for _, row in top_50_coins.iterrows():
            symbol = row['symbol']
            last_price = float(row['last'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            volume = float(row['volValue'])
            
            resistance, support, take_profit, stop_loss = analyze_coin(symbol, last_price, high_price, low_price, volume)
            
            if symbol not in open_trades:
                open_trades[symbol] = {"entry": last_price, "take_profit": take_profit, "stop_loss": stop_loss}
                asyncio.run_coroutine_threadsafe(
                    bot.send_message(chat_id=CHAT_ID, text=f"🔔 دخول صفقة {symbol} بسعر {last_price}"), async_loop
                )
            
            trade = open_trades[symbol]
            if last_price >= trade["take_profit"]:
                trade_stats["wins"] += 1
                asyncio.run_coroutine_threadsafe(
                    bot.send_message(chat_id=CHAT_ID, text=f"✅ {symbol} حققت الهدف عند {trade['take_profit']}"), async_loop
                )
                del open_trades[symbol]
            elif last_price <= trade["stop_loss"]:
                trade_stats["losses"] += 1
                asyncio.run_coroutine_threadsafe(
                    bot.send_message(chat_id=CHAT_ID, text=f"❌ {symbol} ضرب وقف الخسارة عند {trade['stop_loss']}"), async_loop
                )
                del open_trades[symbol]
            
        trade_stats["total"] = trade_stats["wins"] + trade_stats["losses"]
        if trade_stats["total"] % 20 == 0:  # تحديث الإحصائيات كل 20 صفقة
            asyncio.run_coroutine_threadsafe(
                bot.send_message(chat_id=CHAT_ID, text=f"📊 الإحصائيات: {trade_stats['wins']} ربح / {trade_stats['losses']} خسارة"), async_loop
            )
    except Exception as e:
        print(f"❌ خطأ أثناء التحديث: {e}")

# تشغيل التحديث التلقائي كل 15 دقيقة لمدة 5 ساعات
start_time = time.time()
while time.time() - start_time < 18000:  # 18000 ثانية = 5 ساعات
    update_data()
    time.sleep(900)
