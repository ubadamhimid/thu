import requests
import time
import pandas as pd
import numpy as np
import joblib
import asyncio
import threading
from sklearn.linear_model import LogisticRegression
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

# 🔹 إعدادات KuCoin API
KUCOIN_API_URL = "https://api.kucoin.com/api/v1/market/allTickers"

# 🔹 إعدادات تيليجرام
TELEGRAM_BOT_TOKEN = "7305418909:AAGOeDSbhc7ugfjyIlzGJm4M_Acpb07cKFk"
CHAT_ID = "1638104695"
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# 🔄 تشغيل `asyncio` في Thread منفصل
def start_async_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


async_loop = asyncio.new_event_loop()
t = threading.Thread(target=start_async_loop, args=(async_loop,), daemon=True)
t.start()


# 🏦 جلب بيانات السوق من KuCoin API
def get_market_data():
    response = requests.get(KUCOIN_API_URL)
    if response.status_code != 200:
        print(f"❌ خطأ في API: {response.status_code}, {response.text}")
        return []

    data = response.json()
    return data.get("data", {}).get("ticker", [])


# 🔍 تحليل العملات لاختيار أفضل 50 عملة تداول يومي
def get_top_coins(data):
    df = pd.DataFrame(data)
    if df.empty:
        print("⚠️ لا توجد بيانات متاحة من API!")
        return df

    df['changeRate'] = pd.to_numeric(df['changeRate'], errors='coerce').fillna(0) * 100
    df['volValue'] = pd.to_numeric(df['volValue'], errors='coerce').fillna(0)

    # 🔥 اختيار أعلى 50 عملة من حيث التغير اليومي والسيولة
    top_50 = df.nlargest(50, ['changeRate', 'volValue'])
    return top_50[['symbol', 'changeRate', 'last', 'high', 'low', 'volValue']]


# 📊 تحليل العملة: حساب المقاومة والدعم وأهداف التداول
def analyze_coin(symbol, last_price, high_price, low_price, volume):
    resistance = float(high_price)
    support = float(low_price)

    # 🎯 وضع أهداف الربح ووقف الخسارة
    take_profit = float(last_price) * 1.03  # +3%
    stop_loss = float(last_price) * 0.97  # -3%

    # 🔥 تحديد قوة الصفقة بناءً على البيانات
    strength = "⚠️ ضعيفة"
    if volume > 5000000 and (resistance - float(last_price)) / resistance < 0.02:
        strength = "🔥 قوية"
    elif volume > 1000000:
        strength = "✅ متوسطة"

    return resistance, support, take_profit, stop_loss, strength


# 🤖 نموذج الذكاء الاصطناعي للتنبؤ باتجاه السوق
def train_ai_model():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 10]])
    y_train = np.array([0, 0, 1, 1, 1])  # 0 = لا يوجد فرصة، 1 = فرصة جيدة

    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "ai_trading_model.pkl")


def predict_opportunity(change_rate, volume):
    model = joblib.load("ai_trading_model.pkl")
    X_test = np.array([[change_rate, volume]])
    prediction = model.predict(X_test)
    return "✅ فرصة جيدة" if prediction[0] == 1 else "❌ فرصة ضعيفة"


# 📩 إرسال الإشعارات إلى تيليجرام مع أزرار تفاعلية
async def send_telegram_alert(symbol, last_price, resistance, support, take_profit, stop_loss, strength, ai_prediction):
    message = (
        f"🔔 *إشارة تداول - {symbol}* 🔔\n"
        f"💰 السعر الحالي: {last_price}\n"
        f"📈 المقاومة: {resistance}\n"
        f"📉 الدعم: {support}\n"
        f"🎯 هدف الربح: {take_profit}\n"
        f"⛔ وقف الخسارة: {stop_loss}\n"
        f"📊 قوة الصفقة: {strength}\n"
        f"🤖 توقع الذكاء الاصطناعي: {ai_prediction}\n"
    )

    keyboard = [[InlineKeyboardButton("✅ فتح صفقة", callback_data='open_trade')],
                [InlineKeyboardButton("❌ تجاهل", callback_data='ignore')]]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown", reply_markup=reply_markup)


# 🔄 تشغيل الأداة وتحديث البيانات كل 15 دقيقة
open_trades = {}


def update_data():
    global open_trades
    try:
        market_data = get_market_data()
        if not market_data:
            print("❌ لم يتم استرداد بيانات السوق، سيتم إعادة المحاولة بعد 15 دقيقة...")
            return

        top_50_coins = get_top_coins(market_data)
        if top_50_coins.empty:
            print("⚠️ لا توجد بيانات كافية للتحليل، سيتم إعادة المحاولة لاحقًا.")
            return

        for _, row in top_50_coins.iterrows():
            symbol = row['symbol']
            last_price = float(row['last'])
            high_price = float(row['high'])
            low_price = float(row['low'])
            volume = float(row['volValue'])

            resistance, support, take_profit, stop_loss, strength = analyze_coin(symbol, last_price, high_price,
                                                                                 low_price, volume)
            ai_prediction = predict_opportunity(row['changeRate'], volume)

            # 🔔 إشعار دخول جديد
            if symbol not in open_trades and (last_price >= resistance * 0.98 or last_price <= support * 1.02):
                asyncio.run_coroutine_threadsafe(
                    send_telegram_alert(symbol, last_price, resistance, support, take_profit, stop_loss, strength,
                                        ai_prediction),
                    async_loop
                )
                open_trades[symbol] = {"entry": last_price, "take_profit": take_profit, "stop_loss": stop_loss}

            # 📢 إشعار تحقيق الهدف أو ضرب وقف الخسارة
            elif symbol in open_trades:
                trade = open_trades[symbol]
                if last_price >= trade["take_profit"]:
                    asyncio.run_coroutine_threadsafe(
                        bot.send_message(chat_id=CHAT_ID,
                                         text=f"🎉 *{symbol}* وصلت إلى هدف الربح {trade['take_profit']} ✅",
                                         parse_mode="Markdown"),
                        async_loop
                    )
                    del open_trades[symbol]
                elif last_price <= trade["stop_loss"]:
                    asyncio.run_coroutine_threadsafe(
                        bot.send_message(chat_id=CHAT_ID,
                                         text=f"❌ *{symbol}* ضرب وقف الخسارة عند {trade['stop_loss']} ⛔",
                                         parse_mode="Markdown"),
                        async_loop
                    )
                    del open_trades[symbol]

        print("✅ تم التحديث، سيتم الفحص مرة أخرى بعد 15 دقيقة...")
    except Exception as e:
        print(f"❌ خطأ أثناء التحديث: {e}")


# 🚀 تدريب الذكاء الاصطناعي لأول مرة
train_ai_model()

# 🔁 تشغيل التحديث التلقائي كل 15 دقيقة
while True:
    update_data()
    time.sleep(900)  # 900 ثانية = 15 دقيقة
