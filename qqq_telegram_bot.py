import sys
import os
import certifi 
import config 

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.join(base_path, relative_path)

try:
    cacert_path = certifi.where() 
    print(f"程式嘗試使用的 CA 憑證路徑 (certifi.where()): {cacert_path}")
    if os.path.exists(cacert_path):
        os.environ['SSL_CERT_FILE'] = cacert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cacert_path
        print(f"已設定環境變數 SSL_CERT_FILE 和 REQUESTS_CA_BUNDLE 指向: {cacert_path}")
    else:
        print(f"警告：certifi 提供的 CA 憑證路徑 {cacert_path} 不存在。SSL 連線可能會失敗。")

except Exception as e_cert_setup:
    print(f"設定 SSL 憑證路徑時發生錯誤: {e_cert_setup}")
# --- SSL 設定結束 ---

import nest_asyncio
nest_asyncio.apply()
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import datetime
import requests

BOT_TOKEN = config.BOT_TOKEN
MODEL_FILENAME = "tuned_lgbm_long_model_v3.joblib"
EXPECTED_FEATURE_NAMES = [ 
    'QQQ_Open', 'QQQ_High', 'QQQ_Low', 'QQQ_Adj_Close', 'QQQ_Volume',
    'VIX', 'DXY', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI_14',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
    'ATR_14', 'QQQ_Ret_1D', 'QQQ_Ret_3D', 'QQQ_Ret_5D',
    'VIX_Lag1', 'DXY_Lag1', 'market_regime'
]
ENTRY_THRESHOLD = 0.58 
N_PARAM = 1.5
TARGET_RRR_PARAM = 1.3


MODEL_FILENAME = "tuned_lgbm_long_model_v3.joblib"
model_path = resource_path(os.path.join("_internal", MODEL_FILENAME))
model = None

try:
    model = joblib.load(MODEL_FILENAME)
    print(f"模型 '{MODEL_FILENAME}' 已成功加載。")
except Exception as e:
    print(f"啟動時加載模型 '{MODEL_FILENAME}' 失敗: {e}")
    
def get_latest_market_data(tickers, days_history=250):
    """獲取最新的市場數據，確保有足夠歷史計算指標。"""
    print(f"正在獲取最近 {days_history} 天的市場數據...")
    end_date = datetime.date.today()
    start_date_buffer = days_history + 200 
    start_date = end_date - pd.Timedelta(days=start_date_buffer + 50)

    try:
        data = yf.download(tickers, 
                           start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'), 
                           interval='1d', 
                           auto_adjust=False, 
                           actions=True, 
                           progress=False) 

        if data.empty or data.shape[0] < days_history: 
            print(f"錯誤：未能下載到足夠的數據 (需要至少 {days_history} 天，實際獲取 {data.shape[0]} 天)。")
            return None

        df_processed = pd.DataFrame(index=data.index)
        if 'QQQ' in tickers:
            if ('Open', 'QQQ') in data.columns: df_processed['QQQ_Open'] = data[('Open', 'QQQ')]
            if ('High', 'QQQ') in data.columns: df_processed['QQQ_High'] = data[('High', 'QQQ')]
            if ('Low', 'QQQ') in data.columns: df_processed['QQQ_Low'] = data[('Low', 'QQQ')]
            if ('Close', 'QQQ') in data.columns: df_processed['QQQ_Unadj_Close'] = data[('Close', 'QQQ')]
            if ('Adj Close', 'QQQ') in data.columns: df_processed['QQQ_Adj_Close'] = data[('Adj Close', 'QQQ')]
            if ('Volume', 'QQQ') in data.columns: df_processed['QQQ_Volume'] = data[('Volume', 'QQQ')]
        if '^VIX' in tickers and ('Close', '^VIX') in data.columns:
            df_processed['VIX'] = data[('Close', '^VIX')]

        dxy_ticker_primary = 'DX-Y.NYB'; dxy_ticker_fallback = 'UUP'
        dxy_source_used = None 
        if dxy_ticker_primary in tickers and ('Close', dxy_ticker_primary) in data.columns and data[('Close', dxy_ticker_primary)].notna().any():
            df_processed['DXY'] = data[('Close', dxy_ticker_primary)]
            dxy_source_used = dxy_ticker_primary
        elif dxy_ticker_fallback in tickers and ('Adj Close', dxy_ticker_fallback) in data.columns and data[('Adj Close', dxy_ticker_fallback)].notna().any():
            print(f"注意: 使用 UUP 作為 DXY 的替代數據。")
            df_processed['DXY'] = data[('Adj Close', dxy_ticker_fallback)]
            dxy_source_used = dxy_ticker_fallback
        else:
            print(f"警告: DXY 數據 (嘗試了 {dxy_ticker_primary} 和 {dxy_ticker_fallback}) 未能成功獲取或全為NaN。")

        for col in ['VIX', 'DXY']:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].ffill().bfill()

        if 'QQQ_Adj_Close' not in df_processed.columns or df_processed['QQQ_Adj_Close'].isnull().all():
            print("錯誤: QQQ_Adj_Close 數據缺失或全為NaN。")
            return None
        df_processed.dropna(subset=['QQQ_Adj_Close'], inplace=True)

        if df_processed.empty:
            print("處理後數據為空 (可能QQQ數據缺失或VIX/DXY完全缺失)。")
            return None
        print("市場數據獲取成功。")
        return df_processed.tail(days_history) 

    except Exception as e:
        print(f"獲取市場數據時發生錯誤: {e}")
        return None
        
def calculate_features(df):
    print("正在計算特徵...")
    if df is None or df.empty:
        print("錯誤：傳入 calculate_features 的數據為空。")
        return None
    
    df_feat = df.copy()
    try:
        df_feat.ta.sma(close='QQQ_Adj_Close', length=20, append=True, col_names=('SMA_20'))
        df_feat.ta.sma(close='QQQ_Adj_Close', length=50, append=True, col_names=('SMA_50'))
        df_feat.ta.ema(close='QQQ_Adj_Close', length=20, append=True, col_names=('EMA_20'))
        df_feat.ta.rsi(close='QQQ_Adj_Close', length=14, append=True, col_names=('RSI_14'))
        df_feat.ta.macd(close='QQQ_Adj_Close', fast=12, slow=26, signal=9, append=True) # MACD_12_26_9, MACDH_12_26_9, MACDS_12_26_9
        df_feat.ta.bbands(close='QQQ_Adj_Close', length=20, std=2, append=True) # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        
        close_for_atr = 'QQQ_Unadj_Close' if 'QQQ_Unadj_Close' in df_feat.columns and df_feat['QQQ_Unadj_Close'].notna().any() else 'QQQ_Adj_Close'
        if not ('QQQ_High' in df_feat.columns and 'QQQ_Low' in df_feat.columns and close_for_atr in df_feat.columns and \
                df_feat[['QQQ_High', 'QQQ_Low', close_for_atr]].notna().all().all()):
            print(f"警告: ATR 計算所需欄位 (High, Low, {close_for_atr}) 數據不完整或包含NaN值。")
        df_feat.ta.atr(high='QQQ_High', low='QQQ_Low', close=close_for_atr, length=14, append=True, col_names=('ATR_14'))
        
        for n in [1, 3, 5]:
            df_feat[f'QQQ_Ret_{n}D'] = df_feat['QQQ_Adj_Close'].pct_change(n)
            
        for col in ['VIX', 'DXY']:
            if col in df_feat.columns:
                df_feat[f'{col}_Lag1'] = df_feat[col].shift(1)

        sma_long_period = 200
        sma_calc_col = f'SMA_{sma_long_period}_calc'
        df_feat[sma_calc_col] = df_feat['QQQ_Adj_Close'].rolling(window=sma_long_period, min_periods=sma_long_period).mean() # 要求足夠的期數
        
        df_feat['market_regime'] = 0 
        df_feat.loc[df_feat['QQQ_Adj_Close'] > df_feat[sma_calc_col], 'market_regime'] = 1 
        df_feat.loc[df_feat['QQQ_Adj_Close'] < df_feat[sma_calc_col], 'market_regime'] = -1
        
        df_feat.dropna(inplace=True) 
        if df_feat.empty:
            print("計算特徵並移除NaN後數據為空。可能是days_history對於SMA200不足。")
            return None
        print("特徵計算完成。")
        return df_feat
    except Exception as e:
        print(f"計算特徵時發生錯誤: {e}")
        return None
        
# --- Telegram Bot 指令處理函數 ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('你好！使用 /getsignal 指令來獲取最新的QQQ做多訊號評估。')

async def getsignal_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    print(f"收到來自 chat_id {chat_id} 的 /getsignal 指令")

    if model is None:
        await update.message.reply_text("錯誤：模型未能成功加載，請檢查伺服器日誌。")
        return

    await update.message.reply_text("正在獲取最新數據並生成訊號，請稍候...")

    market_data_raw = get_latest_market_data(['QQQ', '^VIX', 'DX-Y.NYB'], days_history=250)
    if market_data_raw is None or market_data_raw.empty:
        await update.message.reply_text("錯誤：無法獲取最新的市場數據。")
        return

    features_df = calculate_features(market_data_raw)
    if features_df is None or features_df.empty:
        await update.message.reply_text("錯誤：無法計算必要的特徵。")
        return

    if not all(feature_name in features_df.columns for feature_name in EXPECTED_FEATURE_NAMES):
        await update.message.reply_text("錯誤：計算後的特徵與模型預期不符，無法預測。")
        return

    latest_features_for_pred = features_df[EXPECTED_FEATURE_NAMES].iloc[-1:]
    if latest_features_for_pred.isnull().any().any():
        await update.message.reply_text("錯誤：最新特徵數據中存在NaN值，無法預測。")
        return

    try:
        pred_proba_latest = model.predict_proba(latest_features_for_pred)[:, 1][0]
        signal_date = features_df.index[-1]

        market_calendar = pd.to_datetime(features_df.index).to_series().asfreq('B').index
        next_trading_day_index = market_calendar[market_calendar > signal_date]
        action_date = next_trading_day_index[0] if not next_trading_day_index.empty else signal_date + pd.Timedelta(days=1)

        atr_on_signal_day = features_df['ATR_14'].iloc[-1]
        proxy_entry_price = features_df['QQQ_Adj_Close'].iloc[-1]

        sl_distance = N_PARAM * atr_on_signal_day
        tp_distance = TARGET_RRR_PARAM * sl_distance
        estimated_sl = proxy_entry_price - sl_distance
        estimated_tp = proxy_entry_price + tp_distance

        signal_triggered_text = "是" if pred_proba_latest > ENTRY_THRESHOLD else "否"

        message = (
            f"*QQQ 做多模型按需查詢結果*\n\n"
            f"訊號基於數據日期: {signal_date.date()}\n"
            f"預計下一交易日: {action_date.date()}\n"
            f"------------------------------------\n"
            f"模型預測盈利機率: *{pred_proba_latest:.2%}*\n"
            f"預設進場閾值: {ENTRY_THRESHOLD:.2%}\n"
            f"是否達到進場閾值: *{signal_triggered_text}*\n"
            f"------------------------------------\n"
            f"若進場 (基於訊號日收盤價 {proxy_entry_price:.2f} 估算)：\n"
            f"  估計止損 (SL): {estimated_sl:.2f}\n"
            f"  估計止盈 (TP): {estimated_tp:.2f}\n"
            f"  (訊號日 ATR: {atr_on_signal_day:.2f})\n\n"
            f"SL/TP 計算規則 (需基於實際進場價):\n"
            f"  止損距離 = {N_PARAM:.1f} x ATR\n"
            f"  止盈距離 = {TARGET_RRR_PARAM:.1f} x 止損距離\n\n"
            f"_(此為模型分析，非投資建議，請謹慎評估風險。)_"
        )
        await update.message.reply_text(message, parse_mode='Markdown')

    except Exception as e:
        print(f"處理 /getsignal 指令時發生錯誤: {e}")
        await update.message.reply_text(f"處理您的請求時發生錯誤: {str(e)}")

# --- Bot 主函數 ---
def run_interactive_bot():
    """啟動並運行 Telegram Bot。"""
    if BOT_TOKEN == "YOUR_BOT_TOKEN" or model is None:
        print("錯誤：BOT_TOKEN 未設定或模型未加載。Bot無法啟動。")
        return

    print("正在啟動互動式 Telegram Bot...")
    application = Application.builder().token(BOT_TOKEN).build()

    # 添加指令處理器
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("getsignal", getsignal_command_handler))

    print("Bot 已啟動，正在輪詢新訊息...")
    application.run_polling()

# 直接運行這個Bot腳本
if __name__ == '__main__':
    run_interactive_bot()