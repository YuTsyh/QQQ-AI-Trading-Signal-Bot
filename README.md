# QQQ AI 量化交易訊號 Bot (小規模測試專案)

本專案為「優式AI量化新星計畫」而開發，旨在探索、建立並評估一個基於AI模型的QQQ指數ETF做多交易策略。專案實現了從數據獲取、特徵工程、模型訓練、訊號產生到Telegram Bot互動查詢的完整流程。

## 專案目標
* 運用機器學習方法（LightGBM）預測QQQ的做多交易機會。
* 建立一套包含進場、止損、止盈規則的交易策略。
* 透過歷史數據回測評估策略的有效性，追求正期望值和合理的勝率。
* 實現一個Telegram Bot，能按需提供最新的交易訊號評估。
* 嘗試將應用打包為Windows可執行檔案。

## 主要功能
* 每日自動（需配合排程）或按需（透過Bot指令）獲取最新的QQQ、VIX、DXY市場數據。
* 基於獲取的數據計算28項特徵，包括技術指標（SMA, EMA, RSI, MACD, BBands, ATR）、市場情緒指標（VIX, DXY的滯後值）以及市場狀態（基於QQQ與SMA200的關係判斷牛熊）。
* 使用預先訓練好的LightGBM模型 (`tuned_lgbm_long_model_v3.joblib`) 預測下一交易日做多盈利的機率。
* 當預測機率超過預設閾值 (0.58) 時，產生做多訊號。
* 交易結構：止損設置為進場價 - 1.5 x ATR；止盈設置為進場價 + 1.3 x (1.5 x ATR)。
* 實現Telegram Bot，可透過 `/getsignal` 指令查詢當前最新的訊號評估。

## 方法論與工作流程
1.  **數據獲取：** 使用 `yfinance` 函式庫下載QQQ、^VIX、DX-Y.NYB (或UUP作為替代) 的每日歷史數據 (2010年至今)。
2.  **特徵工程：** 計算了包括移動平均線、RSI、MACD、布林通道、ATR、短期報酬率、VIX/DXY滯後值以及基於SMA200的`market_regime`等28個特徵。
3.  **目標變數定義：**
    * 做多 (`y_long_outcome`)：基於 `N=1.5` (ATR乘數) 設定止損，`target_rrr_long=1.3` 設定止盈。若先觸及止盈則標記為1（盈利），先觸及止損則標記為0（虧損）。
    * 做空 (`y_short_outcome`)：初步探索顯示效果不佳，故專案主要聚焦於做多策略。
4.  **模型訓練與選擇：**
    * 選用 `LightGBMClassifier`。
    * 對做多模型進行了超參數調優 (`RandomizedSearchCV` 配合 `TimeSeriesSplit`)，並針對 `target_rrr_long=1.3` 的標籤進行了訓練。
    * 通過預測機率閾值分析，選定最佳操作點 (閾值0.58)。
5.  **回測與評估：**
    * 在測試集 (2022-04-04 到 2025-04-21) 上對選定的做多策略進行了回測。
    * 評估指標包括勝率、期望值、總回報率、最大回撤、夏普比率，並與同期QQQ買入並持有策略進行比較。
6.  **Telegram Bot 與打包：**
    * 使用 `python-telegram-bot` 函式庫實現互動式查詢。
    * 使用 `PyInstaller` 嘗試打包為Windows執行檔，並解決了執行過程中的依賴、SSL憑證等問題。

## 主要回測成果 (做多策略 @ 閾值0.58, N=1.5, RRR=1.3)
* **回測期間：** 2022-04-04 到 2025-04-21 (約3年)
* **總交易次數：** 38 次
* **勝率 (實際)：** 55.26%
* **平均每次盈利：** +4.92%
* **平均每次虧損：** -3.93%
* **已實現風險報酬比 (AvgWin/AvgLoss)：** 1.25
* **策略期望值 (WR\*RRR - LR)：** 0.2450
* **策略總回報率 (複利)：** +38.58%
* **同期 QQQ 買入並持有回報率：** +19.70%
* **策略最大回撤：** -13.45%
* **策略年化夏普比率 (估算)：** 0.76
* **QQQ同期年化夏普比率 (估算)：** 0.37

## 安裝與設定
1.  **環境要求：** Python 3.11+ (建議與 `requirements.txt` 中的版本一致)。
2.  **複製檔案：**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/Looooooong/QQQ-AI-Trading-Signal-Bot.git)
    ```
3.  **建立並啟用虛擬環境 (推薦)：**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    # source venv/bin/activate
    ```
4.  **安裝依賴：**
    ```bash
    pip install -r requirements.txt
    ```
5.  **設定組態檔：**
    * 在 `config.py` 中填入你的 Telegram Bot Token：
        ```python
        # config.py
        BOT_TOKEN = "YOUR_ACTUAL_TELEGRAM_BOT_TOKEN_HERE"
        ```

## 使用說明
1.  **啟動 Telegram Bot (互動模式)：**
    在已啟用虛擬環境並完成設定後，從專案根目錄運行：
    ```bash
    python qqq_telegram_bot.py
    ```
    Bot啟動後，你可以在Telegram中與其對話：
    * `/start`：查看歡迎訊息。
    * `/getsignal`：獲取對下一交易日的QQQ做多訊號評估。
