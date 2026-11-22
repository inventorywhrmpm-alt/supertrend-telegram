import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf

MAX_SELECTED_STOCKS = 10


def send_telegram_message(bot_token: str, chat_id: str, message: str) -> tuple[bool, str]:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            return True, "Notifikasi berhasil dikirim ke Telegram."
        return False, f"Gagal kirim Telegram: {data.get('description', 'Unknown error')}"
    except Exception as e:
        return False, f"Error koneksi Telegram: {e}"


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 1.0) -> pd.DataFrame:
    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=period, mamode="rma")
    df["src"] = (df["High"] + df["Low"]) / 2

    df["up_base"] = df["src"] - (multiplier * df["atr"])
    df["dn_base"] = df["src"] + (multiplier * df["atr"])

    close = df["Close"].tolist()
    upperband = df["up_base"].tolist()
    lowerband = df["dn_base"].tolist()

    size = len(df)
    trend = [1] * size

    for i in range(1, size):
        if np.isnan(upperband[i]) or np.isnan(lowerband[i]):
            continue

        if close[i - 1] > upperband[i - 1]:
            upperband[i] = max(upperband[i], upperband[i - 1])

        if close[i - 1] < lowerband[i - 1]:
            lowerband[i] = min(lowerband[i], lowerband[i - 1])

        if close[i] > lowerband[i - 1] and trend[i - 1] == -1:
            trend[i] = 1
        elif close[i] < upperband[i - 1] and trend[i - 1] == 1:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    df["trend"] = trend
    df["signal"] = "-"
    df.loc[(df["trend"] == 1) & (df["trend"].shift(1) == -1), "signal"] = "buy"
    df.loc[(df["trend"] == -1) & (df["trend"].shift(1) == 1), "signal"] = "sell"

    return df


def parse_selected_tickers(raw_text: str, max_stocks: int = MAX_SELECTED_STOCKS) -> list[str]:
    cleaned = raw_text.replace("\n", ",").replace(";", ",")
    tickers = []
    for item in cleaned.split(","):
        sym = item.strip().upper()
        if not sym:
            continue
        if not sym.endswith(".JK"):
            sym = f"{sym}.JK"
        if sym not in tickers:
            tickers.append(sym)
        if len(tickers) >= max_stocks:
            break
    return tickers


def load_sent_signals(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return set(str(x) for x in data)
    except Exception as e:
        print(f"Peringatan: gagal membaca state file ({e}), mulai dari kosong.")
    return set()


def save_sent_signals(state_path: Path, keys: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed = sorted(keys)[-500:]
    state_path.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")


def format_message(signals_df: pd.DataFrame, total_selected: int, period: int, multiplier: float) -> str:
    buy_df = signals_df[signals_df["signal"] == "buy"]
    sell_df = signals_df[signals_df["signal"] == "sell"]

    lines = [
        f"<b>Sinyal Supertrend ({total_selected} Saham Pilihan)</b>",
        f"Param: ATR({period}), Mult({multiplier}) | Baru: <b>{len(signals_df)}</b> (BUY: {len(buy_df)} | SELL: {len(sell_df)})",
        "",
    ]

    if not buy_df.empty:
        lines.append("<b>🟢 SINYAL BUY</b>")
        for idx, (_, r) in enumerate(buy_df.iterrows(), start=1):
            lines.append(f"{idx}. <b>{r['emiten']}</b> ({r['tanggal']}) — Close: Rp {r['close']:,.0f}")
        lines.append("")

    if not sell_df.empty:
        lines.append("<b>🔴 SINYAL SELL</b>")
        for idx, (_, r) in enumerate(sell_df.iterrows(), start=1):
            lines.append(f"{idx}. <b>{r['emiten']}</b> ({r['tanggal']}) — Close: Rp {r['close']:,.0f}")

    return "\n".join(lines).strip()


def main() -> None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    default_10_stocks = "BRMS,BBRI,BUMI,ERAL,DOOH,MBMA,CDIA,INET,SCMA"
    raw_watchlist = os.getenv("WATCHLIST", default_10_stocks)
    atr_period = int(os.getenv("ATR_PERIOD", "10"))
    atr_mult = float(os.getenv("ATR_MULTIPLIER", "1.0"))
    state_file = Path(os.getenv("STATE_FILE", "state/sent_signals.json"))

    if not bot_token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN dan TELEGRAM_CHAT_ID wajib diisi.")

    tickers = parse_selected_tickers(raw_watchlist, MAX_SELECTED_STOCKS)
    if not tickers:
        raise ValueError("Masukkan minimal 1 kode saham pilihan pada WATCHLIST.")

    print(f"Memindai {len(tickers)} saham pilihan: {', '.join(tickers)}")

    sent_keys = load_sent_signals(state_file)
    new_signal_rows = []
    new_keys_to_record = []

    for sym in tickers:
        try:
            df_raw = yf.download(sym, period="1y", interval="1d", progress=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)

            if df_raw.empty or len(df_raw) < atr_period + 2:
                print(f"[SKIP] {sym}: data tidak cukup.")
                continue

            df_res = calculate_supertrend(df_raw.copy(), atr_period, atr_mult)
            latest_idx = df_res.index[-1]
            latest_row = df_res.iloc[-1]
            sig = str(latest_row["signal"])
            tgl = latest_idx.strftime("%d-%b-%y")
            close_price = float(latest_row["Close"])

            if sig in ("buy", "sell"):
                sig_key = f"{sym}_{tgl}_{sig}_{atr_period}_{atr_mult}"
                if sig_key not in sent_keys:
                    new_signal_rows.append({
                        "emiten": sym,
                        "tanggal": tgl,
                        "signal": sig,
                        "close": close_price,
                    })
                    new_keys_to_record.append(sig_key)
                else:
                    print(f"[DUPLIKAT] {sym} ({tgl} - {sig}) sudah pernah dikirim.")
        except Exception as e:
            print(f"[ERROR] {sym}: {e}")

    if not new_signal_rows:
        print("Tidak ada sinyal BUY/SELL baru dari saham pilihan.")
        return

    msg = format_message(pd.DataFrame(new_signal_rows), len(tickers), atr_period, atr_mult)
    ok, info = send_telegram_message(bot_token, chat_id, msg)
    print(info)

    if ok:
        sent_keys.update(new_keys_to_record)
        save_sent_signals(state_file, sent_keys)
    else:
        raise RuntimeError(info)


if __name__ == "__main__":
    main()
