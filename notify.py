"""
Telegram alerts.

Settings (bot token, chat id, which events) live in data/telegram.json, not in git.

    python app.py --setup-telegram

Create the bot first: in Telegram, message @BotFather, send /newbot, pick a
name, and copy the token it gives you.
"""

import json
import queue
import threading
import time

import requests

import exchange
import scanner

CONFIG_PATH = scanner.DATA_DIR / "telegram.json"
DEFAULTS = {
    "enabled": True,
    "events": "confirmed",              # "confirmed" = after the 1h check passes; "all" = every new alert
    "setups": ["trend"],                # "trend", "ignition"
    "trend_changes": True,              # BTC daily trend turning on/off
}

_q = queue.Queue()
_worker = None
last_error = None


def config():
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return None
    return {**DEFAULTS, **cfg} if cfg.get("token") and cfg.get("chat_id") else None


def public_config():
    """Settings safe to show in the browser (no token)."""
    cfg = config()
    if not cfg:
        return {"configured": False}
    return {"configured": True, **{k: cfg[k] for k in DEFAULTS}, "last_error": last_error}


def update(settings):
    cfg = config()
    if not cfg:
        raise ValueError("Telegram isn't set up yet")
    for k in DEFAULTS:
        if k in settings:
            cfg[k] = settings[k]
    CONFIG_PATH.write_text(json.dumps(cfg, indent=1))
    return public_config()


def wants(alert, event):
    """Should this alert be sent for this event ("new" or "confirmed")?"""
    cfg = config()
    if not cfg or not cfg["enabled"]:
        return False
    if alert.get("setup") not in cfg["setups"]:
        return False
    return (cfg["events"] == "all" and event == "new") or (cfg["events"] == "confirmed" and event == "confirmed")


def _api(cfg, method, **params):
    r = requests.post(f"https://api.telegram.org/bot{cfg['token']}/{method}", json=params, timeout=20)
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("description", f"HTTP {r.status_code}"))
    return data["result"]


def _run():
    global last_error
    while True:
        text = _q.get()
        cfg = config()
        if not cfg:
            continue
        for attempt in range(3):
            try:
                _api(cfg, "sendMessage", chat_id=cfg["chat_id"], text=text, parse_mode="HTML",
                     disable_web_page_preview=True)
                last_error = None
                break
            except Exception as e:
                last_error = f"{time.strftime('%H:%M')}: {e}"
                time.sleep(3 * (attempt + 1))
        time.sleep(1.1)  # Telegram allows ~1 message/second per chat


def send(text):
    """Queue a message; a background thread delivers it (never blocks the scanner)."""
    global _worker
    if not config():
        return False
    if _worker is None:
        _worker = threading.Thread(target=_run, daemon=True)
        _worker.start()
    _q.put(text)
    return True


def _h(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_alerts(alerts, event):
    """One message for a batch of alerts from the same scan."""
    sig = lambda v: "—" if v is None else f"{v * 100:+.1f}%"
    head = "✅ <b>Passed the 1h check</b>" if event == "confirmed" else "🔔 <b>New alert</b>"
    lines = [head + (f" · {len(alerts)}" if len(alerts) > 1 else "")]
    for a in alerts[:10]:
        base = a["symbol"].removesuffix("USDT")
        url = exchange.trade_url(a["symbol"])
        setup = "Trend" if a["setup"] == "trend" else "Ignition"
        detail = (f"{sig(a.get('chg_1h'))} after 1h" if event == "confirmed"
                  else f"vol {a.get('vol_x') or 0:.1f}× · candle {sig(a.get('chg'))}")
        tag = " · ⚠️ risk" if a.get("tags") else ""
        lines.append(f"<b>{_h(base)}</b>/USDT · {setup} · {detail}"
                     f" · price {a['price']:.6g}{tag} · <a href=\"{url}\">chart</a>")
    if len(alerts) > 10:
        lines.append(f"…and {len(alerts) - 10} more on the dashboard")
    lines.append("<i>Backtest: these alerts don't predict direction better than chance after fees. Not advice.</i>")
    return "\n".join(lines)


def setup_cli():
    print("Telegram setup\n")
    print("1. In Telegram, open @BotFather, send /newbot, choose a name and a username ending in 'bot'.")
    print("2. BotFather replies with a token like 123456789:AA... — paste it here.\n")
    token = input("Bot token: ").strip()
    cfg = {"token": token}
    try:
        me = _api(cfg, "getMe")
    except Exception as e:
        print(f"That token didn't work ({e}). Check it and run --setup-telegram again.")
        return
    print(f"\nConnected to @{me['username']}.")
    print(f"3. Now open https://t.me/{me['username']} in Telegram and press START (or send any message).")
    print("   Waiting up to 3 minutes…")
    offset, chat = None, None
    deadline = time.time() + 180
    while time.time() < deadline and not chat:
        try:
            updates = _api(cfg, "getUpdates", timeout=20, **({"offset": offset} if offset else {}))
        except Exception:
            time.sleep(3)
            continue
        for u in updates:
            offset = u["update_id"] + 1
            msg = u.get("message") or u.get("my_chat_member") or {}
            if msg.get("chat", {}).get("type") == "private":
                chat = msg["chat"]
    if not chat:
        print("Didn't see your message. Press START in the bot chat, then run --setup-telegram again.")
        return
    scanner.DATA_DIR.mkdir(exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({**DEFAULTS, "token": token, "chat_id": chat["id"]}, indent=1))
    _api({"token": token}, "sendMessage", chat_id=chat["id"],
         text="✅ Pump Scanner connected. You'll get Trend alerts that pass the 1h check, and BTC daily trend changes. "
              "Change this on the Alerts tab.")
    print(f"\nSaved. Sent a test message to {chat.get('first_name', 'you')}. Telegram alerts are on.")
