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
import prepump
import scanner

CONFIG_PATH = scanner.DATA_DIR / "telegram.json"
DEFAULTS = {
    "enabled": True,
    "events": "confirmed",              # "confirmed" = after the 1h check passes; "all" = every new alert
    "setups": ["trend"],                # "trend", "ignition"
    "trend_changes": True,              # BTC daily trend turning on/off
    "prepump": "squeeze",               # pre-pump watch: "off", "squeeze" (low long/short ratio only), "all"
    "blowoff": "off",                   # blow-off short setups: "off", "weak" (weak altcoin market only), "all"
    "radar": "off",                     # big-move radar picks: "off", "all"
    "early": "2",                       # early entries: "off", "2" or "3" signals lining up on one coin
    "atr": "confirmed",                 # ATR_SM 4h Buy: "off", "confirmed" (candle closed), "all" (also blinking)
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


def wants_prepump(flag):
    cfg = config()
    if not cfg or not cfg["enabled"] or cfg["prepump"] == "off":
        return False
    return cfg["prepump"] == "all" or (flag.get("ls") is not None and flag["ls"] <= prepump.SQUEEZE_LS)


def format_prepump(flags):
    lines = [f"👀 <b>Pre-pump watch</b>" + (f" · {len(flags)}" if len(flags) > 1 else "")]
    for f in flags[:10]:
        base = f["symbol"].removesuffix("USDT")
        squeeze = f.get("ls") is not None and f["ls"] <= prepump.SQUEEZE_LS
        ls = "L/S —" if f.get("ls") is None else f"L/S {f['ls']:.2f}" + (" 🔥 squeeze setup" if squeeze else "")
        tag = " · ⚠️ risk" if f.get("tags") else ""
        if (f.get("vol_before") or 0) >= prepump.VOLATILE_X:
            tag += f" · ⚡ volatile (volume already {f['vol_before']:.0f}× before)"
        lines.append(f"<b>{_h(base)}</b>/USDT · vol {f['vol_x']:.0f}× · candles {f['rng_x']:.1f}× · "
                     f"4h {f['ret4'] * 100:+.1f}% · {ls} · price {f['price']:.6g}{tag} · "
                     f"<a href=\"{exchange.trade_url(f['symbol'])}\">chart</a>")
    if len(flags) > 10:
        lines.append(f"…and {len(flags) - 10} more on the dashboard")
    lines.append("<i>A watchlist, not a buy signal: last month ~5% of flags (8% of squeeze setups) "
                 "touched +30% within 24h; most did not. Not advice.</i>")
    return "\n".join(lines)


def wants_radar():
    cfg = config()
    return bool(cfg and cfg["enabled"] and cfg.get("radar") == "all")


def format_radar(picks, record):
    lines = ["📡 <b>Radar: big move likely</b>" + (f" · {len(picks)}" if len(picks) > 1 else "")]
    for p in picks[:10]:
        base = p["symbol"].removesuffix("USDT")
        why = ", ".join(p.get("reasons") or [])
        tag = " · ⚠️ risk" if p.get("tags") else ""
        lines.append(f"<b>{_h(base)}</b>/USDT · price {p['price']:.6g} · {_h(why)}{tag} · "
                     f"<a href=\"{exchange.trade_url(p['symbol'])}\">chart</a>")
    if record and record.get("n"):
        lines.append(f"<i>Since the model's cut-off: {record['hit30'] * 100:.0f}% of picks touched +30% within 24h, "
                     f"but {record['drop15'] * 100:.0f}% dropped −15%. Up OR down — not a buy signal.</i>")
    return "\n".join(lines)


def atr_mode():
    cfg = config()
    if not cfg or not cfg["enabled"] or cfg.get("atr", "off") == "off":
        return None
    return cfg["atr"]


def format_atr(items, btc4h):
    conf = [x for x in items if x["status"] == "confirmed"]
    blink = [x for x in items if x["status"] == "blinking"]
    lines = []
    for title, group in (("▲ <b>ATR Buy · 4h confirmed</b>", conf), ("◌ <b>ATR Buy blinking · 4h candle still open</b>", blink)):
        if not group:
            continue
        lines.append(title + (f" · {len(group)}" if len(group) > 1 else ""))
        for x in group[:12]:
            base = x["symbol"].removesuffix("USDT")
            h1 = {1: "1h Buy ✓", -1: "1h Sell ✗"}.get(x.get("h1"), "1h —")
            line = (x["price"] - x["line"]) / x["price"]
            lines.append(f"<b>{_h(base)}</b>/USDT · price {x['price']:.6g} · ATR line −{line * 100:.0f}% · {h1} · "
                         f"<a href=\"{exchange.trade_url(x['symbol'])}\">chart</a>")
        if len(group) > 12:
            lines.append(f"…and {len(group) - 12} more in the ATR tab")
    if btc4h:
        weak = btc4h["score"] <= 2
        lines.append(f"BTC 4h: {'bearish ⚠️ alts fall harder, use smaller size' if weak else 'ok'} ({btc4h['score']}/{btc4h['score_max']})")
    lines.append("<i>Blinking: about half fade before the close. Use your own stop (~−5%), the ATR line is far. Not advice.</i>")
    return "\n".join(lines)


def early_threshold():
    cfg = config()
    if not cfg or not cfg["enabled"] or cfg.get("early", "off") == "off":
        return 0
    return int(cfg["early"])


def format_early(boards, btc4h):
    lines = ["🚀 <b>Early entry</b>" + (f" · {len(boards)} coins" if len(boards) > 1 else "")]
    for b in boards[:10]:
        base = b["symbol"].removesuffix("USDT")
        sig = ", ".join(x["label"] for x in b["signals"])
        price = f" · price {b['price']:.6g}" if b.get("price") else ""
        lines.append(f"<b>{_h(base)}</b>/USDT · {b['score']} signals: {_h(sig)}{price} · "
                     f"<a href=\"{exchange.trade_url(b['symbol'])}\">chart</a>")
    if btc4h:
        weak = btc4h["score"] <= 2
        lines.append(f"BTC 4h: {'bearish ⚠️ alts fall harder, use smaller size' if weak else 'ok'} ({btc4h['score']}/{btc4h['score_max']})")
    lines.append("<i>Blinking = 4h candle not closed yet (about half fade). Use your own stop (~−5%). Not advice.</i>")
    return "\n".join(lines)


def wants_blowoff(item):
    cfg = config()
    if not cfg or not cfg["enabled"] or cfg["blowoff"] == "off":
        return False
    return cfg["blowoff"] == "all" or item.get("weak_market") is True


def format_blowoff(items, phase):
    market = "" if not phase else (" · altcoin market: weak ✅" if phase["weak"] else " · altcoin market: strong ⚠️")
    lines = [f"📉 <b>Blow-off short setup</b> (paper){market}" + (f" · {len(items)}" if len(items) > 1 else "")]
    for x in items[:10]:
        base = x["symbol"].removesuffix("USDT")
        tag = " · ⚠️ risk" if x.get("tags") else ""
        e = x["entry"]
        lines.append(f"<b>{_h(base)}</b>/USDT · ran {x['run_24h'] * 100:+.0f}% on {x['vol_x']:.0f}× volume, now 10% off the top · "
                     f"short {e:.6g} → target {e * 0.9:.6g} · stop {e * 1.1:.6g} · max 48h{tag} · "
                     f"<a href=\"{exchange.trade_url(x['symbol'])}\">chart</a>")
    lines.append("<i>Paper-tracked research setup: about +1%/trade in weak altcoin markets since 2023, "
                 "but it lost through all of 2024. Not advice.</i>")
    return "\n".join(lines)


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
