"""
Trade journal: your own closed trades, stored privately in data/journal.json (not in git).

Trades come from a Bybit "Closed P&L" CSV export (mapped in the browser) or are
added by hand. Each trade is a dict:
  time (UTC seconds, when it closed), symbol, side ("long"/"short"), qty, entry, exit,
  pnl (USDT, after fees as Bybit reports it), leverage (optional), fees (optional),
  tag (your setup name, optional), note (optional), source ("import"/"manual")
"""

import hashlib
import json
import threading
import time

import scanner

PATH = scanner.DATA_DIR / "journal.json"
_lock = threading.Lock()
FIELDS = ("time", "symbol", "side", "qty", "entry", "exit", "pnl", "leverage", "fees", "tag", "note", "source")


def _load():
    try:
        return json.loads(PATH.read_text())
    except (OSError, ValueError):
        return []


def _save(trades):
    scanner.DATA_DIR.mkdir(exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(trades, indent=1))
    tmp.replace(PATH)  # atomic: never leaves a half-written journal


def _clean(t):
    """Validate one trade; raises ValueError with a readable message."""
    out = {k: t.get(k) for k in FIELDS}
    try:
        out["time"] = int(float(out["time"]))
        out["pnl"] = float(out["pnl"])
        for k in ("qty", "entry", "exit", "leverage", "fees"):
            out[k] = None if out[k] in (None, "") else float(out[k])
    except (TypeError, ValueError):
        raise ValueError("time and P&L must be numbers")
    out["symbol"] = str(out["symbol"] or "").upper().strip()
    out["side"] = str(out["side"] or "").lower()
    if not out["symbol"] or out["side"] not in ("long", "short"):
        raise ValueError("each trade needs a symbol and a side (long/short)")
    out["tag"] = (out["tag"] or "").strip()[:40]
    out["note"] = (out["note"] or "").strip()[:500]
    out["source"] = out["source"] or "manual"
    key = f"{out['time']}|{out['symbol']}|{out['side']}|{out['qty']}|{out['entry']}|{out['exit']}|{out['pnl']:.6f}"
    out["id"] = hashlib.sha1(key.encode()).hexdigest()[:12]
    return out


def all_trades():
    with _lock:
        return sorted(_load(), key=lambda t: t["time"])


def add(trades):
    """Add trades, skipping exact duplicates (re-importing the same CSV is safe). Returns (added, skipped)."""
    cleaned = [_clean(t) for t in trades]
    with _lock:
        cur = _load()
        have = {t["id"] for t in cur}
        new = [t for t in cleaned if t["id"] not in have]
        _save(cur + new)
    return len(new), len(cleaned) - len(new)


def update(trade_id, fields):
    """Edit tag/note of one trade."""
    with _lock:
        cur = _load()
        for t in cur:
            if t["id"] == trade_id:
                for k in ("tag", "note"):
                    if k in fields:
                        t[k] = str(fields[k] or "").strip()[: 40 if k == "tag" else 500]
                _save(cur)
                return t
    return None


def delete(trade_id=None):
    """Delete one trade, or all of them when trade_id is None."""
    with _lock:
        cur = _load()
        keep = [] if trade_id is None else [t for t in cur if t["id"] != trade_id]
        _save(keep)
        return len(cur) - len(keep)
