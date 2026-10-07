"""
Password login for the dashboard.

The password is stored only as a salted hash in data/auth.json (not in git).
When no password is set, the app runs open — fine on 127.0.0.1, refused on a
public address (see app.py).

    python app.py --set-password
"""

import getpass
import hashlib
import json
import secrets
import threading
import time

from werkzeug.security import check_password_hash, generate_password_hash

import scanner

AUTH_PATH = scanner.DATA_DIR / "auth.json"
SECRET_PATH = scanner.DATA_DIR / "secret_key"
MAX_FAILS = 5          # per IP, then a lockout
LOCKOUT_S = 15 * 60

_fails = {}            # ip -> (count, first_fail_time)
_fail_lock = threading.Lock()


def _load():
    try:
        return json.loads(AUTH_PATH.read_text())
    except (OSError, ValueError):
        return {}


def enabled():
    return bool(_load().get("hash"))


def session_token():
    """Changes whenever the password changes, which signs out every old session."""
    h = _load().get("hash", "")
    return hashlib.sha256(h.encode()).hexdigest()[:32]


def secret_key():
    scanner.DATA_DIR.mkdir(exist_ok=True)
    if not SECRET_PATH.exists():
        SECRET_PATH.write_text(secrets.token_hex(32))
    return SECRET_PATH.read_text().strip()


def locked_out(ip):
    with _fail_lock:
        n, since = _fails.get(ip, (0, 0))
        if n >= MAX_FAILS and time.time() - since < LOCKOUT_S:
            return int(LOCKOUT_S - (time.time() - since))
        if time.time() - since >= LOCKOUT_S:
            _fails.pop(ip, None)
    return 0


def check(ip, password):
    ok = check_password_hash(_load().get("hash", ""), password or "")
    with _fail_lock:
        if ok:
            _fails.pop(ip, None)
        else:
            n, since = _fails.get(ip, (0, time.time()))
            _fails[ip] = (n + 1, since)
    if not ok:
        time.sleep(1)  # slow down guessing
    return ok


def set_password_cli():
    print("Set the dashboard password (at least 8 characters).")
    while True:
        pw = getpass.getpass("New password: ")
        if len(pw) < 8:
            print("Too short — use at least 8 characters.")
            continue
        if getpass.getpass("Repeat password: ") != pw:
            print("Passwords don't match, try again.")
            continue
        break
    scanner.DATA_DIR.mkdir(exist_ok=True)
    AUTH_PATH.write_text(json.dumps({"hash": generate_password_hash(pw), "set_at": int(time.time())}))
    print(f"Password saved (hashed) to {AUTH_PATH}. Everyone signed in before is now signed out.")


LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pump Scanner · Sign in</title>
<style>
:root { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --text:#0b0b0b; --text-2:#52514e; --border:rgba(11,11,11,.12); --accent:#2a78d6; --bad:#b3261e; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --text:#fff; --text-2:#c3c2b7; --border:rgba(255,255,255,.12); --accent:#3987e5; --bad:#ff8a80; } }
* { box-sizing: border-box; }
body { margin:0; min-height:100vh; display:grid; place-items:center; background:var(--page); color:var(--text);
       font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; padding:16px; }
form { width:100%; max-width:340px; background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; }
h1 { font-size:18px; margin:0 0 4px; display:flex; align-items:center; gap:8px; }
p { color:var(--text-2); margin:0 0 18px; }
label { display:block; font-size:13px; color:var(--text-2); margin-bottom:6px; }
input { width:100%; height:38px; padding:0 10px; border-radius:8px; border:1px solid var(--border); background:var(--page); color:var(--text); font:inherit; }
input:focus { outline:2px solid var(--accent); outline-offset:1px; }
button { width:100%; height:38px; margin-top:14px; border:0; border-radius:8px; background:var(--accent); color:#fff; font:inherit; font-weight:600; cursor:pointer; }
.err { color:var(--bad); font-size:13px; margin-top:10px; }
</style></head>
<body><form method="post">
<h1><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 19 L9 12 L13 15 L21 5"/><path d="M15 5 H21 V11"/></svg>Pump Scanner</h1>
<p>Sign in to continue.</p>
<label for="pw">Password</label>
<input id="pw" name="password" type="password" autocomplete="current-password" autofocus required>
<input type="hidden" name="next" value="{next}">
<button type="submit">Sign in</button>
{error}
</form></body></html>"""
