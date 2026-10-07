# Run Pump Scanner 24/7 on Oracle Cloud (free)

This puts the app on a free Oracle Cloud server in Europe, starts it on boot,
restarts it if it crashes, and lets you open the dashboard privately from your
PC and phone. Telegram alerts work regardless.

Time needed: about 45 minutes, most of it Oracle sign-up.

> Free-tier terms change; check Oracle's current "Always Free" page. Oracle may
> reclaim Always Free servers that sit nearly idle for a week. This app is light,
> so to be safe, upgrade the account to **Pay As You Go** (step 1.4): you are
> still not charged as long as you stay within the Always Free shapes below.

---

## 1. Create the server

1. Sign up at **cloud.oracle.com** → *Start for free*. Choose a **home region in
   Europe** (e.g. *Germany Central (Frankfurt)*). The region can't be changed later,
   and US regions get blocked by Bybit and Binance's main API.
   A card is needed for identity checks; Always Free resources are not charged.
2. In the console: **Compute → Instances → Create instance**.
   - **Image:** Canonical **Ubuntu 24.04**.
   - **Shape:** *Change shape* → **Ampere** → **VM.Standard.A1.Flex**, **2 OCPUs, 12 GB memory**
     (inside the free 4 OCPU / 24 GB allowance).
   - **Networking:** keep the defaults (a public IP is assigned).
   - **SSH keys:** *Generate a key pair for me* → **download the private key** (keep it safe).
   - **Boot volume:** default 50 GB is plenty.
   - Click **Create**. If you get *"Out of capacity"*, try another availability
     domain or try again later. Free ARM capacity is often busy at first.
3. When it shows **Running**, copy its **Public IP address**.
4. (Recommended) **Billing → Upgrade to Pay As You Go**, so the server isn't
   reclaimed for being idle. Stay on the Always Free shapes and it costs nothing.

## 2. Connect to the server

From PowerShell on your PC (replace the key path and IP):

```bash
ssh -i C:\Users\YOU\Downloads\ssh-key.key ubuntu@YOUR.SERVER.IP
```

If Windows complains the key is "too open": right-click the key file →
Properties → Security → Advanced → *Disable inheritance* → remove everyone except
your own user.

All remaining commands run **on the server**.

## 3. Install the app

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install python3-venv python3-pip git gh
sudo timedatectl set-timezone UTC
```

Your repo is private, so log in to GitHub once (choose *GitHub.com → HTTPS →
Login with a web browser*, then enter the code it shows at github.com/login/device):

```bash
gh auth login
gh repo clone sxs47/Crypto-scanner
cd Crypto-scanner
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 4. Download the market data and rebuild the reports (about 20 minutes)

Run these in order (each one uses the previous one's data):

```bash
.venv/bin/python scanner.py --days 90
.venv/bin/python bybit.py --download
.venv/bin/python trend.py --download
.venv/bin/python futures.py --download
.venv/bin/python setups.py
.venv/bin/python setups.py --bybit
.venv/bin/python signals.py
```

The last three rebuild the backtests and the Signals model on the server's own
data (a model file saved on another computer may not load with different
library versions).

## 5. Password and Telegram (on the server)

```bash
.venv/bin/python app.py --set-password
.venv/bin/python app.py --setup-telegram
```

For Telegram, paste the **same bot token** as on your PC, then send any message to
your bot so it finds your chat.

**Stop the app on your PC afterwards** (or turn Telegram off there in the Alerts
tab), otherwise you'll get every alert twice.

## 6. Start it automatically, forever

```bash
sudo tee /etc/systemd/system/pump-scanner.service >/dev/null <<'EOF'
[Unit]
Description=Pump Scanner dashboard and live scanner
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Crypto-scanner
ExecStart=/home/ubuntu/Crypto-scanner/.venv/bin/python app.py --host 127.0.0.1 --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now pump-scanner
systemctl status pump-scanner --no-pager
```

It now starts on boot and restarts within 10 seconds if it ever crashes.
See its log any time with:

```bash
journalctl -u pump-scanner -f
```

## 7. Open the dashboard from your PC and phone (privately)

The app only listens on the server itself. The simplest safe way to reach it is
**Tailscale** (free for personal use): a private network between your own devices,
with HTTPS, and nothing opened to the internet.

On the server:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Open the link it prints and sign in (Google/Microsoft/GitHub account). Then:

```bash
sudo tailscale serve --bg 5000
```

It prints an address like `https://instance-name.tail1234.ts.net`.

Install **Tailscale** on your PC and phone (app stores / tailscale.com/download),
sign in with the same account, and open that address. You'll see the login page;
use the password from step 5.

> Don't open port 5000 in Oracle's firewall. With Tailscale you never need to.
> If you'd rather have a public web address, ask for a guide with a free domain
> and automatic HTTPS (Caddy). Never expose the app over plain HTTP.

## Updating later

After pushing new code to GitHub from your PC:

```bash
cd ~/Crypto-scanner && git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart pump-scanner
```

## Useful commands

| What | Command |
|---|---|
| Is it running? | `systemctl status pump-scanner --no-pager` |
| Live log | `journalctl -u pump-scanner -f` |
| Restart | `sudo systemctl restart pump-scanner` |
| Stop | `sudo systemctl stop pump-scanner` |
| Change password | `.venv/bin/python app.py --set-password` then restart |
| Disk use | `du -sh ~/Crypto-scanner/data` |
