# DOO4730 — live detection dashboard (Streamlit)

The two-tier tool-failure detector running on the **live InfluxDB feed**, as a
browser dashboard you can open from anywhere. Deploy free on **Streamlit Community Cloud**.

## Deploy (one-time, ~3 minutes)
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub** (or "from existing repo").
3. Select:
   - **Repository:** `Siddharth6121/doo4730-live-dashboard`
   - **Branch:** `main`
   - **Main file path:** `live_sensor_monitor.py`
4. Open **Advanced settings → Secrets** and paste your InfluxDB credentials
   (copy the values from your local `.env`):
   ```toml
   INFLUX_HOST = "your-influx-host"
   INFLUX_TOKEN = "your-influx-token"
   INFLUX_DATABASE = "your-database"
   ```
5. Click **Deploy**. You'll get a public URL like
   `https://doo4730-live-dashboard.streamlit.app` — open it on any device to watch live.

## What you'll see
The same live per-second graph as localhost: spindle current, the self-set 89 A alarm
line, Tier-1 surge dots (amber = normal cutting / watch) and Tier-2 red-star confirmed
failures (surge + machine stoppage), with a live machine-state read.

## Notes
- The app **sleeps when nobody is viewing it** and wakes when you open the URL — it is a
  *watch-live* tool, not an unattended catcher. Unattended 24/7 catching + email alerts
  are handled by the companion GitHub Actions monitor (`doo4730-monitor`).
- Secrets live in Streamlit's secret store, never in the code.
