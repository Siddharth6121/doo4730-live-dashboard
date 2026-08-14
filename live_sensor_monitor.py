"""
DOO4730 — LIVE Sensor Detection Monitor  (two-tier)
----------------------------------------------------
Run:  streamlit run live_sensor_monitor.py --server.port 8503

100% LIVE from InfluxDB. Two-tier detection:
  • TIER 1  surge flag  ....... current jumps over the self-set alarm level (normal
                                cutting does this ~149x/day -> a WATCH item, not an alarm)
  • TIER 2  CONFIRMED FAILURE . a surge that coincides with the machine STOPPING
                                (INTERRUPTED / DISCONNECTED within 30s) OR the sensor
                                feed going silent (heartbeat). This is ~0.5x/day.
Only Tier 2 raises a loud alert.
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="DOO4730 LIVE Detection (two-tier)", layout="wide", page_icon="🟢")

# Bridge Streamlit Cloud secrets -> environment so influx_utils.get_client() picks them up.
import os as _os
for _k in ("INFLUX_HOST", "INFLUX_TOKEN", "INFLUX_DATABASE"):
    try:
        if _k in st.secrets and _k not in _os.environ:
            _os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

CUR = ["spindle_current_leg1", "spindle_current_leg2", "spindle_current_leg3"]
ALARM = 89.4
HEARTBEAT_GAP_S = 3.0
CONFIRM_S = 30            # a surge is CONFIRMED if a stoppage occurs within this many seconds
WINDOW_S = 120
ABNORMAL = ["INTERRUPTED", "DISCONNECTED"]


@st.cache_resource(show_spinner=False)
def get_influx():
    try:
        from influx_utils import get_client
        c = get_client()
        c.query("SHOW TABLES", language="sql")
        return c
    except Exception:
        return None


def fetch_sensor(client, minutes):
    q = (f"SELECT time, {','.join(CUR)} FROM sensor_telemetry WHERE device_id='DOO4730' AND tenant_id=2 "
         f"AND time > now() - INTERVAL '{minutes} minutes' ORDER BY time ASC")
    df = client.query(q, language="sql").to_pandas()
    if not len(df):
        return df
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)
    df["max_current"] = df[CUR].max(axis=1)
    df["surge"] = df["max_current"] - df["max_current"].shift(2)
    df["next_gap"] = (df["time"].shift(-1) - df["time"]).dt.total_seconds()
    df["flag_surge"] = df["surge"] > ALARM
    df["flag_hb"] = df["next_gap"] > HEARTBEAT_GAP_S
    return df


def fetch_status(client, minutes):
    q = (f"SELECT time, run_status FROM telemetry_raw WHERE device_id='DOO4730' AND tenant_id=2 "
         f"AND time > now() - INTERVAL '{minutes} minutes' ORDER BY time ASC")
    df = client.query(q, language="sql").to_pandas()
    if len(df):
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df


@st.cache_data(ttl=120, show_spinner=False)
def today_summary(_client):
    """Cheap daily rollup since 00:00 UTC. Uses the sparse status stream + tiny sensor checks
    (never a full-day 1 Hz pull, which would hit the file-scan limit). Cached for 2 minutes."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    mins = int((now - now.normalize()).total_seconds() // 60) + 2   # minutes since 00:00 UTC
    out = {"failures": 0, "last": None, "stoppages": 0, "parts": 0}
    try:
        sta = _client.query(
            f"SELECT time, run_status, part_count FROM telemetry_raw WHERE device_id='DOO4730' AND tenant_id=2 "
            f"AND time > now() - INTERVAL '{mins} minutes' ORDER BY time ASC", language="sql").to_pandas()
    except Exception:
        return out
    if not len(sta):
        return out
    sta["time"] = pd.to_datetime(sta["time"]).dt.tz_localize(None)
    abn = sta[sta["run_status"].isin(ABNORMAL)]
    out["stoppages"] = int(len(abn))
    pc = sta["part_count"].dropna()
    out["parts"] = int(pc.max() - pc.min()) if len(pc) > 1 else 0
    fails = []
    for t in abn["time"]:                                            # confirm each stoppage with a tiny sensor window
        a = (t - pd.Timedelta(seconds=40)).strftime("%Y-%m-%d %H:%M:%S")
        b = (t + pd.Timedelta(seconds=35)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            w = _client.query(
                f"SELECT time,{','.join(CUR)} FROM sensor_telemetry WHERE device_id='DOO4730' AND tenant_id=2 "
                f"AND time > TIMESTAMP '{a}' AND time < TIMESTAMP '{b}' ORDER BY time ASC",
                language="sql").to_pandas()
        except Exception:
            continue
        if len(w) > 2:
            mc = w[CUR].max(axis=1)
            if ((mc - mc.shift(2)) > ALARM).any():
                fails.append(t)
    out["failures"] = len(fails)
    out["last"] = str(max(fails))[:19] if fails else None
    return out


client = get_influx()
ss = st.session_state
ss.setdefault("running", True)

st.sidebar.header("Controls")
ss.running = st.sidebar.toggle("▶  Live (auto-refresh)", value=ss.running)
lookback = st.sidebar.slider("Fetch look-back (minutes)", 5, 60, 20)
st.sidebar.markdown("---")
st.sidebar.caption("Two-tier detector. Tier 1 = current surge (watch, ~149/day). "
                   "Tier 2 = surge **+ machine stoppage** or sensor dropout (real failure, ~0.5/day). "
                   "Only Tier 2 alarms.")

st.title("🟢  DOO4730 — LIVE Detection Monitor  ·  two-tier")

if client is None:
    st.error("InfluxDB offline."); st.stop()

sen = fetch_sensor(client, lookback)
sta = fetch_status(client, lookback)
now_utc = pd.Timestamp.utcnow().tz_localize(None)

if sen is None or not len(sen):
    st.error("No live sensor data returned yet — retrying…")
    if ss.running: time.sleep(2); st.rerun()
    st.stop()

# ---- abnormal (stoppage) status times in the window ----
abn_times = []
if sta is not None and len(sta):
    abn_times = list(sta[sta["run_status"].isin(ABNORMAL)]["time"])

def confirmed(t):
    """A surge at time t is a CONFIRMED failure if a stoppage occurs within CONFIRM_S after (or ~10s before)."""
    return any((et >= t - pd.Timedelta(seconds=10)) and (et <= t + pd.Timedelta(seconds=CONFIRM_S)) for et in abn_times)

surges = sen[sen["flag_surge"]].copy()
surges["confirmed"] = surges["time"].apply(confirmed) if len(surges) else []
n_watch = int((~surges["confirmed"]).sum()) if len(surges) else 0
n_conf = int(surges["confirmed"].sum()) if len(surges) else 0
hb_now = bool(sen["flag_hb"].iloc[-1])

t_end = sen["time"].max(); end_row = sen.iloc[-1]
age = (now_utc - t_end).total_seconds()
# is there a confirmed catch (or heartbeat) in the last CONFIRM_S seconds?
recent_conf = len(surges) and (surges[surges["confirmed"]]["time"] > t_end - pd.Timedelta(seconds=CONFIRM_S)).any()
recent_abn = any(et > t_end - pd.Timedelta(seconds=CONFIRM_S) for et in abn_times)
state_now = "—"
if sta is not None and len(sta):
    s_ok = sta.dropna(subset=["run_status"])
    if len(s_ok): state_now = str(s_ok["run_status"].iloc[-1])

# ---- metrics ----
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("InfluxDB", "🟢 Connected")
c2.metric("Live current", f"{end_row['max_current']:.0f} A")
c3.metric("Machine state", state_now)
c4.metric("Tier-1 surges (watch)", n_watch)
c5.metric("Tier-2 CONFIRMED", n_conf)
st.caption(f"🕒 LIVE · latest sensor reading **{str(t_end)[:19]} UTC** · now {str(now_utc)[:19]} UTC · "
           f"sampling ≈ {sen['time'].diff().dt.total_seconds().median():.2f}s · look-back {lookback} min")

# ---- today (since 00:00 UTC) rollup ----
d = today_summary(client)
st.markdown("**📅 Today — since 00:00 UTC** (full-day record)")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Confirmed failures today", d["failures"])
d2.metric("Last failure", d["last"] or "—")
d3.metric("Machine stoppages today", d["stoppages"])
d4.metric("Parts made today", d["parts"])
st.caption("Live row above = last 20 min (moving window) · Today row = cumulative since midnight UTC, "
           "refreshed every ~2 min.")

# ---- banner ----
if recent_conf or (recent_abn and (sen['flag_surge'].tail(30).any())) or (hb_now and recent_abn):
    st.error("🚨  CONFIRMED FAILURE — current surge coincided with a machine stoppage")
elif recent_abn:
    st.warning(f"⚠️  Machine stoppage ({state_now}) — watching for a coincident surge")
elif sen["flag_surge"].tail(15).any():
    st.info("● Surge detected — normal cutting (Tier-1 watch, not an alarm)")
else:
    st.success("✓  Normal operation")

# ---- chart ----
w = sen[sen["time"] >= t_end - pd.Timedelta(seconds=WINDOW_S)]
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(w["time"], w["max_current"], color="#2f8fb3", lw=1.6, label="spindle current")
ax.axhline(ALARM, color="#c98a1a", ls="--", lw=1.3, label=f"alarm level ({ALARM:.0f} A)")
# tier 1 watch dots
ws = w[w["flag_surge"]]
if len(surges):
    wsm = ws.merge(surges[["time", "confirmed"]], on="time", how="left")
    watch = wsm[wsm["confirmed"] != True]; conf = wsm[wsm["confirmed"] == True]
    ax.scatter(watch["time"], watch["max_current"], color="#e0a030", s=26, zorder=4,
               label="Tier-1 surge (watch)")
    ax.scatter(conf["time"], conf["max_current"], color="#cf4040", s=90, marker="*", zorder=6,
               label="Tier-2 CONFIRMED failure")
# stoppage lines
for et in abn_times:
    if len(w) and w["time"].min() <= et <= w["time"].max():
        ax.axvline(et, color="#cf4040", lw=1.8, alpha=.55)
        ax.text(et, ax.get_ylim()[1]*0.9, " stoppage", color="#cf4040", fontsize=8, rotation=90, va="top")
ax.set_ylim(0, max(160, w["max_current"].max() * 1.1))
ax.set_ylabel("current (A)"); ax.set_xlabel("time (UTC)")
ax.set_title(f"Two-tier detector LIVE — {str(t_end)[:19]} UTC", fontsize=11)
ax.legend(loc="upper left", fontsize=8.5); ax.grid(alpha=.2)
st.pyplot(fig); plt.close(fig)

with st.expander("How the two tiers work"):
    st.markdown(
        f"- **Amber dots (Tier 1)** — the current surged over {ALARM:.0f} A. Normal cutting does this "
        f"~149×/day, so on its own it is only a **watch** item, not an alarm.\n"
        f"- **Red star (Tier 2)** — a surge that **coincided with the machine stopping** "
        f"(INTERRUPTED/DISCONNECTED within {CONFIRM_S}s) or the sensor feed going silent. "
        f"This is the real failure signal (~0.5×/day) and is the only thing that raises the loud alert.\n"
        f"- **Red vertical line** — a live machine stoppage from the status stream.\n"
        f"- This is how a real catch is told apart from the ~150 daily false surges: the stoppage confirms it."
    )

if ss.running:
    time.sleep(2)
    st.rerun()
