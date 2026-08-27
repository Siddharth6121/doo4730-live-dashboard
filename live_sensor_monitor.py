"""
Live two-tier detection dashboard for DOO4730 (Streamlit + Plotly).

    streamlit run live_sensor_monitor.py

Reads directly from InfluxDB and applies the two-tier rule:
  Tier 1  a current surge over the alarm level. Normal cutting produces these
          frequently, so on its own a surge is a watch item, not an alarm.
  Tier 2  a surge that coincides with the machine stopping (INTERRUPTED/DISCONNECTED
          within 30s) or the sensor feed going silent. Only Tier 2 raises an alarm.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DOO4730 LIVE Detection", layout="wide", page_icon="🟢")

# ---- brand / styling ----
TEAL = "#2F8FB3"; DARK = "#161C22"; AMBER = "#e0a030"; RED = "#cf4040"; GREEN = "#2ca048"
st.markdown("""
<style>
:root { --teal:#2F8FB3; }
.block-container { padding-top: 1.4rem; }
h1, h2, h3 { letter-spacing:.2px; }
div[data-testid="stMetric"]{
  background: linear-gradient(180deg,#ffffff,#f5f9fb);
  border:1px solid #e3edf2; border-left:4px solid var(--teal);
  border-radius:12px; padding:12px 16px; box-shadow:0 1px 3px rgba(20,28,34,.05);
}
div[data-testid="stMetricLabel"] p{ color:#54606A; font-weight:600; font-size:.82rem; }
div[data-testid="stMetricValue"]{ color:#161C22; font-weight:700; }
.hero{ background:linear-gradient(90deg,#2F8FB3,#246b86); color:#fff;
  border-radius:14px; padding:16px 22px; margin-bottom:14px; }
.hero h1{ margin:0; font-size:1.5rem; color:#fff; }
.hero p{ margin:2px 0 0; color:#dbeef5; font-size:.9rem; }
.stAlert{ border-radius:12px; }
</style>
""", unsafe_allow_html=True)

# Bridge Streamlit Cloud secrets into the environment so influx_utils.get_client() finds them.
for _k in ("INFLUX_HOST", "INFLUX_TOKEN", "INFLUX_DATABASE"):
    try:
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

CUR = ["spindle_current_leg1", "spindle_current_leg2", "spindle_current_leg3"]
PK = ["vib_x_pkpk_accel", "vib_y_pkpk_accel", "vib_z_pkpk_accel"]
ALARM = 89.4
HEARTBEAT_GAP_S = 3.0
CONFIRM_S = 30            # a surge is confirmed if a stoppage occurs within this many seconds
WINDOW_S = 150
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
    q = (f"SELECT time, {','.join(CUR + PK)} FROM sensor_telemetry WHERE device_id='DOO4730' AND tenant_id=2 "
         f"AND time > now() - INTERVAL '{minutes} minutes' ORDER BY time ASC")
    try:
        df = client.query(q, language="sql").to_pandas()
    except Exception:
        return None   # transient query error (e.g. file-scan limit); caller retries
    if not len(df):
        return df
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)
    df["max_current"] = df[CUR].max(axis=1)
    df["vib"] = df[PK].max(axis=1)
    df["mc_smooth"] = df["max_current"].rolling(9, center=True, min_periods=1).median()
    df["vib_smooth"] = df["vib"].rolling(9, center=True, min_periods=1).median()
    df["surge"] = df["max_current"] - df["max_current"].shift(2)
    df["next_gap"] = (df["time"].shift(-1) - df["time"]).dt.total_seconds()
    df["flag_surge"] = df["surge"] > ALARM
    df["flag_hb"] = df["next_gap"] > HEARTBEAT_GAP_S
    return df


def fetch_status(client, minutes):
    q = (f"SELECT time, run_status FROM telemetry_raw WHERE device_id='DOO4730' AND tenant_id=2 "
         f"AND time > now() - INTERVAL '{minutes} minutes' ORDER BY time ASC")
    try:
        df = client.query(q, language="sql").to_pandas()
    except Exception:
        return None
    if df is not None and len(df):
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df


@st.cache_data(ttl=120, show_spinner=False)
def today_summary(_client):
    """Daily rollup since 00:00 UTC. Uses the sparse status stream plus small sensor checks
    (never a full-day 1 Hz pull, which would hit the file-scan limit). Cached for 2 minutes."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    mins = int((now - now.normalize()).total_seconds() // 60) + 2
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
    for t in abn["time"]:
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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cycle_profile(_client, s_iso, e_iso, N=90):
    """Normalized (0-100% of cycle), lightly smoothed spindle-current profile for one cycle."""
    try:
        d = _client.query(
            f"SELECT time,{','.join(CUR)} FROM sensor_telemetry WHERE device_id='DOO4730' AND tenant_id=2 "
            f"AND time > TIMESTAMP '{s_iso}' AND time < TIMESTAMP '{e_iso}' ORDER BY time ASC",
            language="sql").to_pandas()
    except Exception:
        return None
    if len(d) < 15:
        return None
    d["time"] = pd.to_datetime(d["time"]).dt.tz_localize(None)
    mc = d[CUR].max(axis=1).values
    t = (d["time"] - d["time"].iloc[0]).dt.total_seconds().values
    if t[-1] <= 0:
        return None
    prof = np.interp(np.linspace(0, 1, N), t / t[-1], mc)
    k = 7
    prof = np.convolve(np.pad(prof, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    return prof.tolist()


@st.cache_data(ttl=90, show_spinner=False)
def todays_worm(_client, cap=45):
    """Overlay of today's completed cycles: normal band + median, plus each failure cycle in red."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    mins = int((now - now.normalize()).total_seconds() // 60) + 2
    try:
        sta = _client.query(
            f"SELECT time, run_status, part_count FROM telemetry_raw WHERE device_id='DOO4730' AND tenant_id=2 "
            f"AND time > now() - INTERVAL '{mins} minutes' ORDER BY time ASC", language="sql").to_pandas()
    except Exception:
        return None
    if not len(sta):
        return None
    sta["time"] = pd.to_datetime(sta["time"]).dt.tz_localize(None)
    inc = sta[sta["part_count"].notna()]
    inc = inc[inc["part_count"].diff().fillna(1) > 0]["time"].tolist()
    cyc = [(inc[i - 1], inc[i]) for i in range(1, len(inc)) if 250 <= (inc[i] - inc[i - 1]).total_seconds() <= 900]
    durs = [(e - s).total_seconds() for s, e in cyc]
    mdur = float(np.median(durs)) if durs else 431.0
    # failures today = a stoppage confirmed by a nearby current surge
    abn = sta[sta["run_status"].isin(ABNORMAL)]["time"].tolist()
    fails = []
    for tt in abn:
        a = (tt - pd.Timedelta(seconds=40)).strftime("%Y-%m-%d %H:%M:%S")
        b = (tt + pd.Timedelta(seconds=35)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            wv = _client.query(
                f"SELECT time,{','.join(CUR)} FROM sensor_telemetry WHERE device_id='DOO4730' AND tenant_id=2 "
                f"AND time > TIMESTAMP '{a}' AND time < TIMESTAMP '{b}' ORDER BY time ASC", language="sql").to_pandas()
        except Exception:
            continue
        if len(wv) > 2:
            mc = wv[CUR].max(axis=1)
            if ((mc - mc.shift(2)) > ALARM).any():
                fails.append((tt, float(mc.max())))
    ftimes = [t for t, _ in fails]
    normals = []
    for (s, e) in cyc[-cap:]:
        if any(s <= ft <= e for ft in ftimes):
            continue
        p = cycle_profile(_client, s.strftime("%Y-%m-%d %H:%M:%S"), e.strftime("%Y-%m-%d %H:%M:%S"))
        if p is not None:
            normals.append(p)
    failprofs = []
    for ft, _pk in fails:
        before = [t for t in inc if t <= ft]
        s0 = before[-1] if before else ft - pd.Timedelta(seconds=mdur * 0.7)
        e0 = s0 + pd.Timedelta(seconds=mdur)
        p = cycle_profile(_client, s0.strftime("%Y-%m-%d %H:%M:%S"), e0.strftime("%Y-%m-%d %H:%M:%S"))
        if p is not None:
            failprofs.append((str(ft)[:19], p))
    return {"normals": normals, "failprofs": failprofs, "fails": fails, "ncyc": len(cyc)}


client = get_influx()
ss = st.session_state
ss.setdefault("running", True)

st.sidebar.header("Controls")
ss.running = st.sidebar.toggle("▶  Live (auto-refresh)", value=ss.running)
lookback = st.sidebar.slider("Fetch look-back (minutes)", 5, 60, 10)
st.sidebar.markdown("---")
st.sidebar.caption("Two-tier detector. Tier 1 = current surge (watch, ~149/day). "
                   "Tier 2 = surge **+ machine stoppage** or sensor dropout (real failure, ~0.5/day). "
                   "Only Tier 2 alarms.")

st.markdown('<div class="hero"><h1>🟢 DOO4730 — Live Tool-Failure Detection</h1>'
            '<p>Real-time two-tier detector · current surge confirmed by a machine stoppage</p></div>',
            unsafe_allow_html=True)

if client is None:
    st.error("InfluxDB offline."); st.stop()

sen = fetch_sensor(client, lookback)
sta = fetch_status(client, lookback)
now_utc = pd.Timestamp.utcnow().tz_localize(None)

if sen is None or not len(sen):
    st.error("No live sensor data returned yet — retrying…")
    if ss.running: time.sleep(2); st.rerun()
    st.stop()

abn_times = []
if sta is not None and len(sta):
    abn_times = list(sta[sta["run_status"].isin(ABNORMAL)]["time"])

def confirmed(t):
    return any((et >= t - pd.Timedelta(seconds=10)) and (et <= t + pd.Timedelta(seconds=CONFIRM_S)) for et in abn_times)

surges = sen[sen["flag_surge"]].copy()
surges["confirmed"] = surges["time"].apply(confirmed) if len(surges) else []
n_watch = int((~surges["confirmed"]).sum()) if len(surges) else 0
n_conf = int(surges["confirmed"].sum()) if len(surges) else 0
hb_now = bool(sen["flag_hb"].iloc[-1])

t_end = sen["time"].max(); end_row = sen.iloc[-1]
age = (now_utc - t_end).total_seconds()
recent_conf = len(surges) and (surges[surges["confirmed"]]["time"] > t_end - pd.Timedelta(seconds=CONFIRM_S)).any()
recent_abn = any(et > t_end - pd.Timedelta(seconds=CONFIRM_S) for et in abn_times)
state_now = "—"
if sta is not None and len(sta):
    s_ok = sta.dropna(subset=["run_status"])
    if len(s_ok): state_now = str(s_ok["run_status"].iloc[-1])

# Live metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("InfluxDB", "🟢 Connected")
c2.metric("Live current", f"{end_row['max_current']:.0f} A")
c3.metric("Machine state", state_now)
c4.metric("Tier-1 surges (watch)", n_watch)
c5.metric("Tier-2 CONFIRMED", n_conf)
st.caption(f"🕒 LIVE · latest sensor reading **{str(t_end)[:19]} UTC** · now {str(now_utc)[:19]} UTC · "
           f"sampling ≈ {sen['time'].diff().dt.total_seconds().median():.2f}s · look-back {lookback} min")

# Today's rollup
d = today_summary(client)
st.markdown("**📅 Today — since 00:00 UTC** (full-day record)")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Confirmed failures today", d["failures"])
d2.metric("Last failure", d["last"] or "—")
d3.metric("Machine stoppages today", d["stoppages"])
d4.metric("Parts made today", d["parts"])
st.caption("Live row above = last 20 min (moving window) · Today row = cumulative since midnight UTC, "
           "refreshed every ~2 min.")

# Status banner
if recent_conf or (recent_abn and (sen['flag_surge'].tail(30).any())) or (hb_now and recent_abn):
    st.error("🚨  CONFIRMED FAILURE — current surge coincided with a machine stoppage")
elif recent_abn:
    st.warning(f"⚠️  Machine stoppage ({state_now}) — watching for a coincident surge")
elif sen["flag_surge"].tail(15).any():
    st.info("● Surge detected — normal cutting (Tier-1 watch, not an alarm)")
else:
    st.success("✓  Normal operation")

tab_live, tab_worm = st.tabs(["📈  Live signal", "🔁  Today's cycles (worm)"])

with tab_live:
    # ---- interactive Plotly chart ----
    w = sen[sen["time"] >= t_end - pd.Timedelta(seconds=WINDOW_S)]
    ws = w[w["flag_surge"]]
    watch = conf = ws.iloc[0:0]
    if len(surges) and len(ws):
        wsm = ws.merge(surges[["time", "confirmed"]], on="time", how="left")
        watch = wsm[wsm["confirmed"] != True]; conf = wsm[wsm["confirmed"] == True]

    ymax = max(160.0, float(w["max_current"].max()) * 1.12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w["time"], y=w["max_current"], mode="lines",
                  line=dict(color=TEAL, width=1.8),
                  fill="tozeroy",
                  fillgradient=dict(type="vertical",
                      colorscale=[[0.0, "rgba(47,143,179,0.03)"], [1.0, "rgba(47,143,179,0.28)"]]),
                  name="Spindle current",
                  hovertemplate="%{x|%H:%M:%S}<br>%{y:.0f} A<extra></extra>"))
    fig.add_trace(go.Scatter(x=w["time"], y=w["vib"], mode="lines",
                  line=dict(color=GREEN, width=1.4, dash="dot"),
                  name="Vibration", yaxis="y2",
                  hovertemplate="%{x|%H:%M:%S}<br>vib %{y:.1f}<extra></extra>"))
    fig.add_hline(y=ALARM, line_dash="dash", line_color=AMBER, line_width=1.4,
                  annotation_text=f"alarm {ALARM:.0f} A", annotation_position="top left",
                  annotation_font_color=AMBER)
    if len(watch):
        fig.add_trace(go.Scatter(x=watch["time"], y=watch["max_current"], mode="markers",
                      marker=dict(color=AMBER, size=9, line=dict(color="#fff", width=1)),
                      name="Tier-1 surge (watch)",
                      hovertemplate="watch surge<br>%{x|%H:%M:%S} · %{y:.0f} A<extra></extra>"))
    if len(conf):
        fig.add_trace(go.Scatter(x=conf["time"], y=conf["max_current"], mode="markers",
                      marker=dict(color=RED, size=17, symbol="star", line=dict(color="#fff", width=1)),
                      name="Tier-2 CONFIRMED failure",
                      hovertemplate="CONFIRMED failure<br>%{x|%H:%M:%S} · %{y:.0f} A<extra></extra>"))
    for et in abn_times:
        if len(w) and w["time"].min() <= et <= w["time"].max():
            fig.add_vline(x=et, line_color=RED, line_width=1.6, opacity=0.5)
            fig.add_annotation(x=et, y=ymax, text="stoppage", showarrow=False,
                               font=dict(color=RED, size=10), textangle=-90, yanchor="top", xshift=-7)
    fig.update_layout(
        template="plotly_white", height=440, margin=dict(t=44, b=10, l=10, r=10),
        title=dict(text=f"Two-tier detector — live · {str(t_end)[:19]} UTC", font=dict(size=15, color=DARK)),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="time (UTC)", showgrid=True, gridcolor="#eef3f5"),
        yaxis=dict(title="current (A)", range=[0, ymax], showgrid=True, gridcolor="#eef3f5"),
        yaxis2=dict(title="vibration (pk-pk)", overlaying="y", side="right", range=[0, 35], showgrid=False),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, key="livechart")

    with st.expander("How the two tiers work"):
        st.markdown(
            f"- **Amber dots (Tier 1)** — the current surged over {ALARM:.0f} A. Normal cutting does this "
            f"~149×/day, so on its own it is only a **watch** item, not an alarm.\n"
            f"- **Red star (Tier 2)** — a surge that **coincided with the machine stopping** "
            f"(INTERRUPTED/DISCONNECTED within {CONFIRM_S}s) or the sensor feed going silent. "
            f"This is the real failure signal (~0.5×/day) and is the only thing that raises the loud alert.\n"
            f"- **Teal line (current)** — the real, instantaneous spindle current; the sharp peaks are "
            f"genuine surges, and a surge over {ALARM:.0f} A is exactly the signal we detect.\n"
            f"- **Red vertical line** — a live machine stoppage from the status stream.\n"
            f"- **Green dotted line (vibration)** — shown for context on the right axis; click it in the legend "
            f"to hide. It stays low/erratic and is not used to trigger — current is the trigger.\n"
            f"- This is how a real catch is told apart from the ~150 daily false surges: the stoppage confirms it."
        )

with tab_worm:
    st.caption("Every completed part-cycle today is overlaid on a 0–100% cycle axis. Normal cycles build the "
               "grey band + black median; a failure cycle is drawn in **red** where it deviates. It grows as parts are made.")
    W = None
    try:
        W = todays_worm(client)
    except Exception as e:
        st.warning(f"Cycle view temporarily unavailable: {e}")
    if W and (W["normals"] or W["failprofs"]):
        xa = list(np.linspace(0, 100, 90))
        fig2 = go.Figure()
        if len(W["normals"]) >= 3:
            NPn = np.array(W["normals"])
            p10 = np.percentile(NPn, 10, axis=0); p90 = np.percentile(NPn, 90, axis=0); med = np.median(NPn, axis=0)
            fig2.add_trace(go.Scatter(x=xa + xa[::-1], y=list(p90) + list(p10[::-1]), fill="toself",
                          fillcolor="rgba(155,188,216,0.40)", line=dict(width=0),
                          name="normal range (p10–p90)", hoverinfo="skip"))
            for i, p in enumerate(W["normals"]):
                fig2.add_trace(go.Scatter(x=xa, y=list(p), mode="lines",
                              line=dict(color="rgba(91,143,181,0.25)", width=1),
                              name="normal cycle", showlegend=(i == 0), hoverinfo="skip"))
            fig2.add_trace(go.Scatter(x=xa, y=list(med), line=dict(color=DARK, width=2.8),
                          name="normal-cycle median"))
        else:
            for i, p in enumerate(W["normals"]):
                fig2.add_trace(go.Scatter(x=xa, y=list(p), mode="lines",
                              line=dict(color="rgba(91,143,181,0.5)", width=1.2),
                              name="normal cycle", showlegend=(i == 0), hoverinfo="skip"))
        sel = st.session_state.get("failsel")
        for lbl, p in W["failprofs"]:
            hot = (sel == lbl)
            fig2.add_trace(go.Scatter(x=xa, y=list(p), line=dict(color=RED, width=4 if hot else 3),
                          name=f"FAILURE {lbl[11:]}"))
        fig2.update_layout(
            template="plotly_white", height=440, margin=dict(t=56, b=10, l=10, r=10),
            title=dict(text=f"Today's cycles overlaid — {len(W['normals'])} normal · {len(W['failprofs'])} failure",
                       font=dict(size=15, color=DARK), x=0, xanchor="left", y=0.97, yanchor="top"),
            legend=dict(orientation="v", x=1.02, y=1, xanchor="left", font=dict(size=10)),
            xaxis=dict(title="% through the production cycle", range=[0, 100], gridcolor="#eef3f5"),
            yaxis=dict(title="spindle current (A)", range=[0, 190], gridcolor="#eef3f5"))
        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False}, key="wormchart")
    else:
        st.info("Building today's cycle view… waiting for at least one completed part-cycle since 00:00 UTC.")

    st.markdown("**🚨 Failures detected today**")
    if W and W["fails"]:
        ftab = pd.DataFrame([{"Time (UTC)": str(t)[:19], "Peak current (A)": round(p, 0),
                              "Type": "surge + stoppage"} for t, p in W["fails"]])
        st.dataframe(ftab, use_container_width=True, hide_index=True)
        st.selectbox("Inspect a specific failure episode (highlights it in red above)",
                     [str(t)[:19] for t, _ in W["fails"]], key="failsel")
    else:
        st.caption("No confirmed failures today. Any detected failure will appear here and as a red cycle above.")

if ss.running:
    time.sleep(3)
    st.rerun()
