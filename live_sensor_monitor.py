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
.block-container { padding-top: 3.4rem; padding-bottom: 0.4rem; max-width: 100%; }
div[data-testid="stVerticalBlock"]{ gap: 0.72rem; }
h1,h2,h3{ letter-spacing:.2px; }
div[data-testid="stMetric"]{
  background:#ffffff;
  border:1px solid #e5edf1; border-left:3px solid var(--teal);
  border-radius:12px; padding:11px 15px; box-shadow:0 1px 3px rgba(20,28,34,.06);
  min-height:78px; display:flex; flex-direction:column; justify-content:center;
}
div[data-testid="stMetric"]:hover{ box-shadow:0 2px 8px rgba(20,28,34,.10); border-left-color:#246b86; }
div[data-testid="stMetricLabel"] p{ color:#5b6a74; font-weight:600; font-size:.66rem;
  text-transform:uppercase; letter-spacing:.05em; }
div[data-testid="stMetricValue"]{ color:#141c22; font-weight:700; font-size:1.28rem; line-height:1.15; }
.hero{ background:linear-gradient(90deg,#2F8FB3,#246b86); color:#fff;
  border-radius:12px; padding:13px 22px; margin-bottom:16px; }
.hero h1{ margin:0; font-size:1.32rem; color:#fff; }
.hero p{ margin:1px 0 0; color:#dbeef5; font-size:.85rem; }
.statuspill{ border-radius:10px; padding:9px 15px; font-weight:600; margin:6px 0 8px;
  font-size:.92rem; display:flex; justify-content:space-between; align-items:center; }
.statuspill .ts{ color:#54606A; font-weight:400; font-size:.78rem; }
.stTabs [data-baseweb="tab-list"]{ gap:6px; }
.stTabs [data-baseweb="tab"]{ padding:4px 10px; }
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
        a = (t - pd.Timedelta(seconds=33)).strftime("%Y-%m-%d %H:%M:%S")
        b = (t + pd.Timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
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
    """Normalized (0-100% of cycle), smoothed current + vibration profiles for one cycle."""
    try:
        d = _client.query(
            f"SELECT time,{','.join(CUR + PK)} FROM sensor_telemetry WHERE device_id='DOO4730' AND tenant_id=2 "
            f"AND time > TIMESTAMP '{s_iso}' AND time < TIMESTAMP '{e_iso}' ORDER BY time ASC",
            language="sql").to_pandas()
    except Exception:
        return None
    if len(d) < 15:
        return None
    d["time"] = pd.to_datetime(d["time"]).dt.tz_localize(None)
    t = (d["time"] - d["time"].iloc[0]).dt.total_seconds().values
    if t[-1] <= 0:
        return None
    xp = np.linspace(0, 1, N); k = 7; ker = np.ones(k) / k
    def norm(vals):
        p = np.interp(xp, t / t[-1], vals)
        return np.convolve(np.pad(p, (k, k), mode="edge"), ker, mode="same")[k:-k].tolist()
    return {"cur": norm(d[CUR].max(axis=1).values), "vib": norm(d[PK].max(axis=1).values)}


@st.cache_data(ttl=60, show_spinner="Building cycle history…")
def worm_data(_client, days=1, cap=80):
    """Overlay of cycles over a window (days=int → rolling N days; days='Today' → since 00:00 UTC)."""
    if days == "Today":
        now = pd.Timestamp.utcnow().tz_localize(None)
        mins = int((now - now.normalize()).total_seconds() // 60) + 2
    else:
        mins = int(days) * 1440 + 2
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
    # failures = a stoppage confirmed by a nearby surge (cap recent to bound query volume)
    abn = sta[sta["run_status"].isin(ABNORMAL)]["time"].tolist()[-60:]
    fails = []
    for tt in abn:
        a = (tt - pd.Timedelta(seconds=33)).strftime("%Y-%m-%d %H:%M:%S")
        b = (tt + pd.Timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            wv = _client.query(
                f"SELECT time,{','.join(CUR + PK)} FROM sensor_telemetry WHERE device_id='DOO4730' AND tenant_id=2 "
                f"AND time > TIMESTAMP '{a}' AND time < TIMESTAMP '{b}' ORDER BY time ASC", language="sql").to_pandas()
        except Exception:
            continue
        if len(wv) > 2:
            wv["time"] = pd.to_datetime(wv["time"]).dt.tz_localize(None)
            mc = wv[CUR].max(axis=1)
            if ((mc - mc.shift(2)) > ALARM).any():
                j = (wv["time"] - tt).abs().idxmin()          # vibration at the failure instant
                vib_at = float(wv.loc[j, PK].max())
                fails.append((tt, float(mc.max()), vib_at))
    ftimes = [t for t, _p, _v in fails]
    normal_cyc = [(s, e) for (s, e) in cyc if not any(s <= ft <= e for ft in ftimes)]
    if len(normal_cyc) > cap:                                   # sample evenly to keep it fast
        idx = sorted(set(np.linspace(0, len(normal_cyc) - 1, cap).round().astype(int)))
        normal_cyc = [normal_cyc[i] for i in idx]
    ncur = []; nvib = []
    for (s, e) in normal_cyc:
        p = cycle_profile(_client, s.strftime("%Y-%m-%d %H:%M:%S"), e.strftime("%Y-%m-%d %H:%M:%S"))
        if p is not None:
            ncur.append(p["cur"]); nvib.append(p["vib"])
    failprofs = []
    for ft, _pk, _vb in fails:
        before = [t for t in inc if t <= ft]
        s0 = before[-1] if before else ft - pd.Timedelta(seconds=mdur * 0.7)
        e0 = s0 + pd.Timedelta(seconds=mdur)
        p = cycle_profile(_client, s0.strftime("%Y-%m-%d %H:%M:%S"), e0.strftime("%Y-%m-%d %H:%M:%S"))
        if p is not None:
            failprofs.append((str(ft)[:19], p["cur"], p["vib"]))
    return {"ncur": ncur, "nvib": nvib, "failprofs": failprofs, "fails": fails, "ncyc": len(cyc)}


client = get_influx()
ss = st.session_state
ss.setdefault("running", True)

st.sidebar.header("Controls")
ss.running = st.sidebar.toggle("▶  Live (auto-refresh)", value=ss.running)
lookback = st.sidebar.slider("Fetch look-back (minutes)", 5, 60, 10)
st.sidebar.markdown("---")
st.sidebar.caption("Two-tier detector. Tier 1 = current surge (watch, ~149/day). "
                   "Tier 2 = surge **+ machine stoppage** or sensor dropout (real anomaly, ~0.5/day). "
                   "Only Tier 2 alarms.")

st.markdown('<div class="hero"><h1>🟢 DOO4730 — Live Tool-Anomaly Detection</h1>'
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

d = today_summary(client)

# ---- slim status pill (colour by live state) ----
if recent_conf or (recent_abn and (sen['flag_surge'].tail(30).any())) or (hb_now and recent_abn):
    stxt, scol = "🚨  CONFIRMED ANOMALY — current surge coincided with a machine stoppage", "#cf4040"
elif recent_abn:
    stxt, scol = f"⚠️  Machine stoppage ({state_now}) — watching for a coincident surge", "#c9821a"
elif sen["flag_surge"].tail(15).any():
    stxt, scol = "●  Surge detected — normal cutting (Tier-1 watch, not an alarm)", "#2f8fb3"
else:
    stxt, scol = "✓  Normal operation", "#2ca048"

# ---- one compact KPI strip (live + today together) ----
m = st.columns(7)
m[0].metric("Live current", f"{end_row['max_current']:.0f} A")
m[1].metric("Machine state", state_now)
m[2].metric("Tier-1 (watch)", n_watch)
m[3].metric("Tier-2 (alarm)", n_conf)
m[4].metric("Anomalies today", d["failures"])
m[5].metric("Last anomaly", (str(d["last"])[5:16] if d["last"] else "—"))
m[6].metric("Parts today", d["parts"])

st.markdown(
    f'<div class="statuspill" style="background:{scol}1a;border-left:4px solid {scol};color:{scol}">'
    f'<span>{stxt}</span><span class="ts">latest {str(t_end)[11:19]} · now {str(now_utc)[11:19]} UTC · '
    f'~{sen["time"].diff().dt.total_seconds().median():.1f}s · today since 00:00 UTC</span></div>',
    unsafe_allow_html=True)

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
                      name="Tier-2 CONFIRMED anomaly",
                      hovertemplate="CONFIRMED anomaly<br>%{x|%H:%M:%S} · %{y:.0f} A<extra></extra>"))
    for et in abn_times:
        if len(w) and w["time"].min() <= et <= w["time"].max():
            fig.add_vline(x=et, line_color=RED, line_width=1.6, opacity=0.5)
            fig.add_annotation(x=et, y=ymax, text="stoppage", showarrow=False,
                               font=dict(color=RED, size=10), textangle=-90, yanchor="top", xshift=-7)
    fig.update_layout(
        template="plotly_white", height=400, margin=dict(t=50, b=10, l=10, r=10),
        title=dict(text=f"Two-tier detector — live · {str(t_end)[:19]} UTC", font=dict(size=15, color=DARK),
                   x=0, xanchor="left", y=0.97, yanchor="top"),
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left", font=dict(size=11)),
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
            f"This is the real anomaly signal (~0.5×/day) and is the only thing that raises the loud alert.\n"
            f"- **Teal line (current)** — the real, instantaneous spindle current; the sharp peaks are "
            f"genuine surges, and a surge over {ALARM:.0f} A is exactly the signal we detect.\n"
            f"- **Red vertical line** — a live machine stoppage from the status stream.\n"
            f"- **Green dotted line (vibration)** — shown for context on the right axis; click it in the legend "
            f"to hide. It stays low/erratic and is not used to trigger — current is the trigger.\n"
            f"- This is how a real catch is told apart from the ~150 daily false surges: the stoppage confirms it."
        )

with tab_worm:
    cwa, _cwb = st.columns([1, 3])
    days = cwa.selectbox("History window", ["Today", 1, 3, 7, 14],
                         format_func=lambda x: "Today (since 00:00 UTC)" if x == "Today"
                         else f"last {x} day" + ("s" if x > 1 else ""), key="wormdays")
    st.caption("Completed part-cycles overlaid on a 0–100% cycle axis. Normal cycles build the light-blue band + "
               "black median; an anomaly cycle is drawn in **red**. Toggle **Current / Vibration** via the legend. "
               "Updates live — a new cycle joins within ~60 s of each part finishing (~every 7 min).")
    W = None
    try:
        W = worm_data(client, days)
    except Exception as e:
        st.warning(f"Cycle view temporarily unavailable: {e}")
    if W and (W["ncur"] or W["failprofs"]):
        xa = list(np.linspace(0, 100, 90))
        fig2 = go.Figure()
        # ---- current group ----
        if len(W["ncur"]) >= 3:
            A = np.array(W["ncur"]); p10 = np.percentile(A, 10, axis=0); p90 = np.percentile(A, 90, axis=0); med = np.median(A, axis=0)
            fig2.add_trace(go.Scatter(x=xa + xa[::-1], y=list(p90) + list(p10[::-1]), fill="toself",
                          fillcolor="rgba(155,188,216,0.40)", line=dict(width=0), legendgroup="Current",
                          showlegend=False, hoverinfo="skip"))
            for i, p in enumerate(W["ncur"]):
                fig2.add_trace(go.Scatter(x=xa, y=list(p), mode="lines", line=dict(color="rgba(91,143,181,0.22)", width=1),
                              legendgroup="Current", name="Current", showlegend=(i == 0), hoverinfo="skip"))
            fig2.add_trace(go.Scatter(x=xa, y=list(med), line=dict(color=DARK, width=2.8),
                          legendgroup="Current", showlegend=False, name="current median"))
        else:
            for i, p in enumerate(W["ncur"]):
                fig2.add_trace(go.Scatter(x=xa, y=list(p), mode="lines", line=dict(color="rgba(91,143,181,0.5)", width=1.2),
                              legendgroup="Current", name="Current", showlegend=(i == 0), hoverinfo="skip"))
        if W["ncur"]:                                          # newest cycle stands out as a teal dotted line
            fig2.add_trace(go.Scatter(x=xa, y=list(W["ncur"][-1]), mode="lines",
                          line=dict(color="#2F8FB3", width=2.6, dash="dot"),
                          legendgroup="Current", name="latest cycle"))
        # ---- vibration group (hidden until toggled on) ----
        if len(W["nvib"]) >= 3:
            Vv = np.array(W["nvib"]); vp10 = np.percentile(Vv, 10, axis=0); vp90 = np.percentile(Vv, 90, axis=0); vmed = np.median(Vv, axis=0)
            fig2.add_trace(go.Scatter(x=xa + xa[::-1], y=list(vp90) + list(vp10[::-1]), fill="toself",
                          fillcolor="rgba(44,160,72,0.16)", line=dict(width=0), legendgroup="Vibration",
                          showlegend=False, hoverinfo="skip", yaxis="y2", visible="legendonly"))
            fig2.add_trace(go.Scatter(x=xa, y=list(vmed), line=dict(color="#166b32", width=2.6),
                          legendgroup="Vibration", name="Vibration", yaxis="y2", visible="legendonly"))
            fig2.add_trace(go.Scatter(x=xa, y=list(W["nvib"][-1]), line=dict(color="#2ca048", width=2.4, dash="dot"),
                          legendgroup="Vibration", name="latest cycle (vib)", showlegend=False,
                          yaxis="y2", visible="legendonly"))
        sel = st.session_state.get("failsel")
        for lbl, curp, vibp in W["failprofs"]:
            hot = (sel == lbl)
            fig2.add_trace(go.Scatter(x=xa, y=list(curp), line=dict(color=RED, width=4 if hot else 3),
                          legendgroup="Current", showlegend=False, name=f"ANOMALY {lbl[11:]}"))
            fig2.add_trace(go.Scatter(x=xa, y=list(vibp), line=dict(color="#e08a00", width=2, dash="dash"),
                          legendgroup="Vibration", showlegend=False, name=f"ANOMALY vib {lbl[11:]}",
                          yaxis="y2", visible="legendonly"))
        wlabel = "today" if days == "Today" else f"last {days}d"
        fig2.update_layout(
            template="plotly_white", height=400, margin=dict(t=50, b=10, l=10, r=10),
            title=dict(text=f"Cycles overlaid ({W['ncyc']} cycles · {wlabel}) — {len(W['ncur'])} normal · {len(W['failprofs'])} anomaly",
                       font=dict(size=15, color=DARK), x=0, xanchor="left", y=0.97, yanchor="top"),
            legend=dict(orientation="v", x=1.02, y=1, xanchor="left", font=dict(size=11), groupclick="togglegroup"),
            xaxis=dict(title="% through the production cycle", range=[0, 100], gridcolor="#eef3f5"),
            yaxis=dict(title="spindle current (A)", range=[0, 190], gridcolor="#eef3f5"),
            yaxis2=dict(title="vibration (pk-pk)", overlaying="y", side="right", range=[0, 35], showgrid=False))
        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False}, key="wormchart")
    else:
        st.info("Building the cycle view… waiting for at least one completed part-cycle in the selected window.")

    st.markdown("**🚨 Anomalies detected in this window**")
    if W and W["fails"]:
        ftab = pd.DataFrame([{"Time (UTC)": str(t)[:19], "Peak current (A)": round(p, 0),
                              "Vibration (pk-pk)": round(v, 1)} for t, p, v in W["fails"]])
        st.dataframe(ftab, use_container_width=True, hide_index=True)
        st.selectbox("Inspect a specific anomaly episode (highlights it in red above)",
                     [str(t)[:19] for t, _p, _v in W["fails"]], key="failsel")
    else:
        st.caption("No confirmed anomalies in this window. Any detected anomaly will appear here and as a red cycle above.")

if ss.running:
    time.sleep(3)
    st.rerun()
