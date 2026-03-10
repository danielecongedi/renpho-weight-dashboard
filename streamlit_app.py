import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client
from datetime import datetime, timedelta, date, time as dtime

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG  (deve essere il primo comando Streamlit)
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Weight Monitor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CSS  — tema bianco, selettori stabili, font Google
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── root vars ── */
:root {
  --c-bg:       #f8f9fb;
  --c-surface:  #ffffff;
  --c-border:   #e4e8ef;
  --c-border2:  #cdd4e0;
  --c-text:     #0f1923;
  --c-text2:    #5a6478;
  --c-text3:    #8c96a8;
  --c-green:    #22a55b;
  --c-green-l:  #e8f7ef;
  --c-green-d:  #166b3c;
  --c-amber:    #e07b20;
  --c-amber-l:  #fef3e2;
  --c-red:      #d63b3b;
  --c-red-l:    #fdeaea;
  --c-blue:     #2563eb;
  --c-blue-l:   #eff4ff;
  --c-purple:   #7c3aed;
  --c-purple-l: #f3eeff;
  --radius:     12px;
  --shadow:     0 1px 4px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
}

/* ── global font ── */
html, body, [class*="css"], .stApp {
  font-family: 'Outfit', sans-serif !important;
}

/* ── sfondo app ── */
.stApp { background: var(--c-bg) !important; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--c-surface) !important;
  border-right: 1px solid var(--c-border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
  font-size: 13px !important;
  font-weight: 700 !important;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--c-text3) !important;
  margin-bottom: 12px;
}

/* ── titolo principale ── */
.wm-title {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 4px;
}
.wm-title h1 {
  font-family: 'Outfit', sans-serif !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: var(--c-text) !important;
  letter-spacing: -.03em;
  margin: 0 !important;
}
.wm-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  background: var(--c-green-l);
  color: var(--c-green-d);
  border: 1px solid #b6e8cc;
  padding: 2px 8px;
  border-radius: 20px;
  letter-spacing: .06em;
}
.wm-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--c-text3);
  letter-spacing: .06em;
  margin-bottom: 20px;
}

/* ── KPI card (metric override) ── */
div[data-testid="stMetric"] {
  background: var(--c-surface) !important;
  border: 1px solid var(--c-border) !important;
  border-radius: var(--radius) !important;
  padding: 18px 20px 14px !important;
  box-shadow: var(--shadow) !important;
  transition: box-shadow .18s, border-color .18s;
}
div[data-testid="stMetric"]:hover {
  border-color: var(--c-border2) !important;
  box-shadow: 0 2px 8px rgba(0,0,0,.08), 0 8px 24px rgba(0,0,0,.06) !important;
}
div[data-testid="stMetricLabel"] > div {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important;
  font-weight: 500 !important;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--c-text3) !important;
}
div[data-testid="stMetricValue"] {
  font-family: 'Outfit', sans-serif !important;
  font-size: 1.65rem !important;
  font-weight: 700 !important;
  color: var(--c-text) !important;
  line-height: 1.2;
}
div[data-testid="stMetricDelta"] svg { display: none; }
div[data-testid="stMetricDelta"] > div {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
}

/* ── sezione banner ── */
.wm-banner {
  background: linear-gradient(135deg, #f0faf4 0%, #e8f7ef 100%);
  border: 1px solid #b6e8cc;
  border-left: 4px solid var(--c-green);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin: 16px 0;
  display: flex;
  align-items: center;
  gap: 12px;
}
.wm-banner-icon { font-size: 24px; }
.wm-banner-text { line-height: 1.4; }
.wm-banner-text strong { color: var(--c-green-d); font-weight: 700; font-size: 15px; }
.wm-banner-text span { font-size: 13px; color: var(--c-text2); }

.wm-banner-amber {
  background: linear-gradient(135deg, #fffaf0 0%, var(--c-amber-l) 100%);
  border: 1px solid #f5d49a;
  border-left: 4px solid var(--c-amber);
}
.wm-banner-amber strong { color: #8c4a0a !important; }

/* ── sezione heading ── */
.wm-section {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 24px 0 12px;
}
.wm-section-icon {
  width: 28px; height: 28px;
  background: var(--c-green-l);
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px;
}
.wm-section-title {
  font-size: 14px;
  font-weight: 700;
  color: var(--c-text);
  letter-spacing: -.01em;
}

/* ── tabs ── */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--c-text3) !important;
  padding: 10px 16px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--c-green) !important;
  border-bottom: 2px solid var(--c-green) !important;
}

/* ── divider ── */
hr { border-color: var(--c-border) !important; margin: 20px 0 !important; }

/* ── bottoni ── */
.stButton > button {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: .05em;
  text-transform: uppercase;
  border-radius: 8px !important;
  transition: all .15s !important;
}
.stButton > button[kind="primary"] {
  background: var(--c-green) !important;
  border-color: var(--c-green) !important;
  color: white !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--c-green-d) !important;
}

/* ── dataframe ── */
div[data-testid="stDataFrame"] {
  border: 1px solid var(--c-border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden;
}

/* ── caption ── */
div[data-testid="stCaptionContainer"] p {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important;
  color: var(--c-text3) !important;
}

/* ── progress bar target ── */
.wm-progress-outer {
  background: var(--c-border);
  border-radius: 20px;
  height: 8px;
  overflow: hidden;
  margin-top: 6px;
}
.wm-progress-inner {
  height: 100%;
  border-radius: 20px;
  background: linear-gradient(90deg, var(--c-green), #56c97a);
  transition: width .4s ease;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════
DEFAULT_BASELINE_DATE   = date(2026, 8, 1)
DEFAULT_BASELINE_WEIGHT = 112.0
DEFAULT_TARGET_WEIGHT   = 72.0
DEFAULT_HEIGHT_M        = 1.82

FORECAST_LOOKBACK_DAYS       = 90
PATTERN_WEEKS                = 6
MAX_LOCAL_ANCHOR_DISTANCE_DAYS = 4
WEEKLY_LOSS_MIN              = 0.10
WEEKLY_LOSS_MAX              = 1.80
SHAPE_CLIP_MAX               = 0.80
SHAPE_CLIP_MIN               = -0.05
TARGET_CONFIRM_DAYS          = 3

# palette Plotly (bianco)
PC = dict(
    green   = "#22a55b",
    green_l = "#56c97a",
    amber   = "#e07b20",
    red     = "#d63b3b",
    blue    = "#2563eb",
    purple  = "#7c3aed",
    text    = "#0f1923",
    text2   = "#5a6478",
    grid    = "#e4e8ef",
    bg      = "#ffffff",
    surface = "#f8f9fb",
)

BASE_LAYOUT = dict(
    paper_bgcolor = PC["bg"],
    plot_bgcolor  = PC["bg"],
    font = dict(family="JetBrains Mono, monospace", size=11, color=PC["text2"]),
    xaxis = dict(gridcolor=PC["grid"], zeroline=False, showline=False,
                 tickfont=dict(size=10, color=PC["text2"])),
    yaxis = dict(gridcolor=PC["grid"], zeroline=False, showline=False,
                 tickfont=dict(size=10, color=PC["text2"])),
    hovermode = "x unified",
    hoverlabel = dict(bgcolor="white", bordercolor=PC["grid"],
                      font=dict(family="JetBrains Mono, monospace", size=11, color=PC["text"])),
    margin = dict(l=10, r=10, t=36, b=10),
    legend = dict(orientation="h", yanchor="bottom", y=1.02,
                  xanchor="right", x=1, font=dict(size=10),
                  bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
)

# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def ts_midnight(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, dtime.min))

def fmt_delta(v: float | None, dec: int = 2) -> str | None:
    if v is None or not np.isfinite(float(v)):
        return None
    return f"{v:+.{dec}f}"

def next_saturday(d: date) -> date:
    days = (5 - d.weekday()) % 7
    r = d + timedelta(days=days)
    return r if r > d else r + timedelta(days=7)

def week_saturday(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts).normalize()
    return ts - pd.Timedelta(days=(ts.dayofweek - 5) % 7)

def add_vline(fig, x_dt, label: str, color: str = "#cdd4e0"):
    xp = pd.to_datetime(x_dt).to_pydatetime()
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=xp, x1=xp, y0=0, y1=1,
                  line=dict(dash="dot", color=color, width=1.2))
    fig.add_annotation(x=xp, y=0.97, xref="x", yref="paper",
                       text=label, showarrow=False,
                       xanchor="left", yanchor="top",
                       font=dict(size=10, color=color))

def progress_html(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    return (f'<div class="wm-progress-outer">'
            f'<div class="wm-progress-inner" style="width:{pct:.1f}%"></div></div>')

def banner_html(icon: str, strong: str, sub: str, amber: bool = False) -> str:
    extra = "wm-banner-amber" if amber else ""
    return (f'<div class="wm-banner {extra}">'
            f'<div class="wm-banner-icon">{icon}</div>'
            f'<div class="wm-banner-text"><strong>{strong}</strong><br>'
            f'<span>{sub}</span></div></div>')

def section_html(icon: str, title: str) -> str:
    return (f'<div class="wm-section">'
            f'<div class="wm-section-icon">{icon}</div>'
            f'<div class="wm-section-title">{title}</div></div>')

# ═══════════════════════════════════════════════════════════════════
# SERIE GIORNALIERA
# ═══════════════════════════════════════════════════════════════════

def build_daily_series(daily_df: pd.DataFrame) -> pd.Series:
    d = daily_df.copy().sort_values("date")
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    s = d.set_index("date")["weight"].astype(float)
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    return s.interpolate(method="time", limit=2, limit_area="inside")

# ═══════════════════════════════════════════════════════════════════
# ANCHOR LOCALE  (WLS, nessun bias verso il minimo)
# ═══════════════════════════════════════════════════════════════════

def local_anchor(series: pd.Series, target: pd.Timestamp,
                 max_dist: int = MAX_LOCAL_ANCHOR_DISTANCE_DAYS) -> float | None:
    target = pd.to_datetime(target).normalize()
    if target in series.index and pd.notna(series.loc[target]):
        return float(series.loc[target])
    s = series.dropna()
    if s.empty:
        return None
    deltas = ((s.index - target) / pd.Timedelta(days=1)).astype(float)
    mask = np.abs(deltas) <= max_dist
    local = s[mask]
    if local.empty:
        return None
    x = deltas[mask].values.astype(float)
    y = local.values.astype(float)
    w = 1.0 / (1.0 + np.abs(x))
    if len(local) >= 3 and (x < 0).sum() >= 1 and (x > 0).sum() >= 1:
        X = np.vstack([np.ones(len(x)), x]).T
        try:
            beta = np.linalg.pinv(X.T @ np.diag(w) @ X) @ (X.T @ np.diag(w) @ y)
            return float(beta[0])
        except Exception:
            pass
    return float(np.sum(y * w) / np.sum(w))

# ═══════════════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_data(ttl=60, show_spinner=False)
def load_manual() -> pd.DataFrame:
    res = get_supabase().table("manual_entries").select("*").order("date").execute()
    rows = res.data or []
    if not rows:
        return pd.DataFrame(columns=["id", "date", "weight", "bmi", "source"])
    df = pd.DataFrame(rows)
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["bmi"]    = pd.to_numeric(df["bmi"], errors="coerce")
    df["source"] = df.get("source", "manual").fillna("manual")
    return df.dropna(subset=["date","weight"]).sort_values("date").reset_index(drop=True)

def insert_manual_entry(dt: pd.Timestamp, weight: float, bmi: float | None):
    existing = load_manual()
    if not existing.empty and ((existing["date"] - dt).abs() < pd.Timedelta(minutes=1)).any():
        raise ValueError("Esiste già una misura in questo orario.")
    get_supabase().table("manual_entries").insert({
        "date": pd.Timestamp(dt).isoformat(),
        "weight": float(weight),
        "bmi": float(bmi) if bmi is not None and pd.notna(bmi) else None,
        "source": "manual",
    }).execute()
    load_manual.clear()

def delete_manual_entries_by_id(ids: list[int]):
    if not ids: return
    get_supabase().table("manual_entries").delete().in_("id", ids).execute()
    load_manual.clear()

def clear_manual_entries():
    get_supabase().table("manual_entries").delete().neq("id", 0).execute()
    load_manual.clear()

# ═══════════════════════════════════════════════════════════════════
# RENPHO CSV
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Scaricamento RENPHO…")
def load_renpho_csv(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url, header=None)
    if raw.shape[1] == 1:
        s = raw[0].astype(str).str.strip().str.replace(r'^"|"$', "", regex=True)
        raw = s.str.split(",", expand=True)
    if raw.shape[1] < 3:
        raise ValueError("CSV non riconosciuto (servono: data, ora, peso).")
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(
        raw[0].astype(str).str.strip() + " " + raw[1].astype(str).str.strip(),
        errors="coerce", dayfirst=True)
    w = (raw[2].astype(str).str.strip()
         .str.replace(",", ".", regex=False)
         .str.replace("kg", "", regex=False).str.strip())
    df["weight"] = pd.to_numeric(w, errors="coerce")
    df = df.dropna(subset=["date","weight"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df["source"] = "renpho"; df["bmi"] = np.nan; df["id"] = np.nan
    return df.reset_index(drop=True)

def combine_data(renpho: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    if "id" not in renpho.columns:
        renpho = renpho.copy(); renpho["id"] = np.nan
    if manual.empty:
        return renpho.sort_values("date").reset_index(drop=True)
    df = pd.concat([renpho, manual], ignore_index=True)
    df["__p"] = df["source"].map({"renpho":0,"manual":1}).fillna(0)
    df = (df.sort_values(["date","__p"]).drop(columns=["__p"])
            .drop_duplicates(subset=["date"], keep="last"))
    return df.sort_values("date").reset_index(drop=True)

def add_bmi(df: pd.DataFrame, h: float) -> pd.DataFrame:
    if h <= 0: return df
    df = df.copy()
    m = df["bmi"].isna() & df["weight"].notna()
    df.loc[m, "bmi"] = df.loc[m, "weight"] / (h ** 2)
    return df

def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy(); d["day"] = d["date"].dt.date
    out = d.groupby("day", as_index=False).agg(
        date=("date","max"), weight=("weight","median"), bmi=("bmi","median"))
    out["source"] = "daily"
    return out.sort_values("date").reset_index(drop=True)

def last_measurement(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("date")
    ld = df["date"].dt.date.max()
    dd = df[df["date"].dt.date == ld]
    m  = dd[dd["source"] == "manual"]
    return (m if not m.empty else dd).sort_values("date").iloc[-1]

# ═══════════════════════════════════════════════════════════════════
# MODELLO FORECAST  v9
# ─────────────────────────────────────────────────────────────────
#  1. weekly_loss  → mediana pesata esponenzialmente (recente > antico)
#  2. anchor       → WLS sul sabato esatto, nessun bias verso il minimo
#  3. weekday shape→ appresa su dati RAW (non smoothed)
#  4. clip loss    → [0.10, 1.80]  (plateau + perdite sostenibili)
#  5. clip shape   → [-0.05, 0.80] (non taglia picchi reali)
#  6. target date  → confermata dopo TARGET_CONFIRM_DAYS consecutivi
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Calcolo modello…")
def fit_forecast_model(daily_df, lookback_days=FORECAST_LOOKBACK_DAYS,
                       smoothing_window=5, pattern_weeks=PATTERN_WEEKS) -> dict:
    if daily_df.empty or len(daily_df) < 14:
        return {"ok": False, "reason": "too_few_points"}

    df = daily_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    end   = df["date"].max()
    sub   = df[df["date"] >= end - pd.Timedelta(days=lookback_days)].copy()
    if len(sub) < 14: sub = df.copy()
    sub   = sub.sort_values("date").reset_index(drop=True)

    daily_raw = build_daily_series(sub)
    daily_smooth = (daily_raw
                    .interpolate(method="time", limit_direction="both")
                    .rolling(window=smoothing_window, min_periods=1).mean())

    # anchor settimanali
    sat_idx = pd.date_range(week_saturday(sub["date"].min()),
                            week_saturday(sub["date"].max()), freq="7D")
    week_rows = []
    for sat in sat_idx:
        est = local_anchor(daily_raw, sat)
        if est is not None and np.isfinite(est):
            week_rows.append({"week_saturday": pd.to_datetime(sat),
                               "anchor_weight": float(est)})

    week_df = pd.DataFrame(week_rows).sort_values("week_saturday").reset_index(drop=True)
    if len(week_df) < 4:
        return {"ok": False, "reason": "not_enough_weeks"}

    week_df["weekly_delta"] = week_df["anchor_weight"].diff()

    # mediana pesata esponenzialmente
    losses_raw = week_df["weekly_delta"].dropna().values.astype(float)
    losses = np.clip(losses_raw, -2.0, 0.0)
    n = len(losses)
    if n == 0: return {"ok": False, "reason": "no_weekly_deltas"}

    exp_w = np.exp(np.linspace(0, 1.1, n))
    order = np.argsort(losses)
    cumw  = np.cumsum(exp_w[order]) / np.sum(exp_w)
    weighted_median = float(np.interp(0.5, cumw, losses[order]))

    recent_avg_weekly_loss = float(np.mean(np.abs(losses)))
    robust_recent_loss     = float(abs(weighted_median))
    weekly_loss = float(np.clip(robust_recent_loss, WEEKLY_LOSS_MIN, WEEKLY_LOSS_MAX))

    # weekday shape su RAW
    hist = (pd.DataFrame({"date": pd.to_datetime(daily_raw.index),
                           "weight_raw": daily_raw.values})
              .dropna())
    hist["weekday"]      = hist["date"].dt.dayofweek
    hist["week_saturday"]= hist["date"].apply(week_saturday)

    recent_weeks = set(week_df.tail(pattern_weeks)["week_saturday"].tolist())
    anchor_map   = week_df.set_index("week_saturday")["anchor_weight"].to_dict()

    rel_rows = []
    for _, row in hist[hist["week_saturday"].isin(recent_weeks)].iterrows():
        aw = anchor_map.get(pd.to_datetime(row["week_saturday"]))
        if aw is not None and pd.notna(row["weight_raw"]):
            rel_rows.append({"weekday": int(row["weekday"]),
                              "rel": float(row["weight_raw"] - aw)})

    default_shape = {0:.40, 1:.28, 2:.18, 3:.10, 4:.04, 5:.00, 6:.08}
    rel_df = pd.DataFrame(rel_rows)
    if rel_df.empty:
        wds = default_shape.copy()
    else:
        wds = rel_df.groupby("weekday")["rel"].median().to_dict()
        for wd in range(7): wds.setdefault(wd, default_shape[wd])

    wds = {k: float(np.clip(v, SHAPE_CLIP_MIN, SHAPE_CLIP_MAX)) for k,v in wds.items()}
    wds[5] = max(wds[5], SHAPE_CLIP_MIN)
    wds[6] = max(wds[6], wds[5])
    for prev, cur in [(4,5),(3,4),(2,3),(1,2),(0,1)]:
        wds[prev] = max(wds[prev], wds[cur])

    last_sat    = pd.to_datetime(week_df["week_saturday"].iloc[-1]).normalize()
    last_anchor = float(week_df["anchor_weight"].iloc[-1])
    last_date   = pd.to_datetime(sub["date"].max()).normalize()
    last_w      = float(sub.loc[sub["date"]==last_date, "weight"].iloc[-1])

    week_df["weekly_loss_abs"] = week_df["weekly_delta"].clip(upper=0).abs()

    return {
        "ok": True, "type": "v9",
        "daily_raw": daily_raw, "daily_smooth": daily_smooth,
        "last_date": last_date, "last_weight": last_w,
        "last_sat": last_sat, "last_anchor": last_anchor,
        "weekday_shape": wds, "weekly_loss": weekly_loss,
        "recent_avg_weekly_loss": recent_avg_weekly_loss,
        "robust_recent_loss": robust_recent_loss,
        "week_df": week_df,
    }

def anchor_for_week(model: dict, sat: pd.Timestamp) -> float:
    sat   = pd.to_datetime(sat).normalize()
    weeks = int((sat - model["last_sat"]).days / 7)
    return model["last_anchor"] if weeks <= 0 else float(model["last_anchor"] - weeks * model["weekly_loss"])

def forecast_series(model: dict, start: pd.Timestamp, horizon: int):
    if not model.get("ok"): return pd.DatetimeIndex([]), []
    start = pd.to_datetime(start).normalize()
    dates = pd.date_range(start, start + pd.Timedelta(days=horizon), freq="D")
    raw   = model["daily_raw"]
    ld    = model["last_date"]
    vals  = []
    for d in dates:
        if d <= ld and d in raw.index and pd.notna(raw.loc[d]):
            vals.append(float(raw.loc[d]))
        else:
            vals.append(anchor_for_week(model, week_saturday(d))
                        + model["weekday_shape"].get(int(d.dayofweek), 0.0))
    return dates, vals

def predict_weight(model: dict, when: pd.Timestamp) -> float:
    _, v = forecast_series(model, when, 0)
    return float(v[0]) if v else np.nan

def estimate_target_date(model, target_weight, start_from, max_horizon=365):
    dates, values = forecast_series(model, pd.Timestamp(start_from), max_horizon)
    if not values: return None, None
    consecutive = 0; first_under = None
    for d, w in zip(dates, values):
        if w <= float(target_weight):
            if consecutive == 0: first_under = pd.to_datetime(d).date()
            consecutive += 1
            if consecutive >= TARGET_CONFIRM_DAYS:
                return first_under, (first_under - start_from).days
        else:
            consecutive = 0; first_under = None
    return None, None

# ═══════════════════════════════════════════════════════════════════
# SECRETS CHECK
# ═══════════════════════════════════════════════════════════════════
csv_url = st.secrets.get("CSV_URL", "")
if not csv_url:
    st.error("⚠️ `CSV_URL` non impostato nei Secrets."); st.stop()
if not all(k in st.secrets for k in ("SUPABASE_URL","SUPABASE_KEY")):
    st.error("⚠️ Credenziali Supabase mancanti."); st.stop()

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Impostazioni")
    target_weight   = st.number_input("🎯 Target (kg)",        value=float(DEFAULT_TARGET_WEIGHT),   step=0.5)
    baseline_weight = st.number_input("⚑ Peso baseline (kg)", value=float(DEFAULT_BASELINE_WEIGHT), step=0.5)
    baseline_date   = st.date_input("⚑ Data baseline",        value=DEFAULT_BASELINE_DATE)
    st.divider()
    height_m         = st.number_input("📏 Altezza (m)",        value=float(DEFAULT_HEIGHT_M), step=0.01)
    st.divider()
    ma_window        = st.selectbox("📈 Media mobile (gg)",    [7,14,21,30], index=0)
    forecast_smooth  = st.selectbox("🧠 Smoothing forecast",   [3,5,7,10],   index=1)
    forecast_horizon = st.selectbox("⏳ Orizzonte (gg)",       [30,60,90,180], index=1)
    st.divider()
    show_raw    = st.toggle("Punti RAW",     value=True)
    show_daily  = st.toggle("Linea DAILY",   value=True)
    show_smooth = st.toggle("Curva MA",      value=True)
    show_volume = st.toggle("Banda variabilità", value=True)
    st.divider()
    if st.button("🔄 Refresh RENPHO", use_container_width=True):
        load_renpho_csv.clear(); st.rerun()

# ═══════════════════════════════════════════════════════════════════
# CARICAMENTO DATI
# ═══════════════════════════════════════════════════════════════════
try:    renpho = load_renpho_csv(csv_url)
except Exception as e: st.error(f"Errore RENPHO: {e}"); st.stop()

try:    manual = load_manual()
except Exception as e: st.error(f"Errore Supabase: {e}"); st.stop()

df    = add_bmi(combine_data(renpho, manual), height_m)
if df.empty: st.warning("Nessun dato."); st.stop()

daily = make_daily(df)
daily["ma"] = daily["weight"].rolling(ma_window, min_periods=1).mean()

min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("📅 Intervallo", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)
start_d, end_d = (date_range if isinstance(date_range,tuple) and len(date_range)==2
                  else (min_date, max_date))

df_f    = df[(df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].copy()
daily_f = daily[(daily["date"].dt.date >= start_d) & (daily["date"].dt.date <= end_d)].copy()
if df_f.empty: st.warning("Intervallo vuoto."); st.stop()
daily_f["ma"] = daily_f["weight"].rolling(ma_window, min_periods=1).mean()

# ═══════════════════════════════════════════════════════════════════
# METRICHE
# ═══════════════════════════════════════════════════════════════════
today       = date.today()
next_sat    = next_saturday(today)
next_sat_ts = ts_midnight(next_sat)

last_row      = last_measurement(df_f)
last_dt       = pd.to_datetime(last_row["date"])
last_w        = float(last_row["weight"])
last_bmi_val  = float(last_row["bmi"]) if pd.notna(last_row.get("bmi")) else np.nan

prev_df  = df_f[df_f["date"] < last_dt].sort_values("date")
prev_row = prev_df.iloc[-1] if not prev_df.empty else None
prev_w   = float(prev_row["weight"]) if prev_row is not None else None
prev_bmi = (float(prev_row["bmi"]) if prev_row is not None
            and pd.notna(prev_row.get("bmi")) else None)

delta_bmi   = ((last_bmi_val - prev_bmi) if prev_bmi is not None
               and np.isfinite(last_bmi_val) and np.isfinite(prev_bmi) else None)
loss_base   = float(baseline_weight - last_w)
dist_target = float(last_w - float(target_weight))
delta_loss  = float(loss_base - (baseline_weight - prev_w)) if prev_w else None
delta_dist  = float(dist_target - (prev_w - float(target_weight))) if prev_w else None

# progresso %
total_journey = float(baseline_weight - float(target_weight))
progress_pct  = max(0.0, min(100.0, loss_base / total_journey * 100)) if total_journey > 0 else 0.0

model = fit_forecast_model(daily, lookback_days=FORECAST_LOOKBACK_DAYS,
                           smoothing_window=forecast_smooth, pattern_weeks=PATTERN_WEEKS)
pred_next_sat  = predict_weight(model, next_sat_ts) if model.get("ok") else None
target_date_est, days_to_target = (
    estimate_target_date(model, float(target_weight), today)
    if model.get("ok") else (None, None))

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="wm-title">'
    '<h1>⚖️ Weight Monitor</h1>'
    '<span class="wm-badge">FORECAST v9</span>'
    '</div>'
    '<div class="wm-subtitle">RENPHO · Aggiornato in tempo reale</div>',
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab_dash, tab_manual, tab_forecast, tab_data = st.tabs(
    ["📊  Cruscotto", "✍️  Manuale", "🔮  Forecast", "🧾  Dataset"])

# ───────────────────────────────────────────────────────────────────
# TAB — CRUSCOTTO
# ───────────────────────────────────────────────────────────────────
with tab_dash:

    # ── KPI row 1 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⚖️ Ultima misura",     f"{last_w:.2f} kg",
              last_dt.strftime("%d %b  %H:%M"), delta_color="off")
    c2.metric("📐 BMI",
              f"{last_bmi_val:.2f}" if np.isfinite(last_bmi_val) else "—",
              (fmt_delta(delta_bmi,2)+" vs prec.") if delta_bmi else "—",
              delta_color="inverse")
    c3.metric("📉 Perso dal baseline", f"{loss_base:+.2f} kg",
              (fmt_delta(delta_loss,2)+" kg") if delta_loss else "—",
              delta_color="normal")
    c4.metric("🎯 Al target",         f"{dist_target:+.2f} kg",
              (fmt_delta(delta_dist,2)+" kg") if delta_dist else "—",
              delta_color="inverse")

    # ── progress bar ──
    st.markdown(
        f"**Progresso verso il target** — {progress_pct:.1f}%"
        f"&nbsp;&nbsp;<span style='color:var(--c-text3);font-size:12px'>"
        f"{last_w:.1f} kg → {float(target_weight):.1f} kg  "
        f"(partenza {baseline_weight:.1f} kg)</span>",
        unsafe_allow_html=True)
    st.markdown(progress_html(progress_pct), unsafe_allow_html=True)

    # ── KPI row 2 ──
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric("🔮 Sabato " + next_sat.strftime("%d %b"),
              f"{pred_next_sat:.2f} kg" if pred_next_sat else "—",
              "prossimo sabato", delta_color="off")
    k2.metric("📅 Target stimato",
              target_date_est.strftime("%d %b %Y") if target_date_est else "—",
              f"{days_to_target} giorni da oggi" if days_to_target else "—")
    if model.get("ok"):
        k3.metric("📈 Ritmo (mediana)",
                  f"−{model['weekly_loss']:.2f} kg/sett",
                  f"media 3m: −{model['recent_avg_weekly_loss']:.2f}")

    # ── banner target ──
    if target_date_est:
        st.markdown(
            banner_html("🏁",
                        f"Target il {target_date_est.strftime('%d %b %Y')} "
                        f"({days_to_target} giorni)",
                        f"Mancano {dist_target:.2f} kg al target di {float(target_weight):.1f} kg "
                        f"— progresso {progress_pct:.1f}%"),
            unsafe_allow_html=True)

    # ── grafico principale ──
    st.markdown(section_html("📈", "Trend peso + Forecast"), unsafe_allow_html=True)

    fig = go.Figure()

    # banda variabilità (±std rolling 7gg)
    if show_volume and not daily_f.empty and len(daily_f) >= 7:
        roll_std = daily_f["weight"].rolling(7, min_periods=3).std().fillna(0)
        fig.add_trace(go.Scatter(
            x=pd.concat([daily_f["date"], daily_f["date"].iloc[::-1]]),
            y=pd.concat([daily_f["weight"]+roll_std, (daily_f["weight"]-roll_std).iloc[::-1]]),
            fill="toself", mode="none",
            fillcolor="rgba(34,165,91,0.08)",
            name="Variabilità ±1σ", showlegend=True,
            hoverinfo="skip"))

    if show_raw:
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["weight"], mode="markers", name="RAW",
            marker=dict(size=3.5, color=PC["text2"], opacity=0.4),
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>%{y:.2f} kg<extra>RAW</extra>"))

    if show_daily and not daily_f.empty:
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["weight"],
            mode="lines+markers", name="Daily",
            line=dict(color=PC["green"], width=1.8),
            marker=dict(size=4, color=PC["green"]),
            hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>Daily</extra>"))

    if show_smooth and not daily_f.empty:
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["ma"], mode="lines",
            name=f"MA {ma_window}g",
            line=dict(color=PC["blue"], width=2.5),
            hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>MA</extra>"))

    # forecast
    if model.get("ok") and len(daily_f) >= 2:
        last_daily_dt = daily_f["date"].max().normalize()
        fd, fv = forecast_series(model, last_daily_dt + pd.Timedelta(days=1), int(forecast_horizon))
        if fv:
            fig.add_trace(go.Scatter(
                x=fd, y=fv, mode="lines", name=f"Forecast {forecast_horizon}g",
                line=dict(dash="dash", color=PC["amber"], width=2),
                hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>Forecast</extra>"))

    # marker prossimo sabato
    if pred_next_sat:
        fig.add_trace(go.Scatter(
            x=[next_sat_ts.to_pydatetime()], y=[pred_next_sat],
            mode="markers+text", name=f"Sab {next_sat.strftime('%d %b')}",
            marker=dict(size=11, color=PC["amber"], symbol="diamond",
                        line=dict(width=2, color="white")),
            text=[f"{pred_next_sat:.2f}"], textposition="top center",
            textfont=dict(color=PC["amber"], size=11, family="JetBrains Mono"),
            hovertemplate=f"Sabato {next_sat.strftime('%d %b')}<br>%{{y:.2f}} kg<extra></extra>"))

    # linee verticali
    add_vline(fig, last_dt, last_dt.strftime("%d %b"), color="#cdd4e0")
    if target_date_est:
        add_vline(fig, ts_midnight(target_date_est),
                  f"target {target_date_est.strftime('%d %b')}", color=PC["green"])

    # linea target
    fig.add_hline(y=float(target_weight),
                  line_dash="dot", line_color=PC["red"], line_width=1.5,
                  annotation_text="🎯 Target",
                  annotation_font_color=PC["red"],
                  annotation_position="bottom right")

    fig.update_layout(**BASE_LAYOUT, height=420,
                      xaxis_title=None, yaxis_title="Peso (kg)")
    st.plotly_chart(fig, use_container_width=True)

    # ── grafico weekday shape ──
    if model.get("ok"):
        st.markdown(section_html("🗓", "Pattern infrasettimanale (shape appresa)"), unsafe_allow_html=True)
        wds = model["weekday_shape"]
        giorni = ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"]
        colors = [PC["red"] if wds[i] > 0.3 else PC["amber"] if wds[i] > 0.1
                  else PC["green"] for i in range(7)]
        fig_shape = go.Figure(go.Bar(
            x=giorni, y=[wds[i] for i in range(7)],
            marker_color=colors,
            text=[f"+{wds[i]:.2f}" for i in range(7)],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=10),
            hovertemplate="<b>%{x}</b><br>+%{y:.3f} kg vs anchor<extra></extra>"))
        fig_shape.update_layout(**BASE_LAYOUT, height=220,
                                yaxis_title="kg sopra l'anchor settimanale",
                                showlegend=False, margin=dict(l=10,r=10,t=10,b=10))
        fig_shape.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_shape, use_container_width=True)

    # ── tendenza settimanale ──
    week_df_v = model.get("week_df", pd.DataFrame()) if model.get("ok") else pd.DataFrame()
    if not week_df_v.empty:
        st.markdown(section_html("📊", "Tendenza settimanale"), unsafe_allow_html=True)

        km1, km2, km3 = st.columns(3)
        km1.metric("Media 3 mesi", f"−{model['recent_avg_weekly_loss']:.2f} kg/sett")
        km2.metric("Mediana pesata (forecast)", f"−{model['robust_recent_loss']:.2f} kg/sett")
        last5 = float(week_df_v["weekly_loss_abs"].dropna().tail(5).mean())
        km3.metric("Ultime 5 settimane", f"−{last5:.2f} kg/sett" if np.isfinite(last5) else "—")

        # doppio grafico: anchor + delta settimanale
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.65, 0.35],
                             vertical_spacing=0.06)

        fig2.add_trace(go.Scatter(
            x=week_df_v["week_saturday"], y=week_df_v["anchor_weight"],
            mode="lines+markers", name="Anchor",
            line=dict(color=PC["green"], width=2),
            marker=dict(size=6, color=PC["green"],
                        line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>Anchor</extra>"),
            row=1, col=1)

        # trend lineare
        wc = week_df_v.dropna(subset=["anchor_weight"])
        if len(wc) >= 3:
            xn = (wc["week_saturday"] - wc["week_saturday"].min()).dt.days.values
            cf = np.polyfit(xn, wc["anchor_weight"].values, 1)
            fig2.add_trace(go.Scatter(
                x=wc["week_saturday"], y=np.polyval(cf, xn),
                mode="lines", name="Trend",
                line=dict(dash="dash", color=PC["amber"], width=1.5),
                hoverinfo="skip"), row=1, col=1)

        # barre perdita settimanale
        bar_colors = [PC["green"] if v >= 0.3 else PC["amber"] if v >= 0.1
                      else PC["red"]
                      for v in week_df_v["weekly_loss_abs"].fillna(0)]
        fig2.add_trace(go.Bar(
            x=week_df_v["week_saturday"],
            y=week_df_v["weekly_loss_abs"],
            name="Δ settimana",
            marker_color=bar_colors,
            hovertemplate="<b>%{x|%d %b}</b><br>−%{y:.2f} kg<extra>Δ sett.</extra>"),
            row=2, col=1)

        fig2.update_layout(**BASE_LAYOUT, height=420,
                           xaxis2_title="Settimana (sabato)",
                           yaxis_title="Peso anchor (kg)",
                           yaxis2_title="Δ kg")
        st.plotly_chart(fig2, use_container_width=True)

        # tabella
        tbl = week_df_v[week_df_v["week_saturday"] >= pd.Timestamp("2025-12-01")].copy()
        tbl["Settimana"]       = tbl["week_saturday"].dt.strftime("%d %b %Y")
        tbl["Peso anchor (kg)"]= tbl["anchor_weight"].map(lambda x: f"{x:.2f}")
        tbl["Δ (kg)"]          = tbl["weekly_loss_abs"].map(lambda x: f"−{x:.2f}" if pd.notna(x) else "")
        st.dataframe(tbl[["Settimana","Peso anchor (kg)","Δ (kg)"]],
                     use_container_width=True, hide_index=True)

    st.markdown(section_html("🕐", "Ultime 10 misure"), unsafe_allow_html=True)
    last10 = df_f.sort_values("date", ascending=False).head(10).copy()
    last10["Data"]     = last10["date"].dt.strftime("%Y-%m-%d  %H:%M")
    last10["Peso (kg)"]= last10["weight"].map(lambda x: f"{x:.2f}")
    last10["BMI"]      = last10["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    last10["Origine"]  = last10["source"]
    st.dataframe(last10[["Data","Peso (kg)","BMI","Origine"]],
                 use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────
# TAB — MANUALE
# ───────────────────────────────────────────────────────────────────
with tab_manual:
    st.markdown(section_html("✍️","Inserimento manuale"), unsafe_allow_html=True)
    st.caption("Se BMI = 0 viene calcolato automaticamente dall'altezza in sidebar.")

    with st.form("manual_form", clear_on_submit=True):
        cA, cB, cC, cD = st.columns(4)
        m_date   = cA.date_input("Data",      value=date.today())
        m_time   = cB.time_input("Ora",       value=datetime.now().time().replace(second=0,microsecond=0))
        m_weight = cC.number_input("Peso (kg)", min_value=0.0, value=float(last_w), step=0.1)
        m_bmi    = cD.number_input("BMI (0=auto)", min_value=0.0, value=0.0, step=0.1)
        if st.form_submit_button("✅ Salva misura", use_container_width=True):
            try:
                dt = pd.Timestamp(datetime.combine(m_date, m_time))
                bv = float(m_bmi) if float(m_bmi) > 0 else (float(m_weight)/(height_m**2) if height_m>0 else np.nan)
                insert_manual_entry(dt, float(m_weight), bv)
                st.success("✓ Misura salvata.")
                st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")

    st.divider()
    st.markdown(section_html("🗑️","Gestione archivio manuale"), unsafe_allow_html=True)
    manual_now = load_manual()

    if manual_now.empty:
        st.info("Nessuna misura manuale salvata.")
    else:
        tmp = manual_now.copy()
        tmp["label"] = (tmp["date"].dt.strftime("%Y-%m-%d %H:%M") + " │ "
                        + tmp["weight"].map(lambda x: f"{x:.2f} kg"))
        sel_labels = st.multiselect("Seleziona record da cancellare", tmp["label"].tolist())
        sel_ids    = tmp.loc[tmp["label"].isin(sel_labels), "id"].astype(int).tolist()

        col1, col2 = st.columns(2)
        if col1.button("🗑️ Cancella selezionate", use_container_width=True, type="primary"):
            if sel_ids:
                try: delete_manual_entries_by_id(sel_ids); st.success("Cancellate."); st.rerun()
                except Exception as e: st.error(f"Errore: {e}")
            else: st.warning("Seleziona almeno un record.")
        if col2.button("⚠️ Cancella TUTTO", use_container_width=True):
            try: clear_manual_entries(); st.success("Archivio azzerato."); st.rerun()
            except Exception as e: st.error(f"Errore: {e}")

        st.markdown(section_html("📋","Archivio"), unsafe_allow_html=True)
        sm = tmp.sort_values("date", ascending=False).copy()
        sm["Data"]     = sm["date"].dt.strftime("%Y-%m-%d  %H:%M")
        sm["Peso (kg)"]= sm["weight"].map(lambda x: f"{x:.2f}")
        sm["BMI"]      = sm["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
        sm["Origine"]  = sm["source"]
        st.dataframe(sm[["Data","Peso (kg)","BMI","Origine"]], use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────────────────────────
# TAB — FORECAST
# ───────────────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown(section_html("🔮","Forecast sabati → target"), unsafe_allow_html=True)

    if not model.get("ok"):
        st.warning("Modello non disponibile (dati insufficienti).")
    elif not target_date_est:
        st.markdown(banner_html("⚠️","Target non raggiungibile",
                    f"Con il ritmo attuale (−{model.get('weekly_loss',0):.2f} kg/sett) "
                    f"il target non viene raggiunto entro 365 giorni.", amber=True),
                    unsafe_allow_html=True)
    else:
        start_sat = next_saturday(today)
        if target_date_est < start_sat:
            st.markdown(banner_html("🎉","Target già raggiunto!",
                f"Stimato entro {start_sat.strftime('%d %b')}"), unsafe_allow_html=True)
        else:
            st.markdown(banner_html("🏁",
                f"Target il {target_date_est.strftime('%d %b %Y')} ({days_to_target} giorni)",
                f"Ritmo forecast: −{model['weekly_loss']:.2f} kg/sett "
                f"(mediana pesata esponenzialmente)"), unsafe_allow_html=True)

            rows = []
            s = start_sat
            while s <= target_date_est:
                wp   = predict_weight(model, ts_midnight(s))
                dist = round(wp - float(target_weight), 2)
                pct  = max(0.0, 100.0 - dist / (last_w - float(target_weight)) * 100)
                rows.append({
                    "Sabato": s.strftime("%d %b %Y"),
                    "Peso previsto (kg)": round(wp, 2),
                    "Al target (kg)": f"{dist:+.2f}",
                    "Progresso %": f"{pct:.0f}%",
                })
                s += timedelta(days=7)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # mini grafico forecast con banda confidenza
            st.markdown(section_html("📈","Curva forecast verso il target"), unsafe_allow_html=True)
            fd, fv = forecast_series(model, pd.Timestamp(today), days_to_target + 14)
            if fv:
                fv_arr = np.array(fv)
                weekly_std = float(model["week_df"]["weekly_loss_abs"].std()) if model.get("week_df") is not None else 0.3
                x_days = np.arange(len(fv_arr))
                sigma  = weekly_std * np.sqrt(x_days / 7 + 1) * 0.5

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=list(fd) + list(fd)[::-1],
                    y=list(fv_arr - sigma) + list((fv_arr + sigma))[::-1],
                    fill="toself", mode="none",
                    fillcolor="rgba(224,123,32,0.10)",
                    name="Intervallo confidenza", showlegend=True))
                fig_fc.add_trace(go.Scatter(
                    x=fd, y=fv_arr, mode="lines", name="Forecast",
                    line=dict(color=PC["amber"], width=2.5),
                    hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra></extra>"))
                fig_fc.add_hline(y=float(target_weight),
                                 line_dash="dot", line_color=PC["red"],
                                 annotation_text="🎯 Target",
                                 annotation_font_color=PC["red"],
                                 annotation_position="bottom right")
                if target_date_est:
                    add_vline(fig_fc, ts_midnight(target_date_est),
                              target_date_est.strftime("%d %b"), color=PC["green"])
                fig_fc.update_layout(**BASE_LAYOUT, height=320, xaxis_title=None, yaxis_title="Peso (kg)")
                st.plotly_chart(fig_fc, use_container_width=True)

    st.caption(
        f"Modello v9 · mediana pesata: {model.get('robust_recent_loss', 0):.3f} kg/sett · "
        f"media 3m: {model.get('recent_avg_weekly_loss', 0):.3f} kg/sett · "
        f"clip [{WEEKLY_LOSS_MIN}, {WEEKLY_LOSS_MAX}] · "
        f"target confermato dopo {TARGET_CONFIRM_DAYS}gg consecutivi")

# ───────────────────────────────────────────────────────────────────
# TAB — DATASET
# ───────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown(section_html("🧾","Dataset completo"), unsafe_allow_html=True)
    out = df_f.sort_values("date", ascending=False).copy()
    out["Data"]     = out["date"].dt.strftime("%Y-%m-%d  %H:%M")
    out["Peso (kg)"]= out["weight"].map(lambda x: f"{x:.2f}")
    out["BMI"]      = out["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    out["Origine"]  = out["source"]
    st.dataframe(out[["Data","Peso (kg)","BMI","Origine"]],
                 use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Scarica CSV",
        data=out[["Data","Peso (kg)","BMI","Origine"]].to_csv(index=False).encode("utf-8"),
        file_name="renpho_export.csv", mime="text/csv", use_container_width=True)

    if model.get("ok") and model.get("week_df") is not None:
        wexp = model["week_df"].copy()
        wexp["Settimana"] = wexp["week_saturday"].dt.strftime("%Y-%m-%d")
        wexp["Anchor (kg)"] = wexp["anchor_weight"].map(lambda x: f"{x:.3f}")
        wexp["Δ (kg)"] = wexp["weekly_loss_abs"].map(lambda x: f"−{x:.3f}" if pd.notna(x) else "")
        c2.download_button("⬇️ Scarica settimane",
            data=wexp[["Settimana","Anchor (kg)","Δ (kg)"]].to_csv(index=False).encode("utf-8"),
            file_name="renpho_settimane.csv", mime="text/csv", use_container_width=True)
