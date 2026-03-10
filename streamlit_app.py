import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import Akima1DInterpolator
from supabase import create_client, Client
from datetime import datetime, timedelta, date, time as dtime

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Weight Monitor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --c-bg:      #f8f9fb;
  --c-surface: #ffffff;
  --c-border:  #e4e8ef;
  --c-border2: #cdd4e0;
  --c-text:    #0f1923;
  --c-text2:   #5a6478;
  --c-text3:   #8c96a8;
  --c-green:   #22a55b;
  --c-green-l: #e8f7ef;
  --c-green-d: #166b3c;
  --c-amber:   #e07b20;
  --c-amber-l: #fef3e2;
  --c-red:     #d63b3b;
  --c-blue:    #2563eb;
  --c-purple:  #7c3aed;
  --radius:    12px;
  --shadow:    0 1px 4px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
}

html, body, [class*="css"], .stApp { font-family: 'Outfit', sans-serif !important; }
.stApp { background: var(--c-bg) !important; }

section[data-testid="stSidebar"] {
  background: var(--c-surface) !important;
  border-right: 1px solid var(--c-border) !important;
}

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
  font-size: 10.5px !important; font-weight: 500 !important;
  letter-spacing: .08em; text-transform: uppercase; color: var(--c-text3) !important;
}
div[data-testid="stMetricValue"] {
  font-family: 'Outfit', sans-serif !important;
  font-size: 1.65rem !important; font-weight: 700 !important;
  color: var(--c-text) !important; line-height: 1.2;
}
div[data-testid="stMetricDelta"] svg { display: none; }
div[data-testid="stMetricDelta"] > div {
  font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important; font-weight: 500 !important;
  letter-spacing: .06em; text-transform: uppercase;
  color: var(--c-text3) !important; padding: 10px 16px !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--c-green) !important;
  border-bottom: 2px solid var(--c-green) !important;
}

hr { border-color: var(--c-border) !important; margin: 20px 0 !important; }

.stButton > button {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important; font-weight: 500 !important;
  letter-spacing: .05em; text-transform: uppercase;
  border-radius: 8px !important; transition: all .15s !important;
}
.stButton > button[kind="primary"] {
  background: var(--c-green) !important;
  border-color: var(--c-green) !important; color: white !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--c-border) !important;
  border-radius: var(--radius) !important; overflow: hidden;
}
div[data-testid="stCaptionContainer"] p {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 10.5px !important; color: var(--c-text3) !important;
}

/* Banner */
.wm-banner {
  background: linear-gradient(135deg,#f0faf4,#e8f7ef);
  border: 1px solid #b6e8cc; border-left: 4px solid var(--c-green);
  border-radius: var(--radius); padding: 16px 20px; margin: 12px 0;
  display: flex; align-items: center; gap: 12px;
}
.wm-banner-amber {
  background: linear-gradient(135deg,#fffaf0,var(--c-amber-l));
  border: 1px solid #f5d49a; border-left: 4px solid var(--c-amber);
}
.wm-banner-red {
  background: linear-gradient(135deg,#fff5f5,#fdeaea);
  border: 1px solid #f5b8b8; border-left: 4px solid var(--c-red);
}
.wm-banner-icon { font-size: 24px; }
.wm-banner-text { line-height: 1.4; }
.wm-banner-text strong { color: var(--c-green-d); font-weight: 700; font-size: 15px; }
.wm-banner-amber .wm-banner-text strong { color: #8c4a0a !important; }
.wm-banner-red   .wm-banner-text strong { color: #8b1a1a !important; }
.wm-banner-text span { font-size: 13px; color: var(--c-text2); }

/* Section heading */
.wm-section {
  display: flex; align-items: center; gap: 8px; margin: 22px 0 4px;
}
.wm-section-icon {
  width: 28px; height: 28px; background: var(--c-green-l);
  border-radius: 6px; display: flex; align-items: center;
  justify-content: center; font-size: 14px;
}
.wm-section-title { font-size: 14px; font-weight: 700; color: var(--c-text); }

/* Nota esplicativa sotto grafici */
.wm-nota {
  background: #f8f9fb;
  border: 1px solid var(--c-border);
  border-radius: 8px;
  padding: 12px 16px;
  margin: 6px 0 18px;
  font-size: 13px;
  color: var(--c-text2);
  line-height: 1.6;
}
.wm-nota b { color: var(--c-text); }
.wm-nota .wm-legend-dot {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 50%; margin-right: 4px; vertical-align: middle;
}

/* Progress bar */
.wm-progress-outer {
  background: var(--c-border); border-radius: 20px;
  height: 8px; overflow: hidden; margin-top: 6px;
}
.wm-progress-inner {
  height: 100%; border-radius: 20px;
  background: linear-gradient(90deg, var(--c-green), #56c97a);
}

/* Stima giornaliera box */
.wm-daily-box {
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: var(--radius); padding: 20px 24px;
  box-shadow: var(--shadow); margin: 8px 0;
}
.wm-daily-value {
  font-family: 'Outfit', sans-serif;
  font-size: 2.4rem; font-weight: 800; color: var(--c-text); line-height: 1;
}
.wm-daily-range {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; color: var(--c-text3); margin-top: 4px;
}
.wm-daily-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; letter-spacing: .08em;
  text-transform: uppercase; color: var(--c-text3); margin-bottom: 6px;
}

/* Glossario parametri */
.wm-glossario {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 10px; margin: 12px 0;
}
.wm-gloss-item {
  background: var(--c-surface); border: 1px solid var(--c-border);
  border-radius: 8px; padding: 12px 14px;
}
.wm-gloss-term {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; font-weight: 500; color: var(--c-green-d);
  text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px;
}
.wm-gloss-def { font-size: 13px; color: var(--c-text2); line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════
DEFAULT_BASELINE_DATE   = date(2026, 8, 1)
DEFAULT_BASELINE_WEIGHT = 112.0
DEFAULT_TARGET_WEIGHT   = 72.0
DEFAULT_HEIGHT_M        = 1.82

FORECAST_LOOKBACK_DAYS         = 120
PATTERN_WEEKS                  = 8
MAX_LOCAL_ANCHOR_DISTANCE_DAYS = 4
WEEKLY_LOSS_MIN                = 0.10
WEEKLY_LOSS_MAX                = 1.80
SHAPE_CLIP_MAX                 = 0.90
SHAPE_CLIP_MIN                 = -0.10
TARGET_CONFIRM_DAYS            = 3
PLATEAU_THRESHOLD_KG           = 0.15

PC = dict(
    green="#22a55b", green_l="#56c97a", amber="#e07b20",
    red="#d63b3b",   blue="#2563eb",    purple="#7c3aed",
    text="#0f1923",  text2="#5a6478",   grid="#e4e8ef",
    bg="#ffffff",    surface="#f8f9fb", text3="#8c96a8",
)

BASE_LAYOUT = dict(
    paper_bgcolor=PC["bg"], plot_bgcolor=PC["bg"],
    font=dict(family="JetBrains Mono, monospace", size=11, color=PC["text2"]),
    xaxis=dict(gridcolor=PC["grid"], zeroline=False, showline=False,
               tickfont=dict(size=10, color=PC["text2"])),
    yaxis=dict(gridcolor=PC["grid"], zeroline=False, showline=False,
               tickfont=dict(size=10, color=PC["text2"])),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", bordercolor=PC["grid"],
                    font=dict(family="JetBrains Mono, monospace", size=12, color=PC["text"])),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11), bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
)
BASE_LAYOUT_NM   = {k: v for k, v in BASE_LAYOUT.items() if k != "margin"}
BASE_LAYOUT_BARE = {k: v for k, v in BASE_LAYOUT.items()
                    if k not in ("margin", "xaxis", "yaxis")}

_BASE_XAXIS = BASE_LAYOUT["xaxis"]
_BASE_YAXIS = BASE_LAYOUT["yaxis"]

def layout_kw(height, margin=None, xaxis_extra=None, yaxis_extra=None,
              xaxis_title=None, yaxis_title=None, **extra):
    """Costruisce kwargs per update_layout senza duplicare xaxis/yaxis."""
    m  = margin if margin is not None else dict(l=10, r=10, t=36, b=10)
    xa = dict(**_BASE_XAXIS)
    ya = dict(**_BASE_YAXIS)
    if xaxis_extra:  xa.update(xaxis_extra)
    if yaxis_extra:  ya.update(yaxis_extra)
    if xaxis_title is not None: xa["title"] = xaxis_title
    if yaxis_title is not None: ya["title"] = yaxis_title
    return {**BASE_LAYOUT_BARE, "height": height, "margin": m,
            "xaxis": xa, "yaxis": ya, **extra}

# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def ts_midnight(d):
    return pd.Timestamp(datetime.combine(d, dtime.min))

def fmt_delta(v, dec=2):
    if v is None or not np.isfinite(float(v)): return None
    return f"{float(v):+.{dec}f}"

def next_saturday(d):
    days = (5 - d.weekday()) % 7
    r = d + timedelta(days=days)
    return r if r > d else r + timedelta(days=7)

def week_saturday(ts):
    ts = pd.to_datetime(ts).normalize()
    return ts - pd.Timedelta(days=(ts.dayofweek - 5) % 7)

def add_vline(fig, x_dt, label, color="#cdd4e0"):
    xp = pd.to_datetime(x_dt).to_pydatetime()
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=xp, x1=xp, y0=0, y1=1,
                  line=dict(dash="dot", color=color, width=1.2))
    fig.add_annotation(x=xp, y=0.97, xref="x", yref="paper",
                       text=label, showarrow=False,
                       xanchor="left", yanchor="top",
                       font=dict(size=10, color=color))

def progress_html(pct):
    pct = max(0.0, min(100.0, pct))
    return (f'<div class="wm-progress-outer">'
            f'<div class="wm-progress-inner" style="width:{pct:.1f}%"></div></div>')

def banner_html(icon, strong, sub, style=""):
    cls = {"amber": "wm-banner-amber", "red": "wm-banner-red"}.get(style, "")
    return (f'<div class="wm-banner {cls}"><div class="wm-banner-icon">{icon}</div>'
            f'<div class="wm-banner-text"><strong>{strong}</strong><br>'
            f'<span>{sub}</span></div></div>')

def section_html(icon, title):
    return (f'<div class="wm-section"><div class="wm-section-icon">{icon}</div>'
            f'<div class="wm-section-title">{title}</div></div>')

def nota_html(testo):
    """Box esplicativo grigio sotto un grafico."""
    return f'<div class="wm-nota">{testo}</div>'

def dot(color):
    return f'<span class="wm-legend-dot" style="background:{color}"></span>'

def glossario_html(voci):
    """voci = lista di (termine, definizione)"""
    items = "".join(
        f'<div class="wm-gloss-item">'
        f'<div class="wm-gloss-term">{t}</div>'
        f'<div class="wm-gloss-def">{d}</div>'
        f'</div>'
        for t, d in voci
    )
    return f'<div class="wm-glossario">{items}</div>'

# ═══════════════════════════════════════════════════════════════════
# SERIE GIORNALIERA  — interpolazione Akima
# ═══════════════════════════════════════════════════════════════════

def build_full_daily_series(daily_df):
    d = daily_df.copy().sort_values("date")
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    s = d.set_index("date")["weight"].astype(float)
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    known = s.dropna()
    if len(known) < 4:
        return s.interpolate(method="time", limit_area="inside")
    x_known = np.array([(dd - known.index[0]).days for dd in known.index], dtype=float)
    y_known = known.values.astype(float)
    x_all   = np.array([(dd - known.index[0]).days for dd in full_idx], dtype=float)
    result  = s.copy()
    try:
        akima = Akima1DInterpolator(x_known, y_known)
        interpolated = akima(x_all)
        missing = s.isna()
        for i, (idx, is_missing) in enumerate(zip(full_idx, missing)):
            if not is_missing: continue
            pos = np.searchsorted(known.index, idx)
            dist_left  = (idx - known.index[pos-1]).days if pos > 0 else 9999
            dist_right = (known.index[pos] - idx).days if pos < len(known) else 9999
            if min(dist_left, dist_right) <= 14:
                result.iloc[i] = float(interpolated[i])
    except Exception:
        result = s.interpolate(method="time", limit=7, limit_area="inside")
    return result

# ═══════════════════════════════════════════════════════════════════
# ANCHOR LOCALE  (WLS)
# ═══════════════════════════════════════════════════════════════════

def local_anchor(series, target, max_dist=MAX_LOCAL_ANCHOR_DISTANCE_DAYS):
    target = pd.to_datetime(target).normalize()
    if target in series.index and pd.notna(series.loc[target]):
        return float(series.loc[target])
    s = series.dropna()
    if s.empty: return None
    deltas = ((s.index - target) / pd.Timedelta(days=1)).astype(float)
    mask = np.abs(deltas) <= max_dist
    local = s[mask]
    if local.empty: return None
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
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_data(ttl=60, show_spinner=False)
def load_manual():
    res = get_supabase().table("manual_entries").select("*").order("date").execute()
    rows = res.data or []
    if not rows:
        return pd.DataFrame(columns=["id","date","weight","bmi","source"])
    df = pd.DataFrame(rows)
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["bmi"]    = pd.to_numeric(df["bmi"], errors="coerce")
    df["source"] = df.get("source", "manual").fillna("manual")
    return df.dropna(subset=["date","weight"]).sort_values("date").reset_index(drop=True)

def insert_manual_entry(dt, weight, bmi):
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

def delete_manual_entries_by_id(ids):
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
def load_renpho_csv(url):
    raw = pd.read_csv(url, header=None)
    if raw.shape[1] == 1:
        s = raw[0].astype(str).str.strip().str.replace(r'^"|"$',"",regex=True)
        raw = s.str.split(",", expand=True)
    if raw.shape[1] < 3:
        raise ValueError("CSV non riconosciuto.")
    df = pd.DataFrame()
    df["date"] = pd.to_datetime(
        raw[0].astype(str).str.strip()+" "+raw[1].astype(str).str.strip(),
        errors="coerce", dayfirst=True)
    w = (raw[2].astype(str).str.strip()
         .str.replace(",",".",regex=False)
         .str.replace("kg","",regex=False).str.strip())
    df["weight"] = pd.to_numeric(w, errors="coerce")
    df = df.dropna(subset=["date","weight"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df["source"]="renpho"; df["bmi"]=np.nan; df["id"]=np.nan
    return df.reset_index(drop=True)

def combine_data(renpho, manual):
    if "id" not in renpho.columns:
        renpho = renpho.copy(); renpho["id"] = np.nan
    if manual.empty:
        return renpho.sort_values("date").reset_index(drop=True)
    df = pd.concat([renpho, manual], ignore_index=True)
    df["__p"] = df["source"].map({"renpho":0,"manual":1}).fillna(0)
    df = (df.sort_values(["date","__p"]).drop(columns=["__p"])
            .drop_duplicates(subset=["date"], keep="last"))
    return df.sort_values("date").reset_index(drop=True)

def add_bmi(df, h):
    if h <= 0: return df
    df = df.copy()
    m = df["bmi"].isna() & df["weight"].notna()
    df.loc[m,"bmi"] = df.loc[m,"weight"] / (h**2)
    return df

def make_daily(df):
    d = df.copy(); d["day"] = d["date"].dt.date
    out = d.groupby("day", as_index=False).agg(
        date=("date","max"), weight=("weight","median"), bmi=("bmi","median"))
    out["source"] = "daily"
    return out.sort_values("date").reset_index(drop=True)

def last_measurement(df):
    df = df.sort_values("date")
    ld = df["date"].dt.date.max()
    dd = df[df["date"].dt.date == ld]
    m  = dd[dd["source"]=="manual"]
    return (m if not m.empty else dd).sort_values("date").iloc[-1]

# ═══════════════════════════════════════════════════════════════════
# MODELLO FORECAST v10
# ═══════════════════════════════════════════════════════════════════

def _huber_weighted_regression(x, y, w, max_iter=20, k=1.345):
    x = np.asarray(x,float); y = np.asarray(y,float); w = np.asarray(w,float)
    X = np.vstack([np.ones(len(x)), x]).T
    try:
        beta = np.linalg.lstsq(X * w[:,None], y * w, rcond=None)[0]
    except Exception:
        return float(np.mean(y)), 0.0
    for _ in range(max_iter):
        resid  = y - X @ beta
        scale  = np.median(np.abs(resid)) / 0.6745 + 1e-9
        u      = resid / (scale * k)
        hw     = np.where(np.abs(u) <= 1, 1.0, 1.0 / np.abs(u))
        cw     = w * hw
        try:
            beta = np.linalg.lstsq(X * cw[:,None], y * cw, rcond=None)[0]
        except Exception:
            break
    return float(beta[0]), float(beta[1])


@st.cache_data(ttl=300, show_spinner="Calcolo modello v10…")
def fit_forecast_model(daily_df, lookback_days=FORECAST_LOOKBACK_DAYS,
                       smoothing_window=5, pattern_weeks=PATTERN_WEEKS):
    if daily_df.empty or len(daily_df) < 14:
        return {"ok": False, "reason": "too_few_points"}

    df = daily_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    end = df["date"].max()
    sub = df[df["date"] >= end - pd.Timedelta(days=lookback_days)].copy()
    if len(sub) < 14: sub = df.copy()
    sub = sub.sort_values("date").reset_index(drop=True)

    daily_full = build_full_daily_series(sub)
    daily_smooth = (daily_full
                    .interpolate(method="time", limit_direction="both")
                    .rolling(window=smoothing_window, min_periods=1).mean())

    sat_idx = pd.date_range(week_saturday(sub["date"].min()),
                            week_saturday(sub["date"].max()), freq="7D")
    week_rows = []
    for sat in sat_idx:
        est = local_anchor(daily_full, sat)
        if est is not None and np.isfinite(est):
            week_rows.append({"week_saturday": pd.to_datetime(sat),
                               "anchor_weight": float(est)})

    week_df = pd.DataFrame(week_rows).sort_values("week_saturday").reset_index(drop=True)
    if len(week_df) < 4:
        return {"ok": False, "reason": "not_enough_weeks"}

    week_df["weekly_delta"] = week_df["anchor_weight"].diff()

    wc    = week_df.dropna(subset=["anchor_weight"])
    x_num = (wc["week_saturday"] - wc["week_saturday"].min()).dt.days.values.astype(float)
    y_num = wc["anchor_weight"].values.astype(float)
    exp_w = np.exp(np.linspace(0, 1.2, len(x_num)))
    intercept_huber, slope_huber = _huber_weighted_regression(x_num, y_num, exp_w)
    weekly_loss_huber = float(-slope_huber * 7)

    losses_raw = week_df["weekly_delta"].dropna().values.astype(float)
    losses = np.clip(losses_raw, -2.0, 0.0)
    n = len(losses)
    if n > 0:
        ew    = np.exp(np.linspace(0, 1.1, n))
        order = np.argsort(losses)
        cumw  = np.cumsum(ew[order]) / ew.sum()
        weighted_median_loss = float(abs(np.interp(0.5, cumw, losses[order])))
    else:
        weighted_median_loss = 0.3

    recent_avg = float(np.mean(np.abs(losses))) if n > 0 else 0.3

    if WEEKLY_LOSS_MIN <= weekly_loss_huber <= WEEKLY_LOSS_MAX:
        weekly_loss = weekly_loss_huber
    else:
        weekly_loss = float(np.clip(weighted_median_loss, WEEKLY_LOSS_MIN, WEEKLY_LOSS_MAX))

    hist = (pd.DataFrame({"date": pd.to_datetime(daily_full.index),
                           "weight": daily_full.values}).dropna())
    hist["weekday"]       = hist["date"].dt.dayofweek
    hist["week_saturday"] = hist["date"].apply(week_saturday)

    recent_weeks = set(week_df.tail(pattern_weeks)["week_saturday"].tolist())
    anchor_map   = week_df.set_index("week_saturday")["anchor_weight"].to_dict()

    rel_rows = []
    for _, row in hist[hist["week_saturday"].isin(recent_weeks)].iterrows():
        aw = anchor_map.get(pd.to_datetime(row["week_saturday"]))
        if aw is not None and pd.notna(row["weight"]):
            rel_rows.append({"weekday": int(row["weekday"]),
                              "rel": float(row["weight"] - aw)})

    default_shape = {0:.40, 1:.28, 2:.18, 3:.10, 4:.04, 5:.00, 6:.08}
    default_mad   = {i: 0.20 for i in range(7)}

    rel_df = pd.DataFrame(rel_rows)
    if rel_df.empty:
        weekday_shape = default_shape.copy()
        weekday_mad   = default_mad.copy()
    else:
        weekday_shape = rel_df.groupby("weekday")["rel"].median().to_dict()
        weekday_mad   = rel_df.groupby("weekday")["rel"].apply(
            lambda x: float(np.median(np.abs(x - np.median(x))))).to_dict()
        for wd in range(7):
            weekday_shape.setdefault(wd, default_shape[wd])
            weekday_mad.setdefault(wd, default_mad[wd])

    weekday_shape = {k: float(np.clip(v, SHAPE_CLIP_MIN, SHAPE_CLIP_MAX))
                     for k,v in weekday_shape.items()}
    weekday_shape[5] = max(weekday_shape[5], SHAPE_CLIP_MIN)
    weekday_shape[6] = max(weekday_shape[6], weekday_shape[5])
    for prev, cur in [(4,5),(3,4),(2,3),(1,2),(0,1)]:
        weekday_shape[prev] = max(weekday_shape[prev], weekday_shape[cur])

    weekday_mad = {k: max(float(v), 0.10) for k,v in weekday_mad.items()}

    last_sat    = pd.to_datetime(week_df["week_saturday"].iloc[-1]).normalize()
    last_anchor = float(week_df["anchor_weight"].iloc[-1])
    last_date   = pd.to_datetime(sub["date"].max()).normalize()
    last_w      = float(sub.loc[sub["date"]==last_date,"weight"].iloc[-1])

    week_df["weekly_loss_abs"] = week_df["weekly_delta"].clip(upper=0).abs()

    backtest_rows = []
    for _, row in week_df.tail(9).iloc[:-1].iterrows():
        sat = pd.to_datetime(row["week_saturday"]).normalize()
        prev_rows = week_df[week_df["week_saturday"] < row["week_saturday"]]
        if len(prev_rows) < 2: continue
        prev_anchor = float(prev_rows["anchor_weight"].iloc[-1])
        fc = prev_anchor - weekly_loss
        backtest_rows.append({
            "week_saturday": sat,
            "actual": float(row["anchor_weight"]),
            "forecast": fc,
            "error": fc - float(row["anchor_weight"]),
        })
    backtest_df = pd.DataFrame(backtest_rows)

    return {
        "ok": True, "type": "v10",
        "daily_full": daily_full, "daily_smooth": daily_smooth,
        "last_date": last_date, "last_weight": last_w,
        "last_sat": last_sat, "last_anchor": last_anchor,
        "weekday_shape": weekday_shape, "weekday_mad": weekday_mad,
        "weekly_loss": weekly_loss,
        "weekly_loss_huber": weekly_loss_huber,
        "robust_recent_loss": weighted_median_loss,
        "recent_avg_weekly_loss": recent_avg,
        "week_df": week_df, "backtest_df": backtest_df,
    }


def anchor_for_week(model, sat):
    """
    Stima il peso-anchor per il sabato `sat`.
    Parte dall'ultima misura REALE (non dall'anchor calcolato) e proietta
    in avanti usando il ritmo settimanale.
    Questo evita che il forecast parta da un valore teorico staccato dalla realtà.
    """
    sat   = pd.to_datetime(sat).normalize()
    # Punto di partenza: ultima misura reale + offset shape sabato
    # → è il "sabato equivalente" dell'ultima settimana con dati reali
    last_real_sat  = week_saturday(model["last_date"])
    last_real_weight = model["last_weight"]
    sat_shape      = model["weekday_shape"].get(5, 0.0)  # shape sabato
    last_wd        = int(pd.to_datetime(model["last_date"]).dayofweek)
    last_day_shape = model["weekday_shape"].get(last_wd, 0.0)
    # Stima il "sabato implicito" della settimana dell'ultima misura
    implied_anchor = last_real_weight - last_day_shape + sat_shape

    weeks = (sat - last_real_sat).days / 7
    if weeks <= 0:
        return float(implied_anchor)
    return float(implied_anchor - weeks * model["weekly_loss"])


def forecast_series(model, start, horizon):
    """
    Genera la serie di forecast giornaliero da `start` per `horizon` giorni.
    Giorni già misurati → valore reale. Giorni futuri → anchor + shape + incertezza.
    L'incertezza cresce con √giorni_futuri per riflettere l'accumulo di errore.
    """
    if not model.get("ok"): return pd.DatetimeIndex([]), [], [], []
    start = pd.to_datetime(start).normalize()
    dates = pd.date_range(start, start + pd.Timedelta(days=horizon), freq="D")
    raw   = model["daily_full"]
    ld    = model["last_date"]
    wds   = model["weekday_shape"]
    mad   = model["weekday_mad"]
    vals_c, vals_lo, vals_hi = [], [], []

    # MAE medio del backtest come floor dell'incertezza base
    bt = model.get("backtest_df", pd.DataFrame())
    base_sigma = float(bt["error"].abs().mean()) if not bt.empty else 0.30
    base_sigma = max(base_sigma, 0.15)   # minimo 150g

    for d in dates:
        if d <= ld and d in raw.index and pd.notna(raw.loc[d]):
            # Dato reale: mostra il valore misurato, nessuna incertezza
            v = float(raw.loc[d])
            vals_c.append(v); vals_lo.append(v); vals_hi.append(v)
        else:
            wd        = int(d.dayofweek)
            anchor    = anchor_for_week(model, week_saturday(d))
            center    = anchor + wds.get(wd, 0.0)
            # Incertezza: MAD del giorno × (1 + crescita temporale) + floor dal MAE storico
            days_out  = max(0, (d - ld).days)
            weeks_out = days_out / 7
            daily_mad = mad.get(wd, 0.20)
            sigma     = (daily_mad + base_sigma * 0.5) * (1.0 + 0.4 * np.sqrt(weeks_out))
            vals_c.append(center)
            vals_lo.append(center - 1.5 * sigma)
            vals_hi.append(center + 1.5 * sigma)
    return dates, vals_c, vals_lo, vals_hi


def predict_weight_with_interval(model, when):
    _, c, lo, hi = forecast_series(model, when, 0)
    if not c: return np.nan, np.nan, np.nan
    return float(c[0]), float(lo[0]), float(hi[0])


def estimate_target_date(model, target_weight, start_from, max_horizon=365):
    dates, vals_c, _, _ = forecast_series(model, pd.Timestamp(start_from), max_horizon)
    if not vals_c: return None, None
    consecutive = 0; first_under = None
    for d, w in zip(dates, vals_c):
        if w <= float(target_weight):
            if consecutive == 0: first_under = pd.to_datetime(d).date()
            consecutive += 1
            if consecutive >= TARGET_CONFIRM_DAYS:
                return first_under, (first_under - start_from).days
        else:
            consecutive = 0; first_under = None
    return None, None


def compute_rolling_speed(week_df, window=4):
    wdf = week_df.dropna(subset=["anchor_weight"]).copy().sort_values("week_saturday")
    wdf["speed"]      = wdf["anchor_weight"].diff(-1) * -1
    wdf["speed_roll"] = wdf["speed"].rolling(window, min_periods=2).mean()
    return wdf

def detect_plateaus(week_df, threshold=PLATEAU_THRESHOLD_KG):
    wdf = week_df.copy()
    wdf["is_plateau"] = wdf["weekly_loss_abs"].fillna(0) < threshold
    return wdf

def monthly_target_line(target_weight, baseline_weight, baseline_dt, target_dt):
    if target_dt is None: return [], []
    dates = pd.date_range(pd.Timestamp(baseline_dt), pd.Timestamp(target_dt), freq="W")
    total = (pd.Timestamp(target_dt) - pd.Timestamp(baseline_dt)).days
    if total <= 0: return [], []
    vals = [baseline_weight - (d - pd.Timestamp(baseline_dt)).days / total *
            (baseline_weight - target_weight) for d in dates]
    return list(dates), vals

# ═══════════════════════════════════════════════════════════════════
# SECRETS
# ═══════════════════════════════════════════════════════════════════
csv_url = st.secrets.get("CSV_URL","")
if not csv_url:
    st.error("⚠️ `CSV_URL` non impostato."); st.stop()
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
    show_raw         = st.toggle("Punti RAW",           value=True)
    show_daily       = st.toggle("Linea peso giornaliero", value=True)
    show_smooth_line = st.toggle("Media mobile (MA)",   value=True)
    show_band        = st.toggle("Banda variabilità",   value=True)
    show_target_line = st.toggle("Percorso ideale",     value=True)
    st.divider()
    if st.button("🔄 Refresh RENPHO", use_container_width=True):
        load_renpho_csv.clear(); st.rerun()

# ═══════════════════════════════════════════════════════════════════
# CARICAMENTO
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
date_range_sel = st.sidebar.date_input("📅 Intervallo", value=(min_date,max_date),
                                        min_value=min_date, max_value=max_date)
start_d, end_d = (date_range_sel if isinstance(date_range_sel,tuple) and len(date_range_sel)==2
                  else (min_date,max_date))

df_f    = df[(df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].copy()
daily_f = daily[(daily["date"].dt.date >= start_d) & (daily["date"].dt.date <= end_d)].copy()
if df_f.empty: st.warning("Intervallo vuoto."); st.stop()
daily_f["ma"] = daily_f["weight"].rolling(ma_window, min_periods=1).mean()

# ═══════════════════════════════════════════════════════════════════
# METRICHE
# ═══════════════════════════════════════════════════════════════════
today    = date.today()
next_sat = next_saturday(today)
tomorrow = today + timedelta(days=1)

last_row     = last_measurement(df_f)
last_dt      = pd.to_datetime(last_row["date"])
last_w       = float(last_row["weight"])
last_bmi_val = float(last_row["bmi"]) if pd.notna(last_row.get("bmi")) else np.nan

prev_df  = df_f[df_f["date"] < last_dt].sort_values("date")
prev_row = prev_df.iloc[-1] if not prev_df.empty else None
prev_w   = float(prev_row["weight"]) if prev_row is not None else None
prev_bmi = float(prev_row["bmi"]) if prev_row is not None and pd.notna(prev_row.get("bmi")) else None

delta_bmi  = (last_bmi_val-prev_bmi) if prev_bmi and np.isfinite(last_bmi_val) and np.isfinite(prev_bmi) else None
loss_base  = float(baseline_weight - last_w)
dist_target= float(last_w - float(target_weight))
delta_loss = float(loss_base-(baseline_weight-prev_w)) if prev_w else None
delta_dist = float(dist_target-(prev_w-float(target_weight))) if prev_w else None

total_journey = float(baseline_weight - float(target_weight))
progress_pct  = max(0.0, min(100.0, loss_base/total_journey*100)) if total_journey > 0 else 0.0

model = fit_forecast_model(daily, lookback_days=FORECAST_LOOKBACK_DAYS,
                           smoothing_window=forecast_smooth, pattern_weeks=PATTERN_WEEKS)

pred_tomorrow_c = pred_tomorrow_lo = pred_tomorrow_hi = np.nan
pred_next_sat_c = pred_next_sat_lo = pred_next_sat_hi = None
if model.get("ok"):
    pred_tomorrow_c, pred_tomorrow_lo, pred_tomorrow_hi = predict_weight_with_interval(
        model, ts_midnight(tomorrow))
    pred_next_sat_c, pred_next_sat_lo, pred_next_sat_hi = predict_weight_with_interval(
        model, ts_midnight(next_sat))

target_date_est, days_to_target = (
    estimate_target_date(model, float(target_weight), today)
    if model.get("ok") else (None, None))

week_df_m  = model.get("week_df", pd.DataFrame()) if model.get("ok") else pd.DataFrame()
speed_df   = compute_rolling_speed(week_df_m) if not week_df_m.empty else pd.DataFrame()
plateau_df = detect_plateaus(week_df_m)        if not week_df_m.empty else pd.DataFrame()

pace_1m = float(speed_df["speed"].tail(4).mean()) if not speed_df.empty else None
pace_3m = model.get("recent_avg_weekly_loss")    if model.get("ok") else None

if pace_1m is not None and pace_3m is not None and pace_3m > 0:
    pace_ratio = pace_1m / pace_3m
    pace_diff  = pace_1m - pace_3m           # positivo = più veloce, negativo = più lento
    pace_diff_str = f"{pace_diff:+.2f} kg/sett rispetto alla media 3 mesi"
    if pace_ratio >= 0.95:
        pace_semaforo, pace_style = "🟢", ""
        pace_label = (f"in linea: −{pace_1m:.2f} vs media −{pace_3m:.2f} kg/sett "
                      f"({pace_diff_str})")
    elif pace_ratio >= 0.70:
        pace_semaforo, pace_style = "🟡", "amber"
        pace_label = (f"rallentamento lieve: −{pace_1m:.2f} vs media −{pace_3m:.2f} kg/sett "
                      f"({pace_diff_str})")
    else:
        pace_semaforo, pace_style = "🔴", "red"
        pace_label = (f"rallentamento marcato: −{pace_1m:.2f} vs media −{pace_3m:.2f} kg/sett "
                      f"({pace_diff_str})")
else:
    pace_semaforo, pace_label, pace_style = "⚪","dati insufficienti",""
    pace_diff_str = ""

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="wm-title" style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">'
    '<h1 style="font-family:Outfit,sans-serif;font-size:2rem;font-weight:800;'
    'color:#0f1923;letter-spacing:-.03em;margin:0">⚖️ Weight Monitor</h1>'
    '<span style="font-family:JetBrains Mono,monospace;font-size:11px;font-weight:500;'
    'background:#e8f7ef;color:#166b3c;border:1px solid #b6e8cc;padding:2px 8px;'
    'border-radius:20px;letter-spacing:.06em">FORECAST v10</span>'
    '</div>'
    '<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#8c96a8;'
    'letter-spacing:.06em;margin-bottom:20px">RENPHO · Akima · Huber · MAD confidence</div>',
    unsafe_allow_html=True)

tab_dash, tab_manual, tab_forecast, tab_data = st.tabs(
    ["📊  Cruscotto","✍️  Manuale","🔮  Forecast","🧾  Dataset"])

# ═══════════════════════════════════════════════════════════════════
# TAB — CRUSCOTTO
# ═══════════════════════════════════════════════════════════════════
with tab_dash:

    # ── KPI row 1 ──
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("⚖️ Ultima misura",      f"{last_w:.2f} kg",
              last_dt.strftime("%d %b  %H:%M"), delta_color="off")
    c2.metric("📐 BMI",
              f"{last_bmi_val:.2f}" if np.isfinite(last_bmi_val) else "—",
              (fmt_delta(delta_bmi,2)+" vs misura prec.") if delta_bmi else "—",
              delta_color="inverse")
    c3.metric("📉 Perso dal baseline", f"{loss_base:+.2f} kg",
              (fmt_delta(delta_loss,2)+" kg vs misura prec.") if delta_loss else "—",
              delta_color="normal")
    c4.metric("🎯 Kg mancanti al target", f"{dist_target:+.2f} kg",
              (fmt_delta(delta_dist,2)+" kg vs misura prec.") if delta_dist else "—",
              delta_color="inverse")

    # ── Progress ──
    st.markdown(
        f"**Progresso verso il target** — <b>{progress_pct:.1f}%</b> completato&nbsp;&nbsp;"
        f"<span style='color:#8c96a8;font-size:12px'>"
        f"Sei a {last_w:.1f} kg, obiettivo {float(target_weight):.1f} kg, "
        f"partito da {baseline_weight:.1f} kg</span>",
        unsafe_allow_html=True)
    st.markdown(progress_html(progress_pct), unsafe_allow_html=True)

    # ── KPI row 2 ──
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    if np.isfinite(pred_tomorrow_c):
        k1.metric(f"🔭 Stima domani ({tomorrow.strftime('%d %b')})",
                  f"{pred_tomorrow_c:.2f} kg",
                  f"range realistico: {pred_tomorrow_lo:.2f}–{pred_tomorrow_hi:.2f} kg",
                  delta_color="off")
    if pred_next_sat_c:
        k2.metric(f"🔮 Sabato {next_sat.strftime('%d %b')}",
                  f"{pred_next_sat_c:.2f} kg",
                  f"range: {pred_next_sat_lo:.2f}–{pred_next_sat_hi:.2f} kg",
                  delta_color="off")
    if model.get("ok"):
        k3.metric("📈 Velocità perdita",
                  f"−{model['weekly_loss']:.2f} kg/sett",
                  f"{pace_semaforo} {pace_label}")
        k4.metric("📅 Arrivo stimato al target",
                  target_date_est.strftime("%d %b %Y") if target_date_est else "oltre 1 anno",
                  f"tra {days_to_target} giorni" if days_to_target else "—")

    # ── Banner ritmo ──
    if pace_1m is not None:
        trend_arrow = "↑ accelerando" if pace_1m > (pace_3m or 0) else "↓ rallentando"
        diff_str    = f"{pace_1m - pace_3m:+.2f} kg/sett" if pace_3m else ""
        st.markdown(banner_html(
            pace_semaforo,
            f"Ultimo mese −{pace_1m:.2f} kg/sett {trend_arrow} vs media 3 mesi −{pace_3m:.2f} kg/sett ({diff_str})",
            (f"Confronto: stai perdendo {'di più' if pace_1m > (pace_3m or 0) else 'di meno'} rispetto "
             f"alla tua media storica di 3 mesi. Il forecast usa −{model['weekly_loss']:.2f} kg/sett."),
            style=pace_style), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # GRAFICO PRINCIPALE
    # ══════════════════════════════════════════════════════
    st.markdown(section_html("📈","Andamento del peso nel tempo"), unsafe_allow_html=True)

    fig = go.Figure()

    # Percorso ideale
    if show_target_line and target_date_est:
        tl_dates, tl_vals = monthly_target_line(
            float(target_weight), baseline_weight,
            pd.Timestamp(baseline_date), pd.Timestamp(target_date_est))
        if tl_dates:
            fig.add_trace(go.Scatter(
                x=tl_dates, y=tl_vals, mode="lines",
                name="Percorso ideale verso il target",
                line=dict(dash="dot", color=PC["purple"], width=1.5),
                hovertemplate="<b>%{x|%d %b}</b><br>Dove dovresti essere: %{y:.2f} kg<extra>Ideale</extra>"))

    # Banda variabilità
    if show_band and not daily_f.empty and len(daily_f) >= 7:
        rs = daily_f["weight"].rolling(7, min_periods=3).std().fillna(0)
        fig.add_trace(go.Scatter(
            x=pd.concat([daily_f["date"], daily_f["date"].iloc[::-1]]),
            y=pd.concat([daily_f["weight"]+rs, (daily_f["weight"]-rs).iloc[::-1]]),
            fill="toself", mode="none",
            fillcolor="rgba(34,165,91,0.07)",
            name="Fascia di variabilità normale (±1σ)",
            hoverinfo="skip"))

    if show_raw:
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["weight"],
            mode="markers", name="Misure singole (RAW)",
            marker=dict(size=3.5, color=PC["text3"], opacity=0.40),
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>Peso misurato: %{y:.2f} kg<extra>Misura</extra>"))

    if show_daily and not daily_f.empty:
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["weight"],
            mode="lines+markers", name="Peso giornaliero (mediana del giorno)",
            line=dict(color=PC["green"], width=1.8),
            marker=dict(size=4, color=PC["green"]),
            hovertemplate="<b>%{x|%d %b}</b><br>Peso del giorno: %{y:.2f} kg<extra>Giornaliero</extra>"))

    if show_smooth_line and not daily_f.empty:
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["ma"],
            mode="lines", name=f"Media mobile {ma_window} giorni (tendenza)",
            line=dict(color=PC["blue"], width=2.5),
            hovertemplate=f"<b>%{{x|%d %b}}</b><br>Media {ma_window}gg: %{{y:.2f}} kg<extra>Tendenza</extra>"))

    # Forecast
    if model.get("ok") and len(daily_f) >= 2:
        last_daily_dt = daily_f["date"].max().normalize()
        fd, fv_c, fv_lo, fv_hi = forecast_series(
            model, last_daily_dt + pd.Timedelta(days=1), int(forecast_horizon))
        if fv_c:
            fig.add_trace(go.Scatter(
                x=list(fd)+list(fd)[::-1],
                y=list(fv_hi)+list(fv_lo)[::-1],
                fill="toself", mode="none",
                fillcolor="rgba(224,123,32,0.09)",
                name="Zona di incertezza del forecast",
                hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=fd, y=fv_c, mode="lines",
                name=f"Previsione ({forecast_horizon} giorni)",
                line=dict(dash="dash", color=PC["amber"], width=2),
                hovertemplate="<b>%{x|%d %b}</b><br>Peso previsto: %{y:.2f} kg<extra>Previsione</extra>"))

    # Marker domani
    if np.isfinite(pred_tomorrow_c):
        fig.add_trace(go.Scatter(
            x=[ts_midnight(tomorrow).to_pydatetime()], y=[pred_tomorrow_c],
            mode="markers+text",
            name=f"Domani {tomorrow.strftime('%d %b')} (stima)",
            marker=dict(size=10, color=PC["blue"], symbol="circle",
                        line=dict(width=2, color="white")),
            text=[f"{pred_tomorrow_c:.2f}"], textposition="top center",
            textfont=dict(color=PC["blue"], size=11, family="JetBrains Mono"),
            hovertemplate=f"Domani {tomorrow.strftime('%d %b')}<br>"
                          f"Stima: %{{y:.2f}} kg<br>"
                          f"Range: {pred_tomorrow_lo:.2f}–{pred_tomorrow_hi:.2f} kg<extra></extra>"))

    # Marker sabato
    if pred_next_sat_c:
        fig.add_trace(go.Scatter(
            x=[ts_midnight(next_sat).to_pydatetime()], y=[pred_next_sat_c],
            mode="markers+text",
            name=f"Sabato {next_sat.strftime('%d %b')} (stima)",
            marker=dict(size=11, color=PC["amber"], symbol="diamond",
                        line=dict(width=2, color="white")),
            text=[f"{pred_next_sat_c:.2f}"], textposition="top center",
            textfont=dict(color=PC["amber"], size=11, family="JetBrains Mono"),
            hovertemplate=f"Sabato {next_sat.strftime('%d %b')}<br>"
                          f"Stima: %{{y:.2f}} kg<br>"
                          f"Range: {pred_next_sat_lo:.2f}–{pred_next_sat_hi:.2f} kg<extra></extra>"))

    fig.add_hline(y=float(target_weight), line_dash="dot",
                  line_color=PC["red"], line_width=1.5,
                  annotation_text="🎯 Target",
                  annotation_font_color=PC["red"],
                  annotation_position="bottom right")
    add_vline(fig, last_dt, f"ultima misura ({last_dt.strftime('%d %b')})", color="#cdd4e0")
    if target_date_est:
        add_vline(fig, ts_midnight(target_date_est),
                  f"arrivo stimato {target_date_est.strftime('%d %b')}", color=PC["green"])

    fig.update_layout(**layout_kw(
        height=440, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Data", yaxis_title="Peso (kg)",
        yaxis_extra=dict(tickformat=".1f", ticksuffix=" kg")))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(nota_html(
        f"{dot(PC['green'])} <b>Peso giornaliero</b> — valore mediano di tutte le misure del giorno. "
        f"Se ti sei pesato più volte, prende il valore centrale.<br>"
        f"{dot(PC['blue'])} <b>Media mobile {ma_window}gg</b> — media degli ultimi {ma_window} giorni: "
        f"filtra l'oscillazione quotidiana e mostra la tendenza reale.<br>"
        f"{dot(PC['amber'])} <b>Previsione (linea tratteggiata)</b> — peso stimato dal modello per i prossimi giorni. "
        f"La zona arancione attorno indica l'<b>incertezza</b>: più vai avanti nel tempo, più si allarga.<br>"
        f"{dot(PC['purple'])} <b>Percorso ideale</b> — retta che va dal tuo peso di partenza al target: "
        f"se sei sopra stai perdendo meno del previsto, se sei sotto stai andando più veloce.<br>"
        f"🔵 <b>Cerchio blu</b> = stima di domani &nbsp;|&nbsp; 🔶 <b>Diamante arancione</b> = stima sabato prossimo"
    ), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # VELOCITÀ PERDITA ROLLING
    # ══════════════════════════════════════════════════════
    if not speed_df.empty:
        st.markdown(section_html("⚡","A che velocità stai perdendo peso?"), unsafe_allow_html=True)

        fig_speed = go.Figure()

        colors_speed = [
            PC["green"] if v >= 0.30 else PC["amber"] if v >= 0.10 else PC["red"]
            for v in speed_df["speed_roll"].fillna(0)
        ]
        fig_speed.add_trace(go.Bar(
            x=speed_df["week_saturday"],
            y=speed_df["speed_roll"],
            marker_color=colors_speed,
            name="Velocità media 4 settimane",
            hovertemplate=(
                "<b>Settimana del %{x|%d %b}</b><br>"
                "Hai perso in media <b>%{y:.2f} kg a settimana</b> "
                "nelle ultime 4 settimane<extra></extra>"
            )))

        fig_speed.add_trace(go.Scatter(
            x=speed_df["week_saturday"],
            y=speed_df["speed"],
            mode="lines+markers",
            name="Perdita di questa settimana",
            line=dict(color=PC["text3"], width=1.2, dash="dot"),
            marker=dict(size=5, color=PC["text3"]),
            hovertemplate=(
                "<b>Settimana del %{x|%d %b}</b><br>"
                "Questa settimana: %{y:.2f} kg persi<extra></extra>"
            )))

        if model.get("ok"):
            wl = model["weekly_loss"]
            fig_speed.add_hline(
                y=wl, line_dash="dash", line_color=PC["amber"], line_width=1.5,
                annotation_text=f"Ritmo usato per il forecast: −{wl:.2f} kg/sett",
                annotation_font_color=PC["amber"],
                annotation_position="bottom right")

        fig_speed.add_hline(y=0, line_color=PC["grid"], line_width=1)
        fig_speed.add_hrect(y0=0, y1=0.10, fillcolor="rgba(213,59,59,0.05)",
                             line_width=0, annotation_text="plateau",
                             annotation_font_size=10, annotation_font_color=PC["red"])
        fig_speed.add_hrect(y0=0.10, y1=0.30, fillcolor="rgba(224,123,32,0.05)",
                             line_width=0, annotation_text="lento",
                             annotation_font_size=10, annotation_font_color=PC["amber"])
        fig_speed.add_hrect(y0=0.30, y1=2.0, fillcolor="rgba(34,165,91,0.04)",
                             line_width=0, annotation_text="buon ritmo",
                             annotation_font_size=10, annotation_font_color=PC["green"])

        fig_speed.update_layout(**layout_kw(
            height=280, margin=dict(l=10,r=10,t=16,b=10),
            xaxis_title="Settimana", yaxis_title="Kg persi per settimana",
            yaxis_extra=dict(rangemode="tozero", tickformat=".2f", ticksuffix=" kg"),
            showlegend=True))
        st.plotly_chart(fig_speed, use_container_width=True)

        st.markdown(nota_html(
            "<b>Come leggere questo grafico:</b><br>"
            f"{dot(PC['green'])} <b>Verde</b> = stai perdendo più di 0.30 kg/settimana — ottimo ritmo.<br>"
            f"{dot(PC['amber'])} <b>Arancione</b> = tra 0.10 e 0.30 kg/settimana — ritmo lento ma presente.<br>"
            f"{dot(PC['red'])} <b>Rosso</b> = meno di 0.10 kg/settimana — plateau (il peso è praticamente fermo).<br>"
            "Le barre mostrano la <b>media delle ultime 4 settimane</b> (più stabile). "
            "La linea tratteggiata grigia è la perdita della singola settimana (più rumorosa).<br>"
            f"La linea arancione tratteggiata è la velocità che il modello usa per calcolare le previsioni."
        ), unsafe_allow_html=True)

    # ── Plateau ──
    if not plateau_df.empty:
        n_plateau      = int(plateau_df["is_plateau"].sum())
        recent_plateau = int(plateau_df.tail(4)["is_plateau"].sum())
        if n_plateau > 0:
            st.markdown(banner_html(
                "⚠️" if recent_plateau > 1 else "ℹ️",
                f"{n_plateau} settimane di plateau rilevate nello storico",
                f"Un plateau è una settimana con meno di {PLATEAU_THRESHOLD_KG} kg di perdita. "
                f"{recent_plateau} nelle ultime 4 settimane.",
                style="amber" if recent_plateau > 1 else ""),
                unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════
    # CONFRONTO FORECAST VS REALE (BACKTEST)
    # ══════════════════════════════════════════════════════
    bt_df = model.get("backtest_df", pd.DataFrame()) if model.get("ok") else pd.DataFrame()
    if not bt_df.empty:
        st.markdown(section_html("🎯","Il modello era accurato? Confronto previsione vs realtà"),
                    unsafe_allow_html=True)

        mae  = float(bt_df["error"].abs().mean())
        bias = float(bt_df["error"].mean())

        bc1, bc2 = st.columns(2)
        bc1.metric(
            "Errore medio (MAE)",
            f"±{mae:.2f} kg",
            f"In media il forecast sbagliava di {mae:.2f} kg")
        bc2.metric(
            "Tendenza dell'errore (Bias)",
            f"{bias:+.2f} kg",
            "tende a sovra-stimare" if bias > 0.05 else
            "tende a sotto-stimare" if bias < -0.05 else
            "nessuna tendenza sistematica")

        # Grafico 1: peso reale vs previsto
        fig_bt1 = go.Figure()
        fig_bt1.add_trace(go.Scatter(
            x=bt_df["week_saturday"], y=bt_df["actual"],
            mode="lines+markers", name="Peso reale (sabato)",
            line=dict(color=PC["green"], width=2.5),
            marker=dict(size=8, color=PC["green"], line=dict(width=2, color="white")),
            hovertemplate="<b>%{x|%d %b}</b><br>Peso reale: <b>%{y:.2f} kg</b><extra></extra>"))
        fig_bt1.add_trace(go.Scatter(
            x=bt_df["week_saturday"], y=bt_df["forecast"],
            mode="lines+markers", name="Peso previsto dal modello",
            line=dict(color=PC["amber"], width=2, dash="dash"),
            marker=dict(size=8, color=PC["amber"], symbol="diamond",
                        line=dict(width=2, color="white")),
            hovertemplate="<b>%{x|%d %b}</b><br>Peso previsto: <b>%{y:.2f} kg</b><extra></extra>"))
        fig_bt1.update_layout(**layout_kw(
            height=240, margin=dict(l=10, r=10, t=36, b=10),
            yaxis_title="Peso (kg)",
            yaxis_extra=dict(tickformat=".1f", ticksuffix=" kg"),
            title=dict(text="Peso reale vs peso previsto",
                       font=dict(size=12, color=PC["text2"]), x=0)))
        st.plotly_chart(fig_bt1, use_container_width=True)

        # Grafico 2: scarto (errore)
        error_colors = [PC["red"] if e > 0.1 else PC["green"] if e < -0.1 else PC["text3"]
                        for e in bt_df["error"]]
        fig_bt2 = go.Figure()
        fig_bt2.add_trace(go.Bar(
            x=bt_df["week_saturday"], y=bt_df["error"],
            name="Scarto (previsione − reale)",
            marker_color=error_colors,
            hovertemplate=(
                "<b>%{x|%d %b}</b><br>"
                "Scarto: <b>%{y:+.2f} kg</b><br>"
                "<i>Rosso = previsto troppo alto (hai fatto meglio)</i><br>"
                "<i>Verde = previsto troppo basso (hai perso meno del previsto)</i><extra></extra>"
            )))
        fig_bt2.add_hline(y=0, line_color=PC["grid"], line_width=1.5)
        fig_bt2.update_layout(**layout_kw(
            height=180, margin=dict(l=10, r=10, t=36, b=10),
            xaxis_title="Settimana (sabato)", yaxis_title="Scarto (kg)",
            yaxis_extra=dict(tickformat="+.2f", ticksuffix=" kg",
                             zeroline=True, zerolinecolor=PC["grid"]),
            title=dict(text="Scarto tra previsione e realtà (positivo = modello troppo ottimista)",
                       font=dict(size=12, color=PC["text2"]), x=0)))
        st.plotly_chart(fig_bt2, use_container_width=True)

        st.markdown(nota_html(
            "<b>Come leggere questo grafico:</b><br>"
            f"{dot(PC['green'])} <b>Linea verde</b> = peso reale misurato ogni sabato.<br>"
            f"{dot(PC['amber'])} <b>Linea arancione tratteggiata</b> = cosa prevedeva il modello per quel sabato "
            f"(calcolato partendo dal sabato precedente).<br>"
            "<b>Grafico in basso — Scarto:</b> mostra di quanto il modello ha sbagliato. "
            f"{dot(PC['red'])} <b>Rosso/sopra lo zero</b> = il modello ha previsto un peso più alto del reale "
            f"(hai fatto meglio del previsto). "
            f"{dot(PC['green'])} <b>Verde/sotto lo zero</b> = il modello ha previsto troppo basso "
            f"(hai perso meno del previsto).<br>"
            f"Un <b>MAE di {mae:.2f} kg</b> significa che in media le previsioni sbagliano di {mae:.2f} kg."
        ), unsafe_allow_html=True)

    # ── BMI ──
    if not daily_f.empty and daily_f["bmi"].notna().sum() >= 3:
        st.markdown(section_html("📐","Andamento del BMI nel tempo"), unsafe_allow_html=True)
        fig_bmi = go.Figure()
        bmi_data = daily_f.dropna(subset=["bmi"])
        fig_bmi.add_trace(go.Scatter(
            x=bmi_data["date"], y=bmi_data["bmi"],
            mode="lines+markers", name="BMI",
            line=dict(color=PC["purple"], width=2),
            marker=dict(size=4, color=PC["purple"]),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.05)",
            hovertemplate="<b>%{x|%d %b}</b><br>BMI: <b>%{y:.2f}</b><extra></extra>"))
        for y_val, label in [(18.5,"< 18.5 Sottopeso"),(25.0,"25–30 Sovrappeso"),(30.0,"≥ 30 Obeso")]:
            fig_bmi.add_hline(y=y_val, line_dash="dot",
                              line_color="rgba(0,0,0,0.12)", line_width=1,
                              annotation_text=label,
                              annotation_font=dict(size=9, color=PC["text3"]),
                              annotation_position="right")
        fig_bmi.update_layout(**layout_kw(
            height=250, margin=dict(l=10,r=10,t=16,b=10),
            xaxis_title="Data", yaxis_title="BMI",
            yaxis_extra=dict(tickformat=".1f")))
        st.plotly_chart(fig_bmi, use_container_width=True)
        st.markdown(nota_html(
            "<b>BMI (Indice di Massa Corporea)</b> = peso (kg) ÷ altezza² (m). "
            "Le linee tratteggiate indicano le soglie OMS: "
            "<b>sotto 18.5</b> = sottopeso, <b>18.5–25</b> = normopeso, "
            "<b>25–30</b> = sovrappeso, <b>sopra 30</b> = obesità."
        ), unsafe_allow_html=True)

    # ── Ultime 10 misure ──
    st.markdown(section_html("🕐","Ultime 10 misure registrate"), unsafe_allow_html=True)
    last10 = df_f.sort_values("date",ascending=False).head(10).copy()
    last10["Data"]      = last10["date"].dt.strftime("%Y-%m-%d  %H:%M")
    last10["Peso (kg)"] = last10["weight"].map(lambda x: f"{x:.2f}")
    last10["BMI"]       = last10["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    last10["Origine"]   = last10["source"]
    st.dataframe(last10[["Data","Peso (kg)","BMI","Origine"]],
                 use_container_width=True, hide_index=True)

    # ── Glossario parametri ──
    st.markdown(section_html("📖","Glossario — cosa significano i parametri"), unsafe_allow_html=True)
    st.markdown(glossario_html([
        ("Velocità perdita (Huber)",
         "Quanti kg perdi mediamente ogni settimana, calcolato con la regressione di Huber: "
         "un metodo statistico che riduce automaticamente il peso delle settimane anomale "
         "(malattia, vacanza, ritenzione) per dare una stima più stabile."),
        ("Mediana pesata",
         "Alternativa alla media: ordina tutte le settimane per perdita e prende quella centrale, "
         "dando più peso alle settimane recenti. Meno sensibile agli outlier rispetto alla media."),
        ("Anchor settimanale",
         "Il peso stimato per il sabato di ogni settimana. Serve come punto di riferimento: "
         "il modello calcola quanto pesi ogni altro giorno sommando un offset (shape) all'anchor."),
        ("Tolleranza ± / Range",
         "La forchetta realistica entro cui cadrà il tuo peso. Calcolata dalla MAD "
         "(deviazione assoluta mediana) storica di quel giorno della settimana. "
         "Si allarga man mano che il forecast va avanti nel tempo."),
        ("MAD (deviazione assoluta mediana)",
         "Misura quanto oscilla il tuo peso in un giorno specifico della settimana rispetto alla media. "
         "Ad esempio: se il lunedì sei sempre tra +0.2 e +0.5 kg sopra il minimo settimanale, "
         "la MAD del lunedì è circa 0.15 kg."),
        ("MAE backtest",
         "Errore Medio Assoluto: misura quanto il modello ha sbagliato nelle ultime 8 settimane. "
         "Un MAE di 0.30 kg significa che in media le previsioni erano sbagliate di 300 grammi. "
         "Più basso è, più il modello è affidabile."),
        ("Bias",
         "Se il modello sbaglia sempre nella stessa direzione (sempre troppo alto o sempre troppo basso). "
         "Un bias di +0.20 significa che il modello tende a prevedere un peso 200g più alto del reale."),
        ("Akima / interpolazione",
         "Metodo usato per stimare il peso nei giorni in cui non ti sei pesato. "
         "Più preciso dell'interpolazione lineare perché segue la curva naturale del tuo peso "
         "invece di tracciare rette tra i punti."),
    ]), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB — MANUALE
# ═══════════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown(section_html("✍️","Inserimento manuale"), unsafe_allow_html=True)
    st.caption("BMI = 0 → calcolato automaticamente dall'altezza impostata in sidebar.")

    with st.form("manual_form", clear_on_submit=True):
        cA,cB,cC,cD = st.columns(4)
        m_date   = cA.date_input("Data",           value=date.today())
        m_time   = cB.time_input("Ora",            value=datetime.now().time().replace(second=0,microsecond=0))
        m_weight = cC.number_input("Peso (kg)",    min_value=0.0, value=float(last_w), step=0.1)
        m_bmi    = cD.number_input("BMI (0=auto)", min_value=0.0, value=0.0, step=0.1)
        if st.form_submit_button("✅ Salva misura", use_container_width=True):
            try:
                dt = pd.Timestamp(datetime.combine(m_date, m_time))
                bv = float(m_bmi) if float(m_bmi) > 0 else (
                    float(m_weight)/(height_m**2) if height_m > 0 else np.nan)
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
        sel_labels = st.multiselect("Seleziona da cancellare", tmp["label"].tolist())
        sel_ids    = tmp.loc[tmp["label"].isin(sel_labels),"id"].astype(int).tolist()
        col1, col2 = st.columns(2)
        if col1.button("🗑️ Cancella selezionate", use_container_width=True, type="primary"):
            if sel_ids:
                try: delete_manual_entries_by_id(sel_ids); st.success("Cancellate."); st.rerun()
                except Exception as e: st.error(f"Errore: {e}")
            else: st.warning("Seleziona almeno un record.")
        if col2.button("⚠️ Cancella TUTTO", use_container_width=True):
            try: clear_manual_entries(); st.success("Azzerato."); st.rerun()
            except Exception as e: st.error(f"Errore: {e}")
        st.markdown(section_html("📋","Archivio"), unsafe_allow_html=True)
        sm = tmp.sort_values("date",ascending=False).copy()
        sm["Data"]      = sm["date"].dt.strftime("%Y-%m-%d  %H:%M")
        sm["Peso (kg)"] = sm["weight"].map(lambda x: f"{x:.2f}")
        sm["BMI"]       = sm["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
        sm["Origine"]   = sm["source"]
        st.dataframe(sm[["Data","Peso (kg)","BMI","Origine"]],
                     use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB — FORECAST
# ═══════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown(section_html("🔮","Previsione giornaliera e tabella sabati"), unsafe_allow_html=True)

    if not model.get("ok"):
        st.warning("Modello non disponibile (dati insufficienti).")
    else:
        # Stima domani
        if np.isfinite(pred_tomorrow_c):
            fa, fb, _ = st.columns([1,1,2])
            fa.markdown(
                f'<div class="wm-daily-box">'
                f'<div class="wm-daily-label">Stima peso domani · {tomorrow.strftime("%A %d %b")}</div>'
                f'<div class="wm-daily-value">{pred_tomorrow_c:.2f} kg</div>'
                f'<div class="wm-daily-range">Range realistico: {pred_tomorrow_lo:.2f} – {pred_tomorrow_hi:.2f} kg</div>'
                f'<div class="wm-daily-range">Tolleranza: ±{(pred_tomorrow_hi-pred_tomorrow_c):.2f} kg</div>'
                f'</div>', unsafe_allow_html=True)

            next7_dates, next7_c, next7_lo, next7_hi = forecast_series(
                model, ts_midnight(today), 7)
            next7_rows = []
            for d,c,lo,hi in zip(next7_dates,next7_c,next7_lo,next7_hi):
                d_date  = pd.to_datetime(d).date()
                is_real = d_date <= model["last_date"].date() if hasattr(model["last_date"],"date") else False
                next7_rows.append({
                    "Giorno":         pd.to_datetime(d).strftime("%a %d %b"),
                    "Peso stimato":   f"{c:.2f} kg",
                    "Range min–max":  f"{lo:.2f} – {hi:.2f} kg",
                    "Incertezza ±":   f"{hi-c:.2f} kg",
                    "Tipo":           "📍 dato reale" if is_real else "🔮 stima",
                })
            fb.markdown("**Prossimi 7 giorni**")
            fb.dataframe(pd.DataFrame(next7_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Curva forecast
        if target_date_est:
            st.markdown(banner_html("🏁",
                f"Arrivo stimato al target: {target_date_est.strftime('%d %b %Y')} (tra {days_to_target} giorni)",
                f"Velocità attuale: −{model['weekly_loss']:.2f} kg/settimana · "
                f"Mediana pesata storica: −{model['robust_recent_loss']:.2f} kg/sett"),
                unsafe_allow_html=True)

        horiz = max(days_to_target+14, int(forecast_horizon)) if days_to_target else int(forecast_horizon)
        fd, fv_c, fv_lo, fv_hi = forecast_series(model, pd.Timestamp(today), horiz)
        if fv_c:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=list(fd)+list(fd)[::-1],
                y=list(fv_hi)+list(fv_lo)[::-1],
                fill="toself", mode="none",
                fillcolor="rgba(224,123,32,0.10)",
                name="Zona di incertezza (±1.5 volte la variabilità storica)"))
            fig_fc.add_trace(go.Scatter(
                x=fd, y=fv_c, mode="lines",
                name="Previsione centrale",
                line=dict(color=PC["amber"], width=2.5),
                hovertemplate="<b>%{x|%d %b}</b><br>Peso previsto: <b>%{y:.2f} kg</b><extra></extra>"))
            fig_fc.add_hline(y=float(target_weight), line_dash="dot",
                             line_color=PC["red"], line_width=1.5,
                             annotation_text="🎯 Target",
                             annotation_font_color=PC["red"],
                             annotation_position="bottom right")
            if target_date_est:
                add_vline(fig_fc, ts_midnight(target_date_est),
                          f"arrivo stimato {target_date_est.strftime('%d %b')}", color=PC["green"])
            fig_fc.update_layout(**layout_kw(
                height=340, margin=dict(l=10,r=10,t=16,b=10),
                xaxis_title="Data", yaxis_title="Peso (kg)",
                yaxis_extra=dict(tickformat=".1f", ticksuffix=" kg")))
            st.plotly_chart(fig_fc, use_container_width=True)

            st.markdown(nota_html(
                f"La <b>linea arancione</b> è la previsione centrale. "
                f"La <b>zona arancione chiara</b> attorno è l'intervallo di incertezza: "
                f"il tuo peso reale dovrebbe cadere in quella fascia circa il 90% delle volte. "
                f"La fascia si allarga nel tempo perché l'incertezza cresce. "
                f"La <b>linea verde verticale</b> è la data stimata di arrivo al target. "
                f"Nota: questa è una stima basata sul ritmo attuale — può cambiare se la velocità cambia."
            ), unsafe_allow_html=True)

        # Tabella sabati
        if target_date_est:
            st.markdown(section_html("📅","Sabati previsti fino al target"), unsafe_allow_html=True)
            rows = []
            s = next_saturday(today)
            while s <= target_date_est:
                wp_c, wp_lo, wp_hi = predict_weight_with_interval(model, ts_midnight(s))
                dist = round(wp_c - float(target_weight), 2)
                pct  = max(0.0, 100.0 - dist/(last_w-float(target_weight))*100) if (last_w-float(target_weight)) > 0 else 0
                wl   = model["weekly_loss"]
                sem  = "🟢 buono" if wl >= 0.30 else "🟡 lento" if wl >= 0.10 else "🔴 plateau"
                rows.append({
                    "Sabato":             s.strftime("%d %b %Y"),
                    "Peso stimato (kg)":  f"{wp_c:.2f}",
                    "Range min–max":      f"{wp_lo:.2f} – {wp_hi:.2f}",
                    "Mancano al target":  f"{dist:+.2f} kg",
                    "Progresso":          f"{pct:.0f}%",
                    "Ritmo":              sem,
                })
                s += timedelta(days=7)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown(nota_html(
                "<b>Range min–max</b>: forchetta realistica in cui cadrà il tuo peso quel sabato. "
                "Più è ampia, più incertezza c'è (settimane lontane = range più largo). "
                "<b>Ritmo</b>: indica se la velocità di perdita attuale è buona 🟢, lenta 🟡, o ferma 🔴."
            ), unsafe_allow_html=True)

        st.caption(
            f"Modello v10 · Akima interpolation · Huber regression · "
            f"Velocità usata: {model['weekly_loss']:.3f} kg/sett · "
            f"Mediana pesata: {model['robust_recent_loss']:.3f} kg/sett · "
            f"Target confermato dopo {TARGET_CONFIRM_DAYS} giorni consecutivi sotto soglia")

# ═══════════════════════════════════════════════════════════════════
# TAB — DATASET
# ═══════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown(section_html("🧾","Dataset completo"), unsafe_allow_html=True)
    out = df_f.sort_values("date",ascending=False).copy()
    out["Data"]      = out["date"].dt.strftime("%Y-%m-%d  %H:%M")
    out["Peso (kg)"] = out["weight"].map(lambda x: f"{x:.2f}")
    out["BMI"]       = out["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    out["Origine"]   = out["source"]
    st.dataframe(out[["Data","Peso (kg)","BMI","Origine"]],
                 use_container_width=True, hide_index=True)

    c1,c2 = st.columns(2)
    c1.download_button("⬇️ Scarica CSV misure",
        data=out[["Data","Peso (kg)","BMI","Origine"]].to_csv(index=False).encode("utf-8"),
        file_name="renpho_export.csv", mime="text/csv", use_container_width=True)

    if model.get("ok") and model.get("week_df") is not None:
        wexp = model["week_df"].copy()
        wexp["Settimana"]   = wexp["week_saturday"].dt.strftime("%Y-%m-%d")
        wexp["Anchor (kg)"] = wexp["anchor_weight"].map(lambda x: f"{x:.3f}")
        wexp["Δ (kg)"]      = wexp["weekly_loss_abs"].map(lambda x: f"−{x:.3f}" if pd.notna(x) else "")
        c2.download_button("⬇️ Scarica andamento settimanale",
            data=wexp[["Settimana","Anchor (kg)","Δ (kg)"]].to_csv(index=False).encode("utf-8"),
            file_name="renpho_settimane.csv", mime="text/csv", use_container_width=True)
