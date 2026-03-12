import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Akima1DInterpolator
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from supabase import create_client, Client
from datetime import datetime, timedelta, date, time as dtime
import warnings
warnings.filterwarnings("ignore")

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
  border-radius: 8px !important;
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

.wm-banner {
  background: linear-gradient(135deg,#f0faf4,#e8f7ef);
  border: 1px solid #b6e8cc; border-left: 4px solid var(--c-green);
  border-radius: var(--radius); padding: 14px 18px; margin: 10px 0;
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
.wm-banner-icon { font-size: 22px; }
.wm-banner-text { line-height: 1.4; }
.wm-banner-text strong { color: var(--c-green-d); font-weight: 700; font-size: 14px; }
.wm-banner-amber .wm-banner-text strong { color: #8c4a0a !important; }
.wm-banner-red   .wm-banner-text strong { color: #8b1a1a !important; }
.wm-banner-text span { font-size: 13px; color: var(--c-text2); }

.wm-section {
  display: flex; align-items: center; gap: 8px; margin: 20px 0 6px;
}
.wm-section-icon {
  width: 26px; height: 26px; background: var(--c-green-l);
  border-radius: 6px; display: flex; align-items: center;
  justify-content: center; font-size: 13px;
}
.wm-section-title { font-size: 13px; font-weight: 700; color: var(--c-text); }

.wm-progress-outer {
  background: var(--c-border); border-radius: 20px;
  height: 7px; overflow: hidden; margin-top: 5px;
}
.wm-progress-inner {
  height: 100%; border-radius: 20px;
  background: linear-gradient(90deg, var(--c-green), #56c97a);
}

/* Tabella settimane custom */
.wm-week-table { width:100%; border-collapse:collapse; font-size:13px; }
.wm-week-table th {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 500; letter-spacing: .07em;
  text-transform: uppercase; color: var(--c-text3);
  padding: 8px 12px; border-bottom: 1px solid var(--c-border);
  text-align: right;
}
.wm-week-table th:first-child { text-align: left; }
.wm-week-table td {
  padding: 9px 12px; border-bottom: 1px solid var(--c-border);
  color: var(--c-text); text-align: right;
}
.wm-week-table td:first-child { text-align: left; color: var(--c-text2); }
.wm-week-table tr:last-child td { border-bottom: none; }
.wm-week-table tr:hover td { background: var(--c-bg); }
.wm-week-table .wm-loss-good { color: var(--c-green); font-weight: 600; }
.wm-week-table .wm-loss-slow { color: var(--c-amber); font-weight: 600; }
.wm-week-table .wm-loss-zero { color: var(--c-red); font-weight: 600; }
.wm-week-table .wm-fc-row td { color: var(--c-text3); font-style: italic; }
.wm-week-table .wm-fc-row td:first-child { color: var(--c-text3); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════════════
DEFAULT_BASELINE_DATE   = date(2026, 8, 1)
DEFAULT_BASELINE_WEIGHT = 112.0
DEFAULT_TARGET_WEIGHT   = 72.0
DEFAULT_HEIGHT_M        = 1.82

HW_LOOKBACK_WEEKS   = 16   # settimane storiche per fit Holt-Winters
HW_FORECAST_SATS    = 16   # sabati futuri da prevedere
TARGET_CONFIRM_DAYS = 3

PC = dict(
    green="#22a55b", green_l="#56c97a", amber="#e07b20",
    red="#d63b3b",   blue="#2563eb",
    text="#0f1923",  text2="#5a6478",  text3="#8c96a8",
    grid="#e4e8ef",  bg="#ffffff",     surface="#f8f9fb",
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
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11), bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
)
_BASE_BARE = {k: v for k, v in BASE_LAYOUT.items() if k not in ("margin","xaxis","yaxis")}

def layout_kw(height, margin=None, xtitle=None, ytitle=None,
              xextra=None, yextra=None, **kw):
    m  = margin or dict(l=10, r=10, t=36, b=10)
    xa = dict(**BASE_LAYOUT["xaxis"])
    ya = dict(**BASE_LAYOUT["yaxis"])
    if xextra: xa.update(xextra)
    if yextra: ya.update(yextra)
    if xtitle: xa["title"] = xtitle
    if ytitle: ya["title"] = ytitle
    return {**_BASE_BARE, "height": height, "margin": m, "xaxis": xa, "yaxis": ya, **kw}

# ═══════════════════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════════════════

def banner_html(icon, strong, sub, style=""):
    cls = {"amber":"wm-banner-amber","red":"wm-banner-red"}.get(style,"")
    return (f'<div class="wm-banner {cls}"><div class="wm-banner-icon">{icon}</div>'
            f'<div class="wm-banner-text"><strong>{strong}</strong><br>'
            f'<span>{sub}</span></div></div>')

def section_html(icon, title):
    return (f'<div class="wm-section"><div class="wm-section-icon">{icon}</div>'
            f'<div class="wm-section-title">{title}</div></div>')

def progress_html(pct):
    pct = max(0.0, min(100.0, pct))
    return (f'<div class="wm-progress-outer">'
            f'<div class="wm-progress-inner" style="width:{pct:.1f}%"></div></div>')

def nota_html(text):
    return f'<div style="background:#f8f9fb;border:1px solid #e4e8ef;border-radius:8px;padding:10px 14px;margin:4px 0 14px;font-size:12.5px;color:#5a6478;line-height:1.6">{text}</div>'

# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def fmt_delta(v, dec=2):
    if v is None or not np.isfinite(float(v)): return None
    return f"{float(v):+.{dec}f}"

def next_saturday(d: date) -> date:
    days = (5 - d.weekday()) % 7
    r = d + timedelta(days=days)
    return r if r > d else r + timedelta(days=7)

def week_saturday(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts).normalize()
    return ts - pd.Timedelta(days=(ts.dayofweek - 5) % 7)

def ts_midnight(d) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, dtime.min))

def loss_class(kg_lost):
    """CSS class per colorare la perdita settimanale."""
    if kg_lost is None or not np.isfinite(float(kg_lost)): return ""
    v = float(kg_lost)
    if v >= 0.30: return "wm-loss-good"
    if v >= 0.10: return "wm-loss-slow"
    return "wm-loss-zero"

# ═══════════════════════════════════════════════════════════════════
# SERIE GIORNALIERA (Akima)
# ═══════════════════════════════════════════════════════════════════

def build_daily_series(daily_df: pd.DataFrame) -> pd.Series:
    """
    Ritorna una Series con indice DatetimeIndex giornaliero completo.
    Usa Akima per i gap ≤ 14 giorni, lascia NaN per i gap più lunghi.
    """
    d = daily_df.copy().sort_values("date")
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    s = d.set_index("date")["weight"].astype(float)
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    known = s.dropna()
    if len(known) < 4:
        return s.interpolate(method="time", limit_area="inside")
    x_k = np.array([(dd - known.index[0]).days for dd in known.index], dtype=float)
    y_k = known.values.astype(float)
    x_a = np.array([(dd - known.index[0]).days for dd in full_idx], dtype=float)
    result = s.copy()
    try:
        akima = Akima1DInterpolator(x_k, y_k)
        interp = akima(x_a)
        for i, (idx, is_nan) in enumerate(zip(full_idx, s.isna())):
            if not is_nan: continue
            pos = np.searchsorted(known.index, idx)
            dl = (idx - known.index[pos-1]).days if pos > 0 else 9999
            dr = (known.index[pos] - idx).days  if pos < len(known) else 9999
            if min(dl, dr) <= 14:
                result.iloc[i] = float(interp[i])
    except Exception:
        result = s.interpolate(method="time", limit=7, limit_area="inside")
    return result

# ═══════════════════════════════════════════════════════════════════
# ANCHOR SETTIMANALE (WLS sul sabato)
# ═══════════════════════════════════════════════════════════════════

def local_anchor(series: pd.Series, target, max_dist=4):
    target = pd.to_datetime(target).normalize()
    if target in series.index and pd.notna(series.loc[target]):
        return float(series.loc[target])
    s = series.dropna()
    if s.empty: return None
    deltas = ((s.index - target) / pd.Timedelta(days=1)).astype(float)
    mask   = np.abs(deltas) <= max_dist
    local  = s[mask]
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

def build_weekly_anchors(daily_series: pd.Series) -> pd.DataFrame:
    """
    Costruisce la serie degli anchor settimanali (un valore per sabato).
    """
    s_min = daily_series.dropna().index.min()
    s_max = daily_series.dropna().index.max()
    sat_range = pd.date_range(week_saturday(s_min), week_saturday(s_max), freq="7D")
    rows = []
    for sat in sat_range:
        v = local_anchor(daily_series, sat)
        if v is not None:
            rows.append({"saturday": pd.to_datetime(sat), "anchor": float(v)})
    df = pd.DataFrame(rows).sort_values("saturday").reset_index(drop=True)
    df["delta"] = df["anchor"].diff()           # negativo = perdita
    df["loss"]  = (-df["delta"]).clip(lower=0)  # perdita come valore positivo
    return df

# ═══════════════════════════════════════════════════════════════════
# MODELLO HOLT-WINTERS SUI SABATI
# ═══════════════════════════════════════════════════════════════════
# Perché Holt-Winters:
#   - La serie degli anchor settimanali ha un trend lineare discendente
#     (perdita di peso) e rumore stocastico settimana per settimana.
#   - Holt-Winters con trend additivo (senza stagionalità, dato che gli
#     anchor sono già il "valore pulito" del sabato) produce:
#       * α  — smorzamento del livello (risponde ai dati recenti)
#       * β  — smorzamento del trend   (stima il ritmo di perdita)
#   - Gli intervalli di confidenza sono calcolati sulla deviazione
#     residua in-sample, scalata con √t (propagazione dell'incertezza).
#   - Fallback a regressione lineare robusta se i dati sono troppo pochi.

@st.cache_data(ttl=300, show_spinner=False)
def find_optimal_lookback(weekly_df: pd.DataFrame, candidates=None):
    """
    Walk-forward cross-validation: prova diversi lookback e sceglie quello
    con RMSE out-of-sample minimo.
    Ritorna (best_lookback, best_oos_rmse).
    """
    if candidates is None:
        candidates = [6, 8, 10, 12, 16, 20]
    valid = weekly_df.dropna(subset=["anchor"]).reset_index(drop=True)
    best_k, best_rmse = candidates[0], np.inf
    for k in candidates:
        if len(valid) < k + 3:
            continue
        errors = []
        for i in range(k, len(valid)):
            y = valid.iloc[i - k:i]["anchor"].values.astype(float)
            actual = float(valid.iloc[i]["anchor"])
            if not np.all(np.isfinite(y)):
                continue
            try:
                fit = ExponentialSmoothing(
                    y, trend="add", seasonal=None,
                    initialization_method="estimated"
                ).fit(optimized=True, remove_bias=True)
                pred = float(fit.level[-1]) + float(fit.trend[-1])
                errors.append((actual - pred) ** 2)
            except Exception:
                continue
        if len(errors) >= 3:
            rmse = float(np.sqrt(np.mean(errors)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
    return best_k, best_rmse


def detect_trend_change(weekly_df: pd.DataFrame, short_weeks: int = 4, long_weeks: int = 12):
    """
    Confronta il trend lineare (kg/sett) delle ultime `short_weeks` settimane
    con quello delle ultime `long_weeks`.
    Ritorna dict con short_trend, long_trend, change_pct oppure None se dati insufficienti.
    """
    valid = weekly_df.dropna(subset=["anchor"]).sort_values("saturday").reset_index(drop=True)
    if len(valid) < long_weeks:
        return None

    def lin_trend(sub):
        x = np.arange(len(sub), dtype=float)
        y = sub["anchor"].values.astype(float)
        return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else None

    short = lin_trend(valid.tail(short_weeks))
    long_ = lin_trend(valid.tail(long_weeks))
    if short is None or long_ is None or abs(long_) < 1e-9:
        return None
    return {
        "short_trend":  short,
        "long_trend":   long_,
        "change_pct":   (short - long_) / abs(long_) * 100,
        "short_weeks":  short_weeks,
        "long_weeks":   long_weeks,
    }


@st.cache_data(ttl=300, show_spinner="Calcolo modello Holt-Winters…")
def fit_hw_model(weekly_df: pd.DataFrame, lookback_weeks: int = HW_LOOKBACK_WEEKS):
    """
    Fitta Holt-Winters additivo (trend senza stagionalità) sulla serie
    degli anchor settimanali degli ultimi `lookback_weeks` sabati.

    Ritorna un dizionario con:
      ok, fitted_values, residuals, rmse,
      last_level, last_trend, last_saturday, last_anchor,
      alpha, beta, n_obs, weekly_df_used
    """
    if weekly_df.empty or len(weekly_df) < 6:
        return {"ok": False, "reason": "too_few_weeks"}

    sub = weekly_df.tail(lookback_weeks).copy().reset_index(drop=True)
    y   = sub["anchor"].values.astype(float)

    # Rimuovi NaN eventali
    mask = np.isfinite(y)
    if mask.sum() < 6:
        return {"ok": False, "reason": "too_few_valid"}

    y_clean = y[mask]
    dates_clean = sub["saturday"].values[mask]

    try:
        model = ExponentialSmoothing(
            y_clean,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, remove_bias=True)
        fitted   = fit.fittedvalues
        residuals = y_clean - fitted
        rmse     = float(np.sqrt(np.mean(residuals**2)))

        # Livello e trend finale (ultimi parametri stimati)
        last_level = float(fit.level[-1])
        last_trend = float(fit.trend[-1])

        return {
            "ok": True,
            "fitted": fitted,
            "residuals": residuals,
            "rmse": rmse,
            "last_level": last_level,
            "last_trend": last_trend,
            "last_saturday": pd.to_datetime(dates_clean[-1]),
            "last_anchor": float(y_clean[-1]),
            "alpha": float(fit.params["smoothing_level"]),
            "beta":  float(fit.params.get("smoothing_trend", np.nan)),
            "n_obs": int(mask.sum()),
            "weekly_df_used": sub,
        }
    except Exception as e:
        return {"ok": False, "reason": str(e)}


def hw_forecast_saturdays(hw: dict, n_saturdays: int = HW_FORECAST_SATS, n_boot: int = 500):
    """
    Produce n_saturdays previsioni future con Holt-Winters.
    Intervallo di confidenza al 95% via bootstrap sui residui in-sample:
    per ogni orizzonte h si campionano h residui con reimmissione e si
    sommano al forecast puntuale — i percentili 2.5/97.5 formano il CI.
    Questo evita l'assunzione di errori i.i.d. gaussiani dell'approccio analitico.
    """
    if not hw.get("ok"):
        return pd.DataFrame()

    level     = hw["last_level"]
    trend     = hw["last_trend"]
    residuals = np.array(hw["residuals"])
    last_sat  = hw["last_saturday"]
    rng       = np.random.default_rng(42)

    rows = []
    for h in range(1, n_saturdays + 1):
        sat    = last_sat + pd.Timedelta(weeks=h)
        center = level + h * trend
        # Accumulo di h innovazioni campionate con reimmissione
        boot = np.array([
            center + float(np.sum(rng.choice(residuals, size=h, replace=True)))
            for _ in range(n_boot)
        ])
        rows.append({
            "saturday": sat,
            "forecast": round(float(center), 2),
            "low":      round(float(np.percentile(boot, 2.5)), 2),
            "high":     round(float(np.percentile(boot, 97.5)), 2),
        })

    return pd.DataFrame(rows)


def estimate_target_date_hw(hw: dict, target_w: float):
    """
    Stima la data in cui il forecast supera per la prima volta il target.
    Usa la previsione puntuale su un orizzonte lungo (max 3 anni = 156 sett).
    """
    if not hw.get("ok"): return None, None
    level = hw["last_level"]
    trend = hw["last_trend"]
    last_sat = hw["last_saturday"]

    if trend >= 0:   # non sta scendendo
        return None, None

    for h in range(1, 157):
        center = level + h * trend
        if center <= float(target_w):
            sat = last_sat + pd.Timedelta(weeks=h)
            days_from_today = (sat.date() - date.today()).days
            return sat.date(), max(0, days_from_today)
    return None, None

# ═══════════════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

@st.cache_data(ttl=60, show_spinner=False)
def load_manual() -> pd.DataFrame:
    res  = get_supabase().table("manual_entries").select("*").order("date").execute()
    rows = res.data or []
    if not rows:
        return pd.DataFrame(columns=["id","date","weight","bmi","source"])
    df = pd.DataFrame(rows)
    df["date"]   = pd.to_datetime(df["date"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["bmi"]    = pd.to_numeric(df["bmi"],    errors="coerce")
    df["source"] = df.get("source","manual").fillna("manual")
    return df.dropna(subset=["date","weight"]).sort_values("date").reset_index(drop=True)

def insert_manual_entry(dt, weight, bmi):
    ex = load_manual()
    if not ex.empty and ((ex["date"] - dt).abs() < pd.Timedelta(minutes=1)).any():
        raise ValueError("Esiste già una misura in questo orario.")
    get_supabase().table("manual_entries").insert({
        "date":   pd.Timestamp(dt).isoformat(),
        "weight": float(weight),
        "bmi":    float(bmi) if bmi is not None and pd.notna(bmi) else None,
        "source": "manual",
    }).execute()
    _invalidate_data_caches()

def delete_manual_by_id(ids):
    if not ids: return
    get_supabase().table("manual_entries").delete().in_("id", ids).execute()
    _invalidate_data_caches()

def clear_manual():
    get_supabase().table("manual_entries").delete().neq("id", 0).execute()
    _invalidate_data_caches()

# ═══════════════════════════════════════════════════════════════════
# RENPHO CSV
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Caricamento RENPHO…")
def load_renpho(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url, header=None)
    if raw.shape[1] == 1:
        s   = raw[0].astype(str).str.strip().str.replace(r'^"|"$', "", regex=True)
        raw = s.str.split(",", expand=True)
    if raw.shape[1] < 3:
        raise ValueError("CSV RENPHO non riconosciuto.")
    df = pd.DataFrame()
    df["date"]   = pd.to_datetime(
        raw[0].astype(str).str.strip() + " " + raw[1].astype(str).str.strip(),
        errors="coerce", dayfirst=True)
    w = (raw[2].astype(str).str.strip()
         .str.replace(",",".",regex=False)
         .str.replace("kg","",regex=False).str.strip())
    df["weight"] = pd.to_numeric(w, errors="coerce")
    df = df.dropna(subset=["date","weight"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df["source"] = "renpho"; df["bmi"] = np.nan; df["id"] = np.nan
    return df.reset_index(drop=True)

def combine(renpho, manual):
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
    m  = df["bmi"].isna() & df["weight"].notna()
    df.loc[m,"bmi"] = df.loc[m,"weight"] / (h**2)
    return df

def make_daily(df):
    d = df.copy(); d["day"] = d["date"].dt.date
    out = d.groupby("day", as_index=False).agg(
        date=("date","max"), weight=("weight","median"), bmi=("bmi","median"))
    out["source"] = "daily"
    return out.sort_values("date").reset_index(drop=True)

def last_meas(df):
    df = df.sort_values("date")
    ld = df["date"].dt.date.max()
    dd = df[df["date"].dt.date == ld]
    m  = dd[dd["source"]=="manual"]
    return (m if not m.empty else dd).sort_values("date").iloc[-1]

# ═══════════════════════════════════════════════════════════════════
# SECRETS CHECK
# ═══════════════════════════════════════════════════════════════════
csv_url = st.secrets.get("CSV_URL","")
if not csv_url:
    st.error("⚠️ `CSV_URL` mancante nei Secrets."); st.stop()
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
    height_m      = st.number_input("📏 Altezza (m)", value=float(DEFAULT_HEIGHT_M), step=0.01)
    ma_window     = st.selectbox("📈 Media mobile (gg)", [7,14,21,30], index=0)
    n_fc_sats     = st.selectbox("🔮 Sabati previsti", [8,12,16,20], index=1)
    st.divider()
    show_raw    = st.toggle("Punti RAW",            value=True)
    show_ma     = st.toggle("Media mobile",         value=True)
    show_band   = st.toggle("Banda variabilità",    value=True)
    st.divider()
    if st.button("🔄 Refresh dati", use_container_width=True):
        load_renpho.clear(); st.rerun()

# ═══════════════════════════════════════════════════════════════════
# CARICAMENTO DATI
# ═══════════════════════════════════════════════════════════════════
try:    renpho = load_renpho(csv_url)
except Exception as e: st.error(f"Errore RENPHO: {e}"); st.stop()
try:    manual = load_manual()
except Exception as e: st.error(f"Errore Supabase: {e}"); st.stop()

df    = add_bmi(combine(renpho, manual), height_m)
if df.empty: st.warning("Nessun dato disponibile."); st.stop()

daily = make_daily(df)
daily["ma"] = daily["weight"].rolling(ma_window, min_periods=1).mean()

# Filtro intervallo
min_d = df["date"].min().date(); max_d = df["date"].max().date()
sel   = st.sidebar.date_input("📅 Intervallo", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)
start_d, end_d = (sel if isinstance(sel,tuple) and len(sel)==2 else (min_d, max_d))

df_f    = df[(df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].copy()
daily_f = daily[(daily["date"].dt.date >= start_d) & (daily["date"].dt.date <= end_d)].copy()
if df_f.empty: st.warning("Intervallo vuoto."); st.stop()
daily_f["ma"] = daily_f["weight"].rolling(ma_window, min_periods=1).mean()

# ═══════════════════════════════════════════════════════════════════
# METRICHE CORE
# ═══════════════════════════════════════════════════════════════════
today    = date.today()
next_sat = next_saturday(today)

last_row     = last_meas(df_f)
last_dt      = pd.to_datetime(last_row["date"])
last_w       = float(last_row["weight"])
last_bmi_val = float(last_row["bmi"]) if pd.notna(last_row.get("bmi")) else np.nan

prev_df  = df_f[df_f["date"] < last_dt].sort_values("date")
prev_row = prev_df.iloc[-1] if not prev_df.empty else None
prev_w   = float(prev_row["weight"]) if prev_row is not None else None

loss_base   = float(baseline_weight - last_w)
dist_target = float(last_w - float(target_weight))
delta_loss  = float(loss_base - (baseline_weight - prev_w)) if prev_w else None
delta_dist  = float(dist_target - (prev_w - float(target_weight))) if prev_w else None

total_journey = float(baseline_weight - float(target_weight))
progress_pct  = max(0.0, min(100.0, loss_base / total_journey * 100)) if total_journey > 0 else 0.0

# ═══════════════════════════════════════════════════════════════════
# SERIE GIORNALIERA + ANCHOR + MODELLO HW
# ═══════════════════════════════════════════════════════════════════
daily_series = build_daily_series(daily)
weekly_df    = build_weekly_anchors(daily_series)
opt_lookback, opt_lookback_rmse = find_optimal_lookback(weekly_df)
hw           = fit_hw_model(weekly_df, lookback_weeks=opt_lookback)
fc_df        = hw_forecast_saturdays(hw, n_saturdays=int(n_fc_sats)) if hw.get("ok") else pd.DataFrame()
target_date_est, days_to_target = estimate_target_date_hw(hw, float(target_weight))
trend_change = detect_trend_change(weekly_df)

# Ritmo HW: trend per settimana (negativo = perdita)
hw_weekly_loss = float(-hw["last_trend"]) if hw.get("ok") else None
hw_rmse        = float(hw["rmse"])        if hw.get("ok") else None

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px">'
    '<h1 style="font-family:Outfit,sans-serif;font-size:2rem;font-weight:800;'
    'color:#0f1923;letter-spacing:-.03em;margin:0">⚖️ Weight Monitor</h1>'
    '<span style="font-family:JetBrains Mono,monospace;font-size:11px;font-weight:500;'
    'background:#e8f7ef;color:#166b3c;border:1px solid #b6e8cc;padding:2px 8px;'
    'border-radius:20px;letter-spacing:.06em">Holt-Winters</span>'
    '</div>'
    '<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#8c96a8;'
    'letter-spacing:.06em;margin-bottom:16px">RENPHO · Akima · Exponential Smoothing con trend</div>',
    unsafe_allow_html=True)

tab_dash, tab_manual, tab_forecast, tab_data = st.tabs(
    ["📊  Cruscotto", "✍️  Manuale", "🔮  Forecast", "🧾  Dataset"])

# ═══════════════════════════════════════════════════════════════════
# TAB — CRUSCOTTO
# ═══════════════════════════════════════════════════════════════════
with tab_dash:

    # ── KPI ────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("⚖️ Ultima misura",
              f"{last_w:.2f} kg", last_dt.strftime("%d %b  %H:%M"), delta_color="off")
    c2.metric("📉 Perso dal baseline",
              f"{loss_base:+.2f} kg",
              (fmt_delta(delta_loss,2)+" kg") if delta_loss else "—", delta_color="normal")
    c3.metric("🎯 Al target",
              f"{dist_target:+.2f} kg",
              (fmt_delta(delta_dist,2)+" kg") if delta_dist else "—", delta_color="inverse")
    c4.metric("📅 Arrivo stimato",
              target_date_est.strftime("%d %b %Y") if target_date_est else "—",
              f"tra {days_to_target} giorni" if days_to_target else
              ("non converge" if hw.get("ok") else "dati insuff."))

    # ── Forecast prossimo sabato ────────────────────────────────────
    st.markdown(section_html("🔮", "Forecast prossimo sabato"), unsafe_allow_html=True)
    if not fc_df.empty:
        nxt = fc_df.iloc[0]
        nxt_date   = pd.Timestamp(nxt["saturday"])
        nxt_fc     = float(nxt["forecast"])
        nxt_low    = float(nxt["low"])
        nxt_high   = float(nxt["high"])
        nxt_delta  = nxt_fc - last_w
        cs1, cs2, cs3 = st.columns(3)
        cs1.metric(
            f"🔮 Sabato {nxt_date.strftime('%d %b %Y')}",
            f"{nxt_fc:.2f} kg",
            f"{nxt_delta:+.2f} kg vs ultima misura",
            delta_color="inverse")
        cs2.metric("📉 IC 95% — limite inferiore", f"{nxt_low:.2f} kg")
        cs3.metric("📈 IC 95% — limite superiore", f"{nxt_high:.2f} kg")
    else:
        reason = hw.get("reason", "dati insufficienti") if not hw.get("ok") else "forecast vuoto"
        st.warning(f"Forecast non disponibile: {reason}. Servono almeno 6 sabati storici.")

    # ── Progress ───────────────────────────────────────────────────
    st.markdown(
        f"**Progresso** — <b>{progress_pct:.1f}%</b> &nbsp;"
        f"<span style='color:#8c96a8;font-size:12px'>"
        f"{last_w:.1f} kg → {float(target_weight):.1f} kg "
        f"(baseline {baseline_weight:.1f} kg)</span>",
        unsafe_allow_html=True)
    st.markdown(progress_html(progress_pct), unsafe_allow_html=True)

    if hw.get("ok") and hw_weekly_loss is not None:
        style = "" if hw_weekly_loss >= 0.30 else "amber" if hw_weekly_loss >= 0.10 else "red"
        st.markdown(banner_html(
            "📈",
            f"Ritmo stimato (Holt-Winters): −{hw_weekly_loss:.2f} kg/settimana "
            f"· Lookback ottimale: {opt_lookback} sett (RMSE OOS={opt_lookback_rmse:.2f} kg)",
            f"α={hw['alpha']:.2f} (reattività al livello) · "
            f"RMSE={hw_rmse:.2f} kg (errore medio in-sample)",
            style=style), unsafe_allow_html=True)

        # Banner "sei in pari con il piano?"
        days_since = (date.today() - hw["last_saturday"].date()).days
        expected_today = hw["last_level"] + (days_since / 7.0) * hw["last_trend"]
        diff_plan = last_w - expected_today
        if abs(diff_plan) > 0.05:
            plan_style = "" if diff_plan < 0 else "red"
            plan_icon  = "✅" if diff_plan < 0 else "⚠️"
            plan_label = "Sei SOTTO il piano" if diff_plan < 0 else "Sei SOPRA il piano"
            st.markdown(banner_html(
                plan_icon,
                f"{plan_label}: {diff_plan:+.2f} kg rispetto alla previsione di oggi",
                f"Il modello si aspettava {expected_today:.2f} kg oggi · "
                f"Ultima misura: {last_w:.2f} kg",
                style=plan_style), unsafe_allow_html=True)

        # Banner cambio trend
        if trend_change is not None:
            cp = trend_change["change_pct"]
            if abs(cp) >= 30:
                tc_icon  = "🚀" if cp < -30 else "🐢"
                tc_label = "Stai accelerando" if cp < -30 else "Stai rallentando (possibile plateau)"
                tc_style = "" if cp < -30 else "amber" if abs(cp) < 60 else "red"
                st.markdown(banner_html(
                    tc_icon,
                    f"{tc_label}: trend ultime {trend_change['short_weeks']} sett "
                    f"= {trend_change['short_trend']:+.2f} kg/sett vs "
                    f"storico {trend_change['long_weeks']} sett "
                    f"= {trend_change['long_trend']:+.2f} kg/sett",
                    f"Variazione del ritmo: {cp:+.0f}% rispetto allo storico.",
                    style=tc_style), unsafe_allow_html=True)

    # ── Medie kg/sett su 3 orizzonti ──────────────────────────────
    if not weekly_df.empty:
        wdf_valid = weekly_df.dropna(subset=["loss"]).copy()
        wdf_valid = wdf_valid[wdf_valid["loss"].between(0, 5)]

        def mean_loss(n_weeks):
            sub = wdf_valid.tail(n_weeks)
            if len(sub) < 2: return None
            return float(sub["loss"].mean())

        pace_4   = mean_loss(4)
        pace_8   = mean_loss(8)
        pace_all = float(wdf_valid["loss"].mean()) if len(wdf_valid) >= 2 else None

        st.markdown(section_html("📊","Media kg/settimana persi"), unsafe_allow_html=True)
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric(
            "Ultime 4 settimane",
            f"−{pace_4:.2f} kg/sett" if pace_4 else "—",
            f"su {min(4, len(wdf_valid))} settimane reali")
        pm2.metric(
            "Ultime 8 settimane",
            f"−{pace_8:.2f} kg/sett" if pace_8 else "—",
            f"su {min(8, len(wdf_valid))} settimane reali")
        pm3.metric(
            "Tutto lo storico",
            f"−{pace_all:.2f} kg/sett" if pace_all else "—",
            f"su {len(wdf_valid)} settimane totali")
        st.markdown(nota_html(
            "La <b>media kg/sett</b> è calcolata sugli <b>anchor settimanali</b> "
            "(peso stimato ogni sabato). Non è la media dei singoli pesaggi: "
            "vengono usati solo i punti sabato per eliminare la variabilità infrasettimanale. "
            "Il confronto tra 4, 8 settimane e storico ti dice se stai accelerando, rallentando o sei stabile."
        ), unsafe_allow_html=True)

    # ── Grafico peso ────────────────────────────────────────────────
    st.markdown(section_html("📈", "Andamento del peso"), unsafe_allow_html=True)

    fig = go.Figure()

    if show_band and len(daily_f) >= 7:
        rs = daily_f["weight"].rolling(7, min_periods=3).std().fillna(0)
        fig.add_trace(go.Scatter(
            x=pd.concat([daily_f["date"], daily_f["date"].iloc[::-1]]),
            y=pd.concat([daily_f["weight"]+rs, (daily_f["weight"]-rs).iloc[::-1]]),
            fill="toself", mode="none",
            fillcolor="rgba(34,165,91,0.07)",
            name="Variabilità ±1σ (7gg)", hoverinfo="skip"))

    if show_raw:
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["weight"], mode="markers",
            name="Misure singole",
            marker=dict(size=3.5, color=PC["text3"], opacity=0.40),
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>%{y:.2f} kg<extra>RAW</extra>"))

    fig.add_trace(go.Scatter(
        x=daily_f["date"], y=daily_f["weight"],
        mode="lines+markers", name="Peso giornaliero",
        line=dict(color=PC["green"], width=1.8),
        marker=dict(size=4, color=PC["green"]),
        hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>Giornaliero</extra>"))

    if show_ma:
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["ma"],
            mode="lines", name=f"Media mobile {ma_window}gg",
            line=dict(color=PC["blue"], width=2.5),
            hovertemplate=f"<b>%{{x|%d %b}}</b><br>MA {ma_window}gg: %{{y:.2f}} kg<extra></extra>"))

    # Punti anchor settimanali nel range visualizzato
    wdf_vis = weekly_df[
        (weekly_df["saturday"] >= pd.Timestamp(start_d)) &
        (weekly_df["saturday"] <= pd.Timestamp(end_d))
    ]
    if not wdf_vis.empty:
        fig.add_trace(go.Scatter(
            x=wdf_vis["saturday"], y=wdf_vis["anchor"],
            mode="markers", name="Peso sabato (anchor)",
            marker=dict(size=8, color=PC["amber"], symbol="diamond",
                        line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{x|%d %b}</b><br>Anchor sabato: %{y:.2f} kg<extra></extra>"))

    # Linea forecast nel grafico principale
    if not fc_df.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_df["saturday"], fc_df["saturday"].iloc[::-1]]),
            y=pd.concat([fc_df["high"], fc_df["low"].iloc[::-1]]),
            fill="toself", mode="none",
            fillcolor="rgba(224,123,32,0.10)",
            name="Forecast IC 95%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=fc_df["saturday"], y=fc_df["forecast"],
            mode="lines+markers", name="Forecast (sabati)",
            line=dict(color=PC["amber"], width=2, dash="dash"),
            marker=dict(size=5, color=PC["amber"]),
            hovertemplate="<b>%{x|%d %b}</b><br>Previsto: %{y:.2f} kg<extra>Forecast</extra>"))

    fig.add_hline(y=float(target_weight), line_dash="dot",
                  line_color=PC["red"], line_width=1.5,
                  annotation_text="🎯 Target",
                  annotation_font_color=PC["red"],
                  annotation_position="bottom right")

    if target_date_est:
        xp = pd.Timestamp(target_date_est).to_pydatetime()
        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=xp, x1=xp, y0=0, y1=1,
                      line=dict(dash="dot", color=PC["green"], width=1.2))
        fig.add_annotation(x=xp, y=0.97, xref="x", yref="paper",
                           text=f"target {target_date_est.strftime('%d %b')}",
                           showarrow=False, xanchor="left", yanchor="top",
                           font=dict(size=10, color=PC["green"]))

    fig.update_layout(**layout_kw(
        height=400, margin=dict(l=10, r=10, t=36, b=10),
        ytitle="Peso (kg)", yextra=dict(tickformat=".1f", ticksuffix=" kg")))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(nota_html(
        f"<b>Linea verde</b> = peso mediano del giorno. "
        f"<b>Linea blu</b> = media mobile {ma_window} giorni (smussata). "
        f"<b>Diamanti arancioni</b> = peso del sabato stimato (anchor), usato per il forecast. "
        f"La <b>fascia verde</b> mostra la variabilità normale (±1σ su 7 giorni)."
    ), unsafe_allow_html=True)

    # ── Tabella settimanale storica ─────────────────────────────────
    st.markdown(section_html("📋", "Peso per settimana — storico"), unsafe_allow_html=True)

    wdf_hist = weekly_df[
        (weekly_df["saturday"] >= pd.Timestamp(start_d)) &
        (weekly_df["saturday"] <= pd.Timestamp(end_d))
    ].copy()

    if not wdf_hist.empty:
        # Costruisci mappa sabato→fitted dal modello HW (dove disponibile)
        fitted_map = {}
        if hw.get("ok"):
            hw_sub = hw["weekly_df_used"].copy()
            hw_sub["saturday"] = pd.to_datetime(hw_sub["saturday"])
            for _, fr in hw_sub.iterrows():
                idx = int(fr.name)
                if idx < len(hw["fitted"]):
                    fitted_map[fr["saturday"].normalize()] = float(hw["fitted"][idx])

        rows_html = ""
        for _, row in wdf_hist.sort_values("saturday", ascending=False).iterrows():
            sat_str  = pd.to_datetime(row["saturday"]).strftime("%d %b %Y")
            anchor_s = f"{row['anchor']:.2f} kg"
            if pd.notna(row["loss"]) and np.isfinite(row["loss"]):
                loss_v = float(row["loss"])
                loss_s = f"−{loss_v:.2f} kg"
                cls    = loss_class(loss_v)
            else:
                loss_s = "—"; cls = ""
            sat_key = pd.to_datetime(row["saturday"]).normalize()
            if sat_key in fitted_map:
                diff = float(row["anchor"]) - fitted_map[sat_key]
                diff_s = f"{diff:+.2f} kg"
                diff_color = "#166b3c" if diff < 0 else "#b94a48"
                vs_fc_s = f'<span style="color:{diff_color};font-weight:600">{diff_s}</span>'
            else:
                vs_fc_s = "—"
            rows_html += (
                f"<tr>"
                f"<td>{sat_str}</td>"
                f"<td>{anchor_s}</td>"
                f'<td class="{cls}">{loss_s}</td>'
                f"<td>{vs_fc_s}</td>"
                f"</tr>"
            )
        st.markdown(
            f'<table class="wm-week-table">'
            f"<thead><tr><th>Settimana (sabato)</th><th>Peso sabato</th>"
            f"<th>Perso vs sett. prec.</th><th>vs Forecast HW</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True)
        st.markdown(nota_html(
            f"<b>Verde</b> = più di 0.30 kg persi · "
            f"<b>Arancione</b> = tra 0.10 e 0.30 kg · "
            f"<b>Rosso</b> = meno di 0.10 kg (plateau). "
            f"<b>vs Forecast HW</b>: differenza tra peso reale e stima del modello per quel sabato "
            f"(verde = sotto la previsione = meglio del previsto, rosso = sopra)."
        ), unsafe_allow_html=True)

    # ── Ultime 10 misure ────────────────────────────────────────────
    st.markdown(section_html("🕐", "Ultime 10 misure registrate"), unsafe_allow_html=True)
    last10 = df_f.sort_values("date", ascending=False).head(10).copy()
    last10["Data"]      = last10["date"].dt.strftime("%Y-%m-%d  %H:%M")
    last10["Peso (kg)"] = last10["weight"].map(lambda x: f"{x:.2f}")
    last10["BMI"]       = last10["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    last10["Origine"]   = last10["source"]
    st.dataframe(last10[["Data","Peso (kg)","BMI","Origine"]],
                 use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB — MANUALE
# ═══════════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown(section_html("✍️","Inserimento manuale"), unsafe_allow_html=True)
    st.caption("BMI = 0 → calcolato automaticamente dall'altezza impostata.")

    with st.form("manual_form", clear_on_submit=True):
        cA,cB,cC,cD = st.columns(4)
        m_date   = cA.date_input("Data",           value=date.today())
        m_time   = cB.time_input("Ora",            value=datetime.now().time().replace(second=0,microsecond=0))
        m_weight = cC.number_input("Peso (kg)",    min_value=0.0, value=float(last_w), step=0.1)
        m_bmi    = cD.number_input("BMI (0=auto)", min_value=0.0, value=0.0,           step=0.1)
        if st.form_submit_button("✅ Salva misura", use_container_width=True):
            try:
                dt = pd.Timestamp(datetime.combine(m_date, m_time))
                bv = float(m_bmi) if float(m_bmi) > 0 else (
                    float(m_weight)/(height_m**2) if height_m > 0 else np.nan)
                insert_manual_entry(dt, float(m_weight), bv)
                st.success("✓ Misura salvata."); st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")

    st.divider()
    st.markdown(section_html("🗑️","Gestione archivio manuale"), unsafe_allow_html=True)
    manual_now = load_manual()
    if manual_now.empty:
        st.info("Nessuna misura manuale.")
    else:
        tmp = manual_now.copy()
        tmp["label"] = (tmp["date"].dt.strftime("%Y-%m-%d %H:%M")+" │ "
                        +tmp["weight"].map(lambda x: f"{x:.2f} kg"))
        sel_labels = st.multiselect("Seleziona da cancellare", tmp["label"].tolist())
        sel_ids    = tmp.loc[tmp["label"].isin(sel_labels),"id"].astype(int).tolist()
        cc1, cc2   = st.columns(2)
        if cc1.button("🗑️ Cancella selezionate", use_container_width=True, type="primary"):
            if sel_ids:
                try: delete_manual_by_id(sel_ids); st.success("Cancellate."); st.rerun()
                except Exception as e: st.error(f"Errore: {e}")
            else: st.warning("Seleziona almeno un record.")
        if cc2.button("⚠️ Cancella TUTTO", use_container_width=True):
            try: clear_manual(); st.success("Azzerato."); st.rerun()
            except Exception as e: st.error(f"Errore: {e}")
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
    st.markdown(section_html("🔮","Previsione sabati — Holt-Winters"), unsafe_allow_html=True)

    if not hw.get("ok"):
        st.warning(f"Modello non disponibile: {hw.get('reason','dati insufficienti')}.")
    else:
        # Riepilogo modello
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Ritmo stimato",
                     f"−{hw_weekly_loss:.2f} kg/sett",
                     "tendenza Holt-Winters")
        col_b.metric("Precisione modello (RMSE)",
                     f"±{hw_rmse:.2f} kg",
                     f"su {hw['n_obs']} sabati storici")
        col_c.metric("Arrivo al target",
                     target_date_est.strftime("%d %b %Y") if target_date_est else "—",
                     f"tra {days_to_target} giorni" if days_to_target else "—")

        st.markdown(nota_html(
            f"Il modello <b>Holt-Winters</b> con trend additivo è stato fittato sugli ultimi "
            f"<b>{hw['n_obs']} sabati</b>. "
            f"Parametro α={hw['alpha']:.3f}: quanto il modello aggiorna il livello a ogni nuova misura "
            f"(più alto = più reattivo). "
            f"Il <b>RMSE ±{hw_rmse:.2f} kg</b> misura quanto il modello era preciso sulle settimane storiche già note. "
            f"L'intervallo nella tabella è <b>±1.96·RMSE·√h</b> dove h è il numero di settimane nel futuro "
            f"(l'incertezza cresce col tempo)."
        ), unsafe_allow_html=True)

        if target_date_est:
            style = "" if hw_weekly_loss >= 0.30 else "amber" if hw_weekly_loss >= 0.10 else "red"
            st.markdown(banner_html(
                "🏁",
                f"Arrivo stimato al target ({float(target_weight):.1f} kg): "
                f"{target_date_est.strftime('%d %b %Y')} — tra {days_to_target} giorni",
                f"Basato su un ritmo di −{hw_weekly_loss:.2f} kg/sett. "
                f"Se il ritmo cambia, la data cambia proporzionalmente.",
                style=style), unsafe_allow_html=True)

        # ── Grafico forecast ─────────────────────────────────────
        st.markdown(section_html("📈","Grafico forecast"), unsafe_allow_html=True)

        if not fc_df.empty:
            fig_fc = go.Figure()

            # Banda ±95%
            fig_fc.add_trace(go.Scatter(
                x=list(fc_df["saturday"]) + list(fc_df["saturday"])[::-1],
                y=list(fc_df["high"])     + list(fc_df["low"])[::-1],
                fill="toself", mode="none",
                fillcolor="rgba(224,123,32,0.12)",
                name="Intervallo ±95%",
                hoverinfo="skip"))

            # Linea forecast
            fig_fc.add_trace(go.Scatter(
                x=fc_df["saturday"], y=fc_df["forecast"],
                mode="lines+markers", name="Previsione (sabati)",
                line=dict(color=PC["amber"], width=2.5),
                marker=dict(size=7, color=PC["amber"],
                            line=dict(width=2, color="white")),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Previsto: <b>%{y:.2f} kg</b><extra></extra>")))

            # Punti anchor storici (ultimi N)
            hist_pts = weekly_df.tail(int(n_fc_sats)).copy()
            fig_fc.add_trace(go.Scatter(
                x=hist_pts["saturday"], y=hist_pts["anchor"],
                mode="markers", name="Storico sabati",
                marker=dict(size=6, color=PC["green"],
                            line=dict(width=1.5, color="white")),
                hovertemplate=(
                    "<b>%{x|%d %b %Y}</b><br>"
                    "Reale: <b>%{y:.2f} kg</b><extra></extra>")))

            # Target line
            fig_fc.add_hline(
                y=float(target_weight), line_dash="dot",
                line_color=PC["red"], line_width=1.5,
                annotation_text="🎯 Target",
                annotation_font_color=PC["red"],
                annotation_position="bottom right")

            # Linea verticale oggi
            today_ts = pd.Timestamp(today).to_pydatetime()
            fig_fc.add_shape(type="line", xref="x", yref="paper",
                             x0=today_ts, x1=today_ts, y0=0, y1=1,
                             line=dict(dash="dot", color=PC["text3"], width=1))
            fig_fc.add_annotation(x=today_ts, y=0.99, xref="x", yref="paper",
                                  text="oggi", showarrow=False,
                                  xanchor="left", yanchor="top",
                                  font=dict(size=10, color=PC["text3"]))

            if target_date_est:
                xp = pd.Timestamp(target_date_est).to_pydatetime()
                fig_fc.add_shape(type="line", xref="x", yref="paper",
                                 x0=xp, x1=xp, y0=0, y1=1,
                                 line=dict(dash="dot", color=PC["green"], width=1.2))
                fig_fc.add_annotation(x=xp, y=0.99, xref="x", yref="paper",
                                      text=f"target {target_date_est.strftime('%d %b')}",
                                      showarrow=False, xanchor="left", yanchor="top",
                                      font=dict(size=10, color=PC["green"]))

            fig_fc.update_layout(**layout_kw(
                height=380, margin=dict(l=10, r=10, t=16, b=10),
                ytitle="Peso (kg)",
                yextra=dict(tickformat=".1f", ticksuffix=" kg")))
            st.plotly_chart(fig_fc, use_container_width=True)

            st.markdown(nota_html(
                f"<b>Linea arancione</b> = previsione puntuale ogni sabato (Holt-Winters). "
                f"<b>Fascia arancione</b> = intervallo ±95%: il peso reale dovrebbe cadere qui ~95% delle volte. "
                f"Si allarga nel tempo perché l'incertezza si accumula (formula: ±1.96 · RMSE · √settimane). "
                f"<b>Punti verdi</b> = sabati storici reali usati per fittare il modello. "
                f"<b>RMSE={hw_rmse:.2f} kg</b>: errore medio del modello sui sabati già noti."
            ), unsafe_allow_html=True)

        # ── Tabella sabati previsti ──────────────────────────────
        if not fc_df.empty:
            st.markdown(section_html("📅","Sabati previsti"), unsafe_allow_html=True)

            # Costruisci HTML tabella con highlight target
            rows_html = ""
            for _, row in fc_df.iterrows():
                sat_str = pd.to_datetime(row["saturday"]).strftime("%d %b %Y")
                fc_v    = float(row["forecast"])
                low_v   = float(row["low"])
                hi_v    = float(row["high"])
                dist    = round(fc_v - float(target_weight), 2)
                dist_s  = f"{dist:+.2f} kg"

                # Highlight se siamo vicini o sotto il target
                row_style = ""
                if fc_v <= float(target_weight):
                    row_style = ' style="background:#e8f7ef;"'

                rows_html += (
                    f"<tr{row_style}>"
                    f"<td>{sat_str}</td>"
                    f"<td><b>{fc_v:.2f} kg</b></td>"
                    f"<td style='color:#8c96a8'>{low_v:.2f} – {hi_v:.2f}</td>"
                    f"<td>{dist_s}</td>"
                    f"</tr>"
                )

            st.markdown(
                f'<table class="wm-week-table">'
                f"<thead><tr>"
                f"<th>Sabato</th>"
                f"<th>Peso previsto</th>"
                f"<th>Range ±95%</th>"
                f"<th>Al target</th>"
                f"</tr></thead>"
                f"<tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True)

            st.markdown(nota_html(
                f"<b>Peso previsto</b>: stima puntuale del modello per quel sabato. "
                f"<b>Range ±95%</b>: forchetta entro cui cadrà il peso reale con probabilità ~95% "
                f"(si allarga nel tempo perché l'incertezza si accumula). "
                f"<b>Al target</b>: kg ancora mancanti — le righe verdi indicano che il target è raggiunto. "
                f"Nota: il forecast assume che il ritmo rimanga costante. "
                f"Se cambia dieta o attività fisica, il modello si aggiornerà alla prossima settimana."
            ), unsafe_allow_html=True)

        st.caption(
            f"Holt-Winters additivo · α={hw['alpha']:.3f} · "
            f"β={hw['beta']:.3f} · RMSE={hw_rmse:.3f} kg · "
            f"n={hw['n_obs']} sabati · lookback={HW_LOOKBACK_WEEKS} settimane")

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

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Scarica CSV misure",
        data=out[["Data","Peso (kg)","BMI","Origine"]].to_csv(index=False).encode("utf-8"),
        file_name="renpho_export.csv", mime="text/csv", use_container_width=True)

    if not weekly_df.empty:
        wexp = weekly_df.copy()
        wexp["Settimana"]   = wexp["saturday"].dt.strftime("%Y-%m-%d")
        wexp["Anchor (kg)"] = wexp["anchor"].map(lambda x: f"{x:.3f}")
        wexp["Perso (kg)"]  = wexp["loss"].map(lambda x: f"−{x:.3f}" if pd.notna(x) and np.isfinite(x) else "")
        c2.download_button("⬇️ Scarica andamento settimanale",
            data=wexp[["Settimana","Anchor (kg)","Perso (kg)"]].to_csv(index=False).encode("utf-8"),
            file_name="renpho_settimane.csv", mime="text/csv", use_container_width=True)
