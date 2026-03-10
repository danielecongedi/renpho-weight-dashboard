import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from supabase import create_client, Client
from datetime import datetime, timedelta, date, time as dtime

# ═══════════════════════════════════════════════════════════
# CONFIGURAZIONE PAGINA
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RENPHO Weight Monitor",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
  }

  /* Background scuro con texture sottile */
  .stApp {
    background: #0d0f14;
    background-image: radial-gradient(ellipse at 20% 20%, rgba(99,179,120,0.06) 0%, transparent 50%),
                      radial-gradient(ellipse at 80% 80%, rgba(99,179,120,0.04) 0%, transparent 50%);
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid rgba(99,179,120,0.15);
  }

  /* Metriche */
  [data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(99,179,120,0.18);
    border-radius: 12px;
    padding: 16px 18px !important;
    transition: border-color 0.2s;
  }
  [data-testid="stMetric"]:hover {
    border-color: rgba(99,179,120,0.4);
  }
  [data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.45) !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    color: #e8f5eb !important;
  }
  [data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
  }

  /* Tabs */
  [data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.5) !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #63b378 !important;
    border-bottom: 2px solid #63b378 !important;
  }

  /* Titoli */
  h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #e8f5eb !important; letter-spacing: -0.02em; }
  h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #c8e6ce !important; }

  /* Dataframe */
  [data-testid="stDataFrame"] {
    border: 1px solid rgba(99,179,120,0.15);
    border-radius: 10px;
    overflow: hidden;
  }

  /* Pulsanti */
  .stButton > button {
    background: rgba(99,179,120,0.1) !important;
    border: 1px solid rgba(99,179,120,0.35) !important;
    color: #63b378 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 8px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: rgba(99,179,120,0.22) !important;
    border-color: #63b378 !important;
  }

  /* Divider */
  hr { border-color: rgba(99,179,120,0.12) !important; }

  /* Caption / piccoli testi */
  [data-testid="stCaptionContainer"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px;
    color: rgba(255,255,255,0.35) !important;
  }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# COSTANTI
# ═══════════════════════════════════════════════════════════
DEFAULT_BASELINE_DATE = date(2026, 8, 1)
DEFAULT_BASELINE_WEIGHT = 112.0
DEFAULT_TARGET_WEIGHT = 72.0
DEFAULT_HEIGHT_M = 1.82
FORECAST_LOOKBACK_DAYS = 90
PATTERN_WEEKS = 6
MAX_LOCAL_ANCHOR_DISTANCE_DAYS = 4

# Limiti forecast ampliati per non tagliare ritmi reali
WEEKLY_LOSS_MIN = 0.10   # 100g/settimana: plateau reale
WEEKLY_LOSS_MAX = 1.80   # 1.8kg: massimo fisiologicamente sostenibile
SHAPE_CLIP_MAX = 0.80    # kg sopra l'anchor nel giorno peggiore della settimana
SHAPE_CLIP_MIN = -0.05   # kg sotto anchor (anchor è già il minimo settimanale)

# Target date: richiede N giorni consecutivi sotto soglia per confermare
TARGET_CONFIRM_DAYS = 3

# Colori palette
C_GREEN = "#63b378"
C_GREEN_LIGHT = "#8ecfa1"
C_AMBER = "#e8a838"
C_RED = "#e05252"
C_BLUE = "#5b9bd5"
C_BG = "#0d0f14"
C_SURFACE = "#111318"
C_TEXT = "#e8f5eb"
C_TEXT_DIM = "rgba(232,245,235,0.45)"
C_GRID = "rgba(99,179,120,0.08)"
C_BORDER = "rgba(99,179,120,0.18)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", size=11, color=C_TEXT),
    xaxis=dict(
        gridcolor=C_GRID,
        zeroline=False,
        showline=False,
        tickfont=dict(size=10, color=C_TEXT_DIM),
    ),
    yaxis=dict(
        gridcolor=C_GRID,
        zeroline=False,
        showline=False,
        tickfont=dict(size=10, color=C_TEXT_DIM),
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#1a1e26",
        bordercolor=C_BORDER,
        font=dict(family="DM Mono, monospace", size=11, color=C_TEXT),
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=10),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
    ),
)

# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════

def ts_midnight(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, dtime.min))


def format_delta(value: float | None, decimals: int = 2) -> str | None:
    if value is None or not np.isfinite(float(value)):
        return None
    return f"{value:+.{decimals}f}"


def next_saturday(d: date) -> date:
    days = (5 - d.weekday()) % 7
    result = d + timedelta(days=days)
    return result if result > d else result + timedelta(days=7)


def week_saturday(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts).normalize()
    return ts - pd.Timedelta(days=(ts.dayofweek - 5) % 7)


def add_vline(fig: go.Figure, x_dt, text: str, color: str = C_BORDER):
    x_py = pd.to_datetime(x_dt).to_pydatetime()
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=x_py, x1=x_py, y0=0, y1=1,
                  line=dict(dash="dot", color=color, width=1))
    fig.add_annotation(x=x_py, y=0.98, xref="x", yref="paper",
                       text=text, showarrow=False,
                       xanchor="left", yanchor="top", yshift=-4,
                       font=dict(size=10, color=color))


# ═══════════════════════════════════════════════════════════
# SERIE GIORNALIERA RAGIONATA
# ═══════════════════════════════════════════════════════════

def build_daily_series(daily_df: pd.DataFrame) -> pd.Series:
    """
    Costruisce una serie giornaliera continua.
    Interpola solo gap brevi (≤2 giorni) per non inventare dati su periodi lunghi.
    """
    d = daily_df.copy().sort_values("date")
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    s = d.set_index("date")["weight"].astype(float)
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    return s.interpolate(method="time", limit=2, limit_area="inside")


# ═══════════════════════════════════════════════════════════
# ANCHOR LOCALE ROBUSTO
# ═══════════════════════════════════════════════════════════

def local_anchor(
    series: pd.Series,
    target: pd.Timestamp,
    max_dist: int = MAX_LOCAL_ANCHOR_DISTANCE_DAYS,
) -> float | None:
    """
    Stima il peso in `target` usando regressione locale pesata sui punti vicini.
    NON introduce bias verso il minimo: usa tutti i punti nell'intorno.
    """
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

    # Regressione WLS se ci sono punti su entrambi i lati
    left_n = int((x < 0).sum())
    right_n = int((x > 0).sum())
    if len(local) >= 3 and left_n >= 1 and right_n >= 1:
        X = np.vstack([np.ones(len(x)), x]).T
        W = np.diag(w)
        try:
            beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
            return float(beta[0])
        except Exception:
            pass

    return float(np.sum(y * w) / np.sum(w))


# ═══════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════

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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["source"] = df.get("source", "manual").fillna("manual")
    return df.dropna(subset=["date", "weight"]).sort_values("date").reset_index(drop=True)


def insert_manual_entry(dt: pd.Timestamp, weight: float, bmi: float | None):
    # Controlla duplicati nell'arco di 1 minuto
    existing = load_manual()
    if not existing.empty:
        diff = (existing["date"] - dt).abs()
        if (diff < pd.Timedelta(minutes=1)).any():
            raise ValueError("Esiste già una misura a questo orario.")
    payload = {
        "date": pd.Timestamp(dt).isoformat(),
        "weight": float(weight),
        "bmi": float(bmi) if bmi is not None and pd.notna(bmi) else None,
        "source": "manual",
    }
    get_supabase().table("manual_entries").insert(payload).execute()
    load_manual.clear()


def delete_manual_entries_by_id(ids: list[int]):
    if not ids:
        return
    get_supabase().table("manual_entries").delete().in_("id", ids).execute()
    load_manual.clear()


def clear_manual_entries():
    get_supabase().table("manual_entries").delete().neq("id", 0).execute()
    load_manual.clear()


# ═══════════════════════════════════════════════════════════
# CARICAMENTO DATI RENPHO
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Scaricamento dati RENPHO…")
def load_renpho_csv(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url, header=None)
    if raw.shape[1] == 1:
        s = raw[0].astype(str).str.strip().str.replace(r'^"|"$', "", regex=True)
        raw = s.str.split(",", expand=True)
    if raw.shape[1] < 3:
        raise ValueError("CSV non riconosciuto: servono almeno (data, ora, peso).")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(
        raw[0].astype(str).str.strip() + " " + raw[1].astype(str).str.strip(),
        errors="coerce", dayfirst=True,
    )
    w = (raw[2].astype(str).str.strip()
         .str.replace(",", ".", regex=False)
         .str.replace("kg", "", regex=False)
         .str.strip())
    df["weight"] = pd.to_numeric(w, errors="coerce")
    df = df.dropna(subset=["date", "weight"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df["source"] = "renpho"
    df["bmi"] = np.nan
    df["id"] = np.nan
    return df.reset_index(drop=True)


def combine_data(renpho: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    if "id" not in renpho.columns:
        renpho = renpho.copy()
        renpho["id"] = np.nan
    if manual.empty:
        return renpho.sort_values("date").reset_index(drop=True)
    df = pd.concat([renpho, manual], ignore_index=True)
    df["__prio"] = df["source"].map({"renpho": 0, "manual": 1}).fillna(0)
    df = (df.sort_values(["date", "__prio"])
          .drop(columns=["__prio"])
          .drop_duplicates(subset=["date"], keep="last"))
    return df.sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def add_bmi(df: pd.DataFrame, height_m: float) -> pd.DataFrame:
    if height_m <= 0:
        return df
    df = df.copy()
    mask = df["bmi"].isna() & df["weight"].notna()
    df.loc[mask, "bmi"] = df.loc[mask, "weight"] / (height_m ** 2)
    return df


def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["day"] = d["date"].dt.date
    out = d.groupby("day", as_index=False).agg(
        date=("date", "max"),
        weight=("weight", "median"),
        bmi=("bmi", "median"),
    )
    out["source"] = "daily"
    return out.sort_values("date").reset_index(drop=True)


def last_measurement(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("date")
    last_day = df["date"].dt.date.max()
    day_df = df[df["date"].dt.date == last_day]
    manual = day_df[day_df["source"] == "manual"]
    return (manual if not manual.empty else day_df).sort_values("date").iloc[-1]


# ═══════════════════════════════════════════════════════════
# MODELLO FORECAST — v9
#
# Miglioramenti rispetto al precedente:
#  1. weekly_loss: usa MEDIANA (non media) delle perdite settimanali → robusta agli outlier
#  2. Anchor settimanale: NON è il minimo del weekend, è la stima sul sabato esatto → no bias
#  3. weekday_shape appresa su dati RAW (non smoothed) → non distorta dalla MA
#  4. Clip weekly_loss: [0.10, 1.80] → copre plateau reali e perdite sostenibili
#  5. Clip shape: [-0.05, 0.80] → non taglia picchi infrasettimanali alti
#  6. Peso esponenziale alle settimane recenti nel calcolo della mediana settimanale
#  7. Smoothing applicato SOLO per la visualizzazione, NON per il calcolo della shape
#  8. Target date: confermato solo dopo TARGET_CONFIRM_DAYS giorni consecutivi sotto soglia
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Calcolo modello forecast…")
def fit_forecast_model(
    daily_df: pd.DataFrame,
    lookback_days: int = FORECAST_LOOKBACK_DAYS,
    smoothing_window: int = 5,
    pattern_weeks: int = PATTERN_WEEKS,
) -> dict:
    if daily_df.empty or len(daily_df) < 14:
        return {"ok": False, "reason": "too_few_points"}

    df = daily_df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    end = df["date"].max()
    start = end - pd.Timedelta(days=lookback_days)
    sub = df[df["date"] >= start].copy()
    if len(sub) < 14:
        sub = df.copy()
    sub = sub.sort_values("date").reset_index(drop=True)

    # Serie giornaliera RAW (per shape) e smoothed (per visualizzazione)
    daily_raw = build_daily_series(sub)
    daily_smooth = (daily_raw
                    .interpolate(method="time", limit_direction="both")
                    .rolling(window=smoothing_window, min_periods=1).mean())

    # ── Calcolo anchor settimanali ──────────────────────────────────
    first_sat = week_saturday(sub["date"].min())
    last_sat = week_saturday(sub["date"].max())
    sat_idx = pd.date_range(first_sat, last_sat, freq="7D")

    week_rows = []
    for sat in sat_idx:
        # Anchor = stima sul sabato ESATTO (nessun bias verso il minimo)
        est = local_anchor(daily_raw, sat, MAX_LOCAL_ANCHOR_DISTANCE_DAYS)
        if est is not None and np.isfinite(est):
            week_rows.append({
                "week_saturday": pd.to_datetime(sat),
                "anchor_weight": float(est),
            })

    week_df = pd.DataFrame(week_rows).sort_values("week_saturday").reset_index(drop=True)
    if len(week_df) < 4:
        return {"ok": False, "reason": "not_enough_weeks"}

    week_df["weekly_delta"] = week_df["anchor_weight"].diff()

    # ── weekly_loss: mediana pesata esponenzialmente (recente pesa di più) ──
    losses_raw = week_df["weekly_delta"].dropna().values.astype(float)
    # Solo settimane di perdita (clip tra -2 e 0)
    losses = np.clip(losses_raw, -2.0, 0.0)
    n = len(losses)
    if n == 0:
        return {"ok": False, "reason": "no_weekly_deltas"}

    # Pesi esponenziali: l'ultima settimana pesa ~3x la prima
    exp_weights = np.exp(np.linspace(0, 1.1, n))
    # Mediana pesata tramite interpolazione della CDF pesata
    order = np.argsort(losses)
    losses_sorted = losses[order]
    w_sorted = exp_weights[order]
    cumw = np.cumsum(w_sorted) / np.sum(w_sorted)
    weighted_median = float(np.interp(0.5, cumw, losses_sorted))

    # Perdita = valore positivo (perdita di peso)
    recent_avg_weekly_loss = float(np.mean(np.abs(losses)))
    robust_recent_loss = float(np.abs(weighted_median))
    # Forecast usa la mediana pesata (più robusta)
    weekly_loss = float(np.clip(robust_recent_loss, WEEKLY_LOSS_MIN, WEEKLY_LOSS_MAX))

    # ── Weekday shape: appresa su dati RAW, NON smoothed ──────────────
    hist = pd.DataFrame({
        "date": pd.to_datetime(daily_raw.index),
        "weight_raw": daily_raw.values,
    }).dropna()
    hist["weekday"] = hist["date"].dt.dayofweek
    hist["week_saturday"] = hist["date"].apply(week_saturday)

    recent_weeks = set(week_df.tail(pattern_weeks)["week_saturday"].tolist())
    anchor_map = week_df.set_index("week_saturday")["anchor_weight"].to_dict()

    rel_rows = []
    for _, row in hist[hist["week_saturday"].isin(recent_weeks)].iterrows():
        ws = pd.to_datetime(row["week_saturday"])
        anchor_w = anchor_map.get(ws)
        if anchor_w is not None and pd.notna(row["weight_raw"]):
            rel_rows.append({
                "weekday": int(row["weekday"]),
                "rel": float(row["weight_raw"] - anchor_w),
            })

    # Shape di default fisiologicamente plausibile (lun > ven, sab ≈ minimo)
    default_shape = {0: 0.40, 1: 0.28, 2: 0.18, 3: 0.10, 4: 0.04, 5: 0.00, 6: 0.08}

    rel_df = pd.DataFrame(rel_rows)
    if rel_df.empty:
        weekday_shape = default_shape.copy()
    else:
        # Mediana per weekday (robusta agli outlier)
        weekday_shape = rel_df.groupby("weekday")["rel"].median().to_dict()
        for wd in range(7):
            weekday_shape.setdefault(wd, default_shape[wd])

    # Clip ampliato: non tagliare valori reali
    weekday_shape = {
        wd: float(np.clip(v, SHAPE_CLIP_MIN, SHAPE_CLIP_MAX))
        for wd, v in weekday_shape.items()
    }
    # Vincoli di monotonia morbida (lunedì ≥ venerdì, sabato = minimo locale)
    weekday_shape[5] = max(weekday_shape[5], SHAPE_CLIP_MIN)       # sab ≥ 0 (anchor)
    weekday_shape[6] = max(weekday_shape[6], weekday_shape[5])     # dom ≥ sab
    weekday_shape[4] = max(weekday_shape[4], weekday_shape[5])     # ven ≥ sab
    weekday_shape[3] = max(weekday_shape[3], weekday_shape[4])     # gio ≥ ven
    weekday_shape[2] = max(weekday_shape[2], weekday_shape[3])     # mer ≥ gio
    weekday_shape[1] = max(weekday_shape[1], weekday_shape[2])     # mar ≥ mer
    weekday_shape[0] = max(weekday_shape[0], weekday_shape[1])     # lun ≥ mar

    last_sat = pd.to_datetime(week_df["week_saturday"].iloc[-1]).normalize()
    last_anchor = float(week_df["anchor_weight"].iloc[-1])
    last_date = pd.to_datetime(sub["date"].max()).normalize()
    last_weight = float(sub.loc[sub["date"] == last_date, "weight"].iloc[-1])

    week_df["weekly_loss_abs"] = week_df["weekly_delta"].clip(upper=0).abs()

    return {
        "ok": True,
        "type": "anchored_week_forecast_v9",
        "daily_raw": daily_raw,
        "daily_smooth": daily_smooth,
        "last_date": last_date,
        "last_weight": last_weight,
        "last_sat": last_sat,
        "last_anchor": last_anchor,
        "weekday_shape": weekday_shape,
        "weekly_loss": weekly_loss,
        "recent_avg_weekly_loss": recent_avg_weekly_loss,
        "robust_recent_loss": robust_recent_loss,
        "week_df": week_df,
    }


def anchor_for_week(model: dict, sat: pd.Timestamp) -> float:
    sat = pd.to_datetime(sat).normalize()
    weeks = int((sat - model["last_sat"]).days / 7)
    if weeks <= 0:
        return model["last_anchor"]
    return float(model["last_anchor"] - weeks * model["weekly_loss"])


def forecast_series(model: dict, start: pd.Timestamp, horizon: int) -> tuple[pd.DatetimeIndex, list[float]]:
    if not model.get("ok"):
        return pd.DatetimeIndex([]), []

    start = pd.to_datetime(start).normalize()
    dates = pd.date_range(start, start + pd.Timedelta(days=horizon), freq="D")
    daily_raw = model["daily_raw"]
    last_date = model["last_date"]

    values = []
    for d in dates:
        if d <= last_date and d in daily_raw.index and pd.notna(daily_raw.loc[d]):
            values.append(float(daily_raw.loc[d]))
        else:
            sat = week_saturday(d)
            anchor = anchor_for_week(model, sat)
            values.append(anchor + model["weekday_shape"].get(int(d.dayofweek), 0.0))

    return dates, values


def predict_weight(model: dict, when: pd.Timestamp) -> float:
    _, vals = forecast_series(model, when, horizon=0)
    return float(vals[0]) if vals else np.nan


def estimate_target_date(
    model: dict,
    target_weight: float,
    start_from: date,
    max_horizon: int = 365,
) -> tuple[date | None, int | None]:
    """
    La data target è confermata solo quando TARGET_CONFIRM_DAYS giorni
    consecutivi rimangono sotto la soglia → evita false anticipazioni da outlier.
    """
    dates, values = forecast_series(model, pd.Timestamp(start_from), max_horizon)
    if not values:
        return None, None

    consecutive = 0
    first_under: date | None = None
    for d, w in zip(dates, values):
        if w <= float(target_weight):
            if consecutive == 0:
                first_under = pd.to_datetime(d).date()
            consecutive += 1
            if consecutive >= TARGET_CONFIRM_DAYS:
                return first_under, (first_under - start_from).days
        else:
            consecutive = 0
            first_under = None

    return None, None


# ═══════════════════════════════════════════════════════════
# SECRETS CHECK
# ═══════════════════════════════════════════════════════════
csv_url = st.secrets.get("CSV_URL", "")
if not csv_url:
    st.error("⚠️ `CSV_URL` non impostato nei Secrets.")
    st.stop()
if not all(k in st.secrets for k in ("SUPABASE_URL", "SUPABASE_KEY")):
    st.error("⚠️ `SUPABASE_URL` o `SUPABASE_KEY` mancanti nei Secrets.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Impostazioni")

    target_weight = st.number_input("🎯 Target (kg)", value=float(DEFAULT_TARGET_WEIGHT), step=0.5)
    baseline_weight = st.number_input("⚑ Peso baseline (kg)", value=float(DEFAULT_BASELINE_WEIGHT), step=0.5)
    baseline_date = st.date_input("⚑ Data baseline", value=DEFAULT_BASELINE_DATE)

    st.divider()
    height_m = st.number_input("📏 Altezza (m)", value=float(DEFAULT_HEIGHT_M), step=0.01)

    st.divider()
    ma_window = st.selectbox("📈 Media mobile (giorni)", [7, 14, 21, 30], index=0)
    forecast_smoothing = st.selectbox("🧠 Smoothing forecast", [3, 5, 7, 10], index=1)
    forecast_horizon = st.selectbox("⏳ Orizzonte (giorni)", [30, 60, 90, 180], index=1)

    st.divider()
    show_raw = st.toggle("Punti RAW", value=True)
    show_daily = st.toggle("Linea DAILY", value=True)
    show_smooth = st.toggle("Curva smoothed", value=True)

    st.divider()
    if st.button("🔄 Forza refresh RENPHO", use_container_width=True):
        load_renpho_csv.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════
# CARICAMENTO E PREPARAZIONE DATI
# ═══════════════════════════════════════════════════════════
try:
    renpho = load_renpho_csv(csv_url)
except Exception as e:
    st.error(f"Errore caricamento RENPHO: {e}")
    st.stop()

try:
    manual = load_manual()
except Exception as e:
    st.error(f"Errore caricamento dati manuali: {e}")
    st.stop()

df = combine_data(renpho, manual)
df = add_bmi(df, height_m)

if df.empty:
    st.warning("Nessun dato disponibile.")
    st.stop()

daily = make_daily(df)
daily["ma"] = daily["weight"].rolling(ma_window, min_periods=1).mean()

# Date filter
min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "📅 Intervallo analisi",
    value=(min_date, max_date),
    min_value=min_date, max_value=max_date,
)
start_d, end_d = (date_range if isinstance(date_range, tuple) and len(date_range) == 2
                  else (min_date, max_date))

df_f = df[(df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].copy()
daily_f = daily[(daily["date"].dt.date >= start_d) & (daily["date"].dt.date <= end_d)].copy()

if df_f.empty:
    st.warning("L'intervallo selezionato non contiene dati.")
    st.stop()

daily_f["ma"] = daily_f["weight"].rolling(ma_window, min_periods=1).mean()

# ═══════════════════════════════════════════════════════════
# CALCOLI METRICHE
# ═══════════════════════════════════════════════════════════
today = date.today()
next_sat = next_saturday(today)
next_sat_ts = ts_midnight(next_sat)

last_row = last_measurement(df_f)
last_dt = pd.to_datetime(last_row["date"])
last_weight_val = float(last_row["weight"])
last_bmi = float(last_row["bmi"]) if pd.notna(last_row.get("bmi")) else np.nan

prev_df = df_f[df_f["date"] < last_dt].sort_values("date")
prev_row = prev_df.iloc[-1] if not prev_df.empty else None

prev_weight = float(prev_row["weight"]) if prev_row is not None else None
prev_bmi = (float(prev_row["bmi"]) if prev_row is not None and pd.notna(prev_row.get("bmi")) else None)

delta_bmi = ((last_bmi - prev_bmi) if prev_bmi is not None
             and np.isfinite(last_bmi) and np.isfinite(prev_bmi) else None)
loss_from_baseline = float(baseline_weight - last_weight_val)
dist_to_target = float(last_weight_val - float(target_weight))

if prev_weight is not None:
    delta_loss = float(loss_from_baseline - (baseline_weight - prev_weight))
    delta_dist = float(dist_to_target - (prev_weight - float(target_weight)))
else:
    delta_loss = delta_dist = None

# ═══════════════════════════════════════════════════════════
# MODELLO
# ═══════════════════════════════════════════════════════════
model = fit_forecast_model(
    daily,
    lookback_days=FORECAST_LOOKBACK_DAYS,
    smoothing_window=forecast_smoothing,
    pattern_weeks=PATTERN_WEEKS,
)

pred_next_sat = predict_weight(model, next_sat_ts) if model.get("ok") else None
target_date_est, days_to_target = (
    estimate_target_date(model, float(target_weight), today)
    if model.get("ok") else (None, None)
)

# ═══════════════════════════════════════════════════════════
# LAYOUT PRINCIPALE
# ═══════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='margin-bottom:0.1em'>⚖ Weight Monitor</h1>"
    "<p style='color:rgba(232,245,235,0.4);font-family:DM Mono,monospace;font-size:12px;"
    "letter-spacing:0.1em;margin-top:0;margin-bottom:1.5em'>RENPHO · FORECAST v9</p>",
    unsafe_allow_html=True,
)

tab_dash, tab_manual, tab_forecast, tab_data = st.tabs(
    ["📊  Cruscotto", "✍  Manuale", "🔮  Forecast", "🧾  Dataset"]
)

# ═══════════════════════════════════════════════════════════
# TAB: CRUSCOTTO
# ═══════════════════════════════════════════════════════════
with tab_dash:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Ultima misura",
        f"{last_weight_val:.2f} kg",
        last_dt.strftime("%d %b %H:%M"),
        delta_color="off",
    )
    c2.metric(
        "BMI",
        f"{last_bmi:.2f}" if np.isfinite(last_bmi) else "—",
        (format_delta(delta_bmi, 2) + " vs prec.") if delta_bmi is not None else "—",
        delta_color="inverse",
    )
    c3.metric(
        "Perso dal baseline",
        f"{loss_from_baseline:+.2f} kg",
        (format_delta(delta_loss, 2) + " kg") if delta_loss is not None else "—",
        delta_color="normal",
    )
    c4.metric(
        "Al target",
        f"{dist_to_target:+.2f} kg",
        (format_delta(delta_dist, 2) + " kg") if delta_dist is not None else "—",
        delta_color="inverse",
    )
    c5.metric(
        "Target stimato",
        target_date_est.strftime("%d %b %Y") if target_date_est else "—",
        f"{days_to_target}gg da oggi" if days_to_target else "—",
    )
    c6.metric(
        f"Sabato {next_sat.strftime('%d %b')}",
        f"{pred_next_sat:.2f} kg" if pred_next_sat else "—",
        "forecast",
        delta_color="off",
    )

    st.markdown("---")
    st.subheader("Trend + Forecast")

    fig = go.Figure()

    if show_raw and not df_f.empty:
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["weight"],
            mode="markers", name="RAW",
            marker=dict(size=4, color=C_TEXT_DIM, opacity=0.5),
            hovertemplate="<b>%{x|%d %b %H:%M}</b><br>%{y:.2f} kg<extra>RAW</extra>",
        ))

    if show_daily and not daily_f.empty:
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["weight"],
            mode="lines+markers", name="Daily",
            line=dict(color=C_GREEN, width=1.5),
            marker=dict(size=5, color=C_GREEN),
            hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>Daily</extra>",
        ))

    if show_smooth and not daily_f.empty:
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["ma"],
            mode="lines", name=f"MA {ma_window}g",
            line=dict(color=C_GREEN_LIGHT, width=2.5),
            hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>MA</extra>",
        ))

    # Forecast
    if model.get("ok") and len(daily_f) >= 2:
        last_daily_dt = daily_f["date"].max().normalize()
        f_dates, f_vals = forecast_series(
            model,
            start=last_daily_dt + pd.Timedelta(days=1),
            horizon=int(forecast_horizon),
        )
        if len(f_vals):
            fig.add_trace(go.Scatter(
                x=f_dates, y=f_vals,
                mode="lines", name=f"Forecast {forecast_horizon}g",
                line=dict(dash="dash", color=C_AMBER, width=2),
                hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>Forecast</extra>",
            ))

    # Prossimo sabato
    if pred_next_sat is not None:
        fig.add_trace(go.Scatter(
            x=[next_sat_ts.to_pydatetime()], y=[pred_next_sat],
            mode="markers+text", name=f"Sab {next_sat.strftime('%d %b')}",
            marker=dict(size=10, color=C_AMBER, symbol="diamond"),
            text=[f"{pred_next_sat:.2f}"], textposition="top center",
            textfont=dict(color=C_AMBER, size=11),
            hovertemplate=f"<b>Sabato {next_sat.strftime('%d %b')}</b><br>%{{y:.2f}} kg<extra>Forecast</extra>",
        ))

    # Target line
    fig.add_hline(
        y=float(target_weight),
        line_dash="dot", line_color=C_RED, line_width=1.5,
        annotation_text="🎯 Target",
        annotation_font_color=C_RED,
        annotation_position="bottom right",
    )

    add_vline(fig, last_dt, last_dt.strftime("%d %b"), color=C_BORDER)
    if target_date_est:
        add_vline(fig, ts_midnight(target_date_est),
                  f"target est. {target_date_est.strftime('%d %b')}", color=C_RED)

    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Data", yaxis_title="Peso (kg)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Prospetti tendenza ──
    st.markdown("---")
    st.subheader("Tendenza settimanale")

    week_df = model.get("week_df", pd.DataFrame()) if model.get("ok") else pd.DataFrame()
    if not week_df.empty:
        k1, k2, k3 = st.columns(3)
        k1.metric("Media 3 mesi", f"−{model['recent_avg_weekly_loss']:.2f} kg/sett")
        k2.metric("Mediana pesata (forecast)", f"−{model['robust_recent_loss']:.2f} kg/sett")
        last5_mean = float(week_df["weekly_loss_abs"].dropna().tail(5).mean())
        k3.metric("Ultime 5 settimane", f"−{last5_mean:.2f} kg/sett" if np.isfinite(last5_mean) else "—")

        fig_week = go.Figure()
        fig_week.add_trace(go.Scatter(
            x=week_df["week_saturday"], y=week_df["anchor_weight"],
            mode="lines+markers", name="Anchor settimanale",
            line=dict(color=C_GREEN, width=2),
            marker=dict(size=6, color=C_GREEN),
            hovertemplate="<b>%{x|%d %b}</b><br>%{y:.2f} kg<extra>Anchor</extra>",
        ))

        # Linea trend lineare
        w_df_clean = week_df.dropna(subset=["anchor_weight"])
        if len(w_df_clean) >= 3:
            x_num = (w_df_clean["week_saturday"] - w_df_clean["week_saturday"].min()).dt.days.values
            y_num = w_df_clean["anchor_weight"].values
            coeffs = np.polyfit(x_num, y_num, 1)
            y_trend = np.polyval(coeffs, x_num)
            fig_week.add_trace(go.Scatter(
                x=w_df_clean["week_saturday"], y=y_trend,
                mode="lines", name="Trend lineare",
                line=dict(dash="dash", color=C_AMBER, width=1.5),
                hoverinfo="skip",
            ))

        fig_week.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="Settimana (sabato)",
            yaxis_title="Peso anchor (kg)",
        )
        st.plotly_chart(fig_week, use_container_width=True)

        # Tabella settimanale
        st.subheader("Tabella riduzione settimanale")
        tbl = week_df[week_df["week_saturday"] >= pd.Timestamp("2025-12-01")].copy()
        tbl["Settimana"] = tbl["week_saturday"].dt.strftime("%d %b %Y")
        tbl["Peso anchor (kg)"] = tbl["anchor_weight"].map(lambda x: f"{x:.2f}")
        tbl["Δ vs prec. (kg)"] = tbl["weekly_loss_abs"].map(lambda x: f"−{x:.2f}" if pd.notna(x) else "")
        st.dataframe(
            tbl[["Settimana", "Peso anchor (kg)", "Δ vs prec. (kg)"]],
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.subheader("Ultime 10 misure")
    last10 = df_f.sort_values("date", ascending=False).head(10).copy()
    last10["Data"] = last10["date"].dt.strftime("%Y-%m-%d %H:%M")
    last10["Peso (kg)"] = last10["weight"].map(lambda x: f"{x:.2f}")
    last10["BMI"] = last10["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    last10["Origine"] = last10["source"]
    st.dataframe(last10[["Data", "Peso (kg)", "BMI", "Origine"]],
                 use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# TAB: MANUALE
# ═══════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("➕ Inserimento manuale")
    st.caption("BMI=0 → calcolato automaticamente dall'altezza impostata nella sidebar.")

    with st.form("manual_form", clear_on_submit=True):
        colA, colB, colC, colD = st.columns(4)
        m_date = colA.date_input("Data", value=date.today())
        m_time = colB.time_input("Ora", value=datetime.now().time().replace(second=0, microsecond=0))
        m_weight = colC.number_input("Peso (kg)", min_value=0.0, value=float(last_weight_val), step=0.1)
        m_bmi = colD.number_input("BMI (0=auto)", min_value=0.0, value=0.0, step=0.1)
        submitted = st.form_submit_button("✅ Salva", use_container_width=True)

        if submitted:
            try:
                dt = pd.Timestamp(datetime.combine(m_date, m_time))
                bmi_val = (float(m_bmi) if float(m_bmi) > 0
                           else (float(m_weight) / (height_m ** 2) if height_m > 0 else np.nan))
                insert_manual_entry(dt, float(m_weight), bmi_val)
                st.success("✓ Misura salvata.")
                st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")

    st.markdown("---")
    st.subheader("🗑️ Cancella misure manuali")
    manual_now = load_manual()

    if manual_now.empty:
        st.info("Nessuna misura manuale salvata.")
    else:
        tmp = manual_now.copy()
        tmp["label"] = (
            tmp["date"].dt.strftime("%Y-%m-%d %H:%M") + " │ " +
            tmp["weight"].map(lambda x: f"{x:.2f} kg")
        )
        selected_labels = st.multiselect("Seleziona record da cancellare", tmp["label"].tolist())
        selected_ids = tmp.loc[tmp["label"].isin(selected_labels), "id"].astype(int).tolist()

        col1, col2 = st.columns(2)
        if col1.button("🗑️ Cancella selezionate", use_container_width=True, type="primary"):
            if selected_ids:
                try:
                    delete_manual_entries_by_id(selected_ids)
                    st.success("Cancellate.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore: {e}")
            else:
                st.warning("Seleziona almeno un record.")

        if col2.button("⚠️ Cancella TUTTO", use_container_width=True):
            try:
                clear_manual_entries()
                st.success("Archivio azzerato.")
                st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")

        st.markdown("---")
        st.subheader("Archivio misure manuali")
        sm = tmp.sort_values("date", ascending=False).copy()
        sm["Data"] = sm["date"].dt.strftime("%Y-%m-%d %H:%M")
        sm["Peso (kg)"] = sm["weight"].map(lambda x: f"{x:.2f}")
        sm["BMI"] = sm["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
        sm["Origine"] = sm["source"]
        st.dataframe(sm[["Data", "Peso (kg)", "BMI", "Origine"]], use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════
# TAB: FORECAST
# ═══════════════════════════════════════════════════════════
with tab_forecast:
    st.subheader("Forecast settimanale — sabati fino al target")

    if not model.get("ok"):
        st.warning("Modello non disponibile (dati insufficienti).")
    elif not target_date_est:
        st.warning("Target non raggiungibile entro 365 giorni con il ritmo attuale.")
    else:
        start_sat = next_saturday(today)

        if target_date_est < start_sat:
            st.info(f"Il target è stimato prima del prossimo sabato ({start_sat.strftime('%d %b')}).")
        else:
            col_info, col_meta = st.columns([2, 1])
            col_info.success(
                f"Target stimato: **{target_date_est.strftime('%d %b %Y')}** "
                f"({days_to_target} giorni da oggi)"
            )
            col_meta.info(
                f"Forecast: **−{model['weekly_loss']:.2f} kg/sett** "
                f"(mediana pesata)"
            )

            rows = []
            s = start_sat
            while s <= target_date_est:
                w_pred = predict_weight(model, ts_midnight(s))
                dist = round(w_pred - float(target_weight), 2)
                rows.append({
                    "Sabato": s.strftime("%d %b %Y"),
                    "Peso previsto (kg)": round(w_pred, 2),
                    "Al target (kg)": f"{dist:+.2f}",
                    "Progresso": f"{max(0, 100 - dist / (last_weight_val - float(target_weight)) * 100):.0f}%",
                })
                s += timedelta(days=7)

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(
            f"Modello v9 · "
            f"Mediana pesata (recente): {model['robust_recent_loss']:.3f} kg/sett · "
            f"Media 3 mesi: {model['recent_avg_weekly_loss']:.3f} kg/sett · "
            f"Forecast clip [{WEEKLY_LOSS_MIN}, {WEEKLY_LOSS_MAX}] · "
            f"Target confermato dopo {TARGET_CONFIRM_DAYS} giorni consecutivi sotto soglia"
        )

# ═══════════════════════════════════════════════════════════
# TAB: DATASET
# ═══════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Dataset completo")

    out = df_f.sort_values("date", ascending=False).copy()
    out["Data"] = out["date"].dt.strftime("%Y-%m-%d %H:%M")
    out["Peso (kg)"] = out["weight"].map(lambda x: f"{x:.2f}")
    out["BMI"] = out["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    out["Origine"] = out["source"]

    st.dataframe(out[["Data", "Peso (kg)", "BMI", "Origine"]], use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Scarica CSV",
        data=out[["Data", "Peso (kg)", "BMI", "Origine"]].to_csv(index=False).encode("utf-8"),
        file_name="renpho_export.csv",
        mime="text/csv",
        use_container_width=True,
    )