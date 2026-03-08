import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from supabase import create_client, Client
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta, date, time as dtime

# -------------------------
# PAGE
# -------------------------
st.set_page_config(
    page_title="RENPHO Weight Monitor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# DEFAULTS
# -------------------------
DEFAULT_BASELINE_DATE = date(2026, 8, 1)
DEFAULT_BASELINE_WEIGHT = 112.0
DEFAULT_TARGET_WEIGHT = 72.0
DEFAULT_HEIGHT_M = 1.82

# -------------------------
# HELPERS
# -------------------------
def ts_at_midnight(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, dtime.min))

def format_delta(value: float | None, decimals: int = 2) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return f"{value:+.{decimals}f}"

def next_saturday(d: date) -> date:
    days_ahead = (5 - d.weekday()) % 7  # Sat=5
    return d + timedelta(days=days_ahead)

def add_vline_robust(fig: go.Figure, x_dt, text: str):
    x_py = pd.to_datetime(x_dt).to_pydatetime()
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=x_py,
        x1=x_py,
        y0=0,
        y1=1,
        line=dict(dash="dot"),
    )
    fig.add_annotation(
        x=x_py,
        y=1,
        xref="x",
        yref="paper",
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        yshift=6,
    )

# -------------------------
# SUPABASE
# -------------------------
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def load_manual() -> pd.DataFrame:
    supabase = get_supabase()
    res = supabase.table("manual_entries").select("*").order("date").execute()
    rows = res.data if res.data else []

    if not rows:
        return pd.DataFrame(columns=["id", "date", "weight", "bmi", "source"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["source"] = df["source"].fillna("manual")
    df = df.dropna(subset=["date", "weight"])
    return df.sort_values("date").reset_index(drop=True)

def insert_manual_entry(dt: pd.Timestamp, weight: float, bmi: float | None):
    supabase = get_supabase()
    payload = {
        "date": pd.Timestamp(dt).isoformat(),
        "weight": float(weight),
        "bmi": float(bmi) if bmi is not None and pd.notna(bmi) else None,
        "source": "manual",
    }
    supabase.table("manual_entries").insert(payload).execute()

def delete_manual_entries_by_id(ids: list[int]):
    if not ids:
        return
    supabase = get_supabase()
    supabase.table("manual_entries").delete().in_("id", ids).execute()

def clear_manual_entries():
    supabase = get_supabase()
    supabase.table("manual_entries").delete().neq("id", 0).execute()

# -------------------------
# LOADERS
# -------------------------
@st.cache_data(ttl=300, show_spinner="Scaricamento dati RENPHO in corso...")
def load_renpho_csv(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url, header=None)

    if raw.shape[1] == 1:
        s = raw[0].astype(str).str.strip()
        s = s.str.replace(r'^"|"$', "", regex=True)
        raw = s.str.split(",", expand=True)

    if raw.shape[1] < 3:
        raise ValueError("CSV RENPHO non riconosciuto. Mancano (data, ora, peso).")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(
        raw[0].astype(str).str.strip() + " " + raw[1].astype(str).str.strip(),
        errors="coerce",
        dayfirst=True,
    )

    w = raw[2].astype(str).str.strip()
    w = w.str.replace(",", ".", regex=False).str.replace("kg", "", regex=False).str.strip()
    df["weight"] = pd.to_numeric(w, errors="coerce")

    df = df.dropna(subset=["date", "weight"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    df["source"] = "renpho"
    df["bmi"] = np.nan
    df["id"] = np.nan
    return df.reset_index(drop=True)

def combine_data(renpho_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    renpho_df = renpho_df.copy()
    manual_df = manual_df.copy()

    if "id" not in renpho_df.columns:
        renpho_df["id"] = np.nan

    if manual_df.empty:
        df = renpho_df.copy()
    else:
        df = pd.concat([renpho_df, manual_df], ignore_index=True)
        df["__prio"] = df["source"].map({"renpho": 0, "manual": 1}).fillna(0)
        df = df.sort_values(["date", "__prio"]).drop(columns=["__prio"])
        df = df.drop_duplicates(subset=["date"], keep="last")

    return df.sort_values("date").reset_index(drop=True)

# -------------------------
# FEATURE ENGINEERING
# -------------------------
def add_bmi_if_missing(df: pd.DataFrame, height_m: float) -> pd.DataFrame:
    df = df.copy()
    if height_m <= 0:
        return df
    miss = df["bmi"].isna() & df["weight"].notna()
    df.loc[miss, "bmi"] = df.loc[miss, "weight"] / (height_m**2)
    return df

def make_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["day"] = d["date"].dt.date
    out = d.groupby("day", as_index=False).agg(
        date=("date", "max"),
        weight=("weight", "median"),
        bmi=("bmi", "median"),
    )
    out["source"] = "daily"
    return out.sort_values("date").reset_index(drop=True)

def pick_last_prefer_manual(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values("date").copy()
    last_day = df["date"].dt.date.max()
    day_df = df[df["date"].dt.date == last_day].copy()
    man = day_df[day_df["source"] == "manual"]
    if not man.empty:
        return man.sort_values("date").iloc[-1]
    return day_df.sort_values("date").iloc[-1]

# -------------------------
# FORECAST MODEL
# -------------------------
def fit_best_forecast_model(
    daily_df: pd.DataFrame,
    lookback_days: int = 90,
    smoothing_window: int = 7,
) -> dict:
    if daily_df.empty or len(daily_df) < 10:
        return {"ok": False, "reason": "too_few_points"}

    df = daily_df.sort_values("date").copy()

    end = df["date"].max()
    start = end - pd.Timedelta(days=lookback_days)
    sub = df[df["date"] >= start].copy()
    if len(sub) < 10:
        sub = df.copy()

    sub = sub.sort_values("date").reset_index(drop=True)

    # indice giornaliero continuo
    full_idx = pd.date_range(sub["date"].min().normalize(), sub["date"].max().normalize(), freq="D")
    s = sub.set_index(sub["date"].dt.normalize())["weight"].reindex(full_idx)

    # interpolazione giorni mancanti
    s = s.interpolate(method="time").ffill().bfill()

    # smoothing per ridurre rumore giornaliero
    s_smooth = s.rolling(window=smoothing_window, min_periods=1).mean()

    if len(s_smooth) < 10:
        return {"ok": False, "reason": "too_few_smoothed_points"}

    try:
        model = ExponentialSmoothing(
            s_smooth,
            trend="add",
            seasonal=None,
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=True)

        fitted = model.fittedvalues
        residuals = (s_smooth - fitted).dropna()
        resid_std = float(residuals.std()) if len(residuals) > 1 else 0.0

        # slope recente per controllo realismo
        recent = s_smooth.tail(min(21, len(s_smooth)))
        x = np.arange(len(recent), dtype=float)
        y = recent.values.astype(float)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        slope = float(slope)

        # clamp prudenziale max ±2kg/settimana
        max_daily_realistic = 2.0 / 7.0
        slope = float(np.clip(slope, -max_daily_realistic, max_daily_realistic))

        return {
            "ok": True,
            "type": "holt_damped",
            "model": model,
            "series": s_smooth,
            "last_date": s_smooth.index.max(),
            "last_weight": float(s_smooth.iloc[-1]),
            "recent_slope": slope,
            "resid_std": resid_std,
        }

    except Exception as e:
        # fallback robusto
        recent = s_smooth.tail(min(21, len(s_smooth)))
        x = np.arange(len(recent), dtype=float)
        y = recent.values.astype(float)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        slope = float(slope)

        max_daily_realistic = 2.0 / 7.0
        slope = float(np.clip(slope, -max_daily_realistic, max_daily_realistic))

        return {
            "ok": True,
            "type": "fallback_trend",
            "last_date": recent.index.max(),
            "last_weight": float(recent.iloc[-1]),
            "recent_slope": slope,
            "resid_std": 0.25,
            "fallback_reason": str(e),
        }

def forecast_series(model_dict: dict, start_date: pd.Timestamp, horizon_days: int) -> tuple[pd.DatetimeIndex, list[float]]:
    if not model_dict.get("ok", False):
        return pd.DatetimeIndex([]), []

    start_date = pd.to_datetime(start_date).normalize()
    future_dates = pd.date_range(start_date, start_date + pd.Timedelta(days=horizon_days), freq="D")

    if model_dict["type"] == "holt_damped":
        model = model_dict["model"]
        last_train_date = pd.to_datetime(model_dict["last_date"]).normalize()

        days_from_train_end = (future_dates[-1] - last_train_date).days
        days_from_train_end = max(days_from_train_end, 0)

        fc = model.forecast(days_from_train_end)
        fc_idx = pd.date_range(last_train_date + pd.Timedelta(days=1), periods=len(fc), freq="D")
        fc_series = pd.Series(fc.values, index=fc_idx)

        values = []
        for d in future_dates:
            if d <= last_train_date:
                values.append(float(model_dict["last_weight"]))
            else:
                values.append(float(fc_series.loc[d]))
        return future_dates, values

    # fallback trend smorzato
    weight = float(model_dict["last_weight"])
    delta = float(model_dict["recent_slope"])
    damping = 0.985

    values = []
    for i, d in enumerate(future_dates):
        if i == 0:
            values.append(weight)
        else:
            weight += delta
            delta *= damping
            values.append(float(weight))
    return future_dates, values

def predict_weight(model_dict: dict, when: pd.Timestamp) -> float:
    when = pd.to_datetime(when).normalize()
    dates, values = forecast_series(model_dict, start_date=when, horizon_days=0)
    if len(values) == 0:
        return np.nan
    return float(values[0])

def estimate_target_date(model_dict: dict, target_weight: float, start_from: date, max_horizon_days: int = 365) -> tuple[date | None, int | None]:
    start_ts = pd.Timestamp(start_from)
    dates, values = forecast_series(model_dict, start_date=start_ts, horizon_days=max_horizon_days)

    if len(values) == 0:
        return None, None

    for d, w in zip(dates, values):
        if w <= float(target_weight):
            target_date = pd.to_datetime(d).date()
            days_to_target = (target_date - start_from).days
            return target_date, days_to_target

    return None, None

# -------------------------
# SECRETS
# -------------------------
csv_url = st.secrets.get("CSV_URL", "")
if not csv_url:
    st.error("CSV_URL non impostato. Mettilo nei Secrets.")
    st.stop()

if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
    st.error("SUPABASE_URL o SUPABASE_KEY mancanti nei Secrets.")
    st.stop()

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("⚙️ Impostazioni")

    target_weight = st.number_input("🎯 Target (kg)", value=float(DEFAULT_TARGET_WEIGHT), step=0.5)
    baseline_weight = st.number_input("⚑ Peso al 01/08 (kg)", value=float(DEFAULT_BASELINE_WEIGHT), step=0.5)
    baseline_date = st.date_input("⚑ Data baseline", value=DEFAULT_BASELINE_DATE)

    st.divider()
    height_m = st.number_input("📏 Altezza (m) per BMI", value=float(DEFAULT_HEIGHT_M), step=0.01)

    st.divider()
    lookback_days = st.selectbox("🔍 Finestra modello (giorni)", [30, 45, 60, 90, 120, 180], index=3)
    ma_window = st.selectbox("📈 Media mobile visuale (giorni)", [7, 14, 21, 30], index=0)
    forecast_smoothing = st.selectbox("🧠 Smoothing forecast", [3, 5, 7, 10], index=2)

    st.divider()
    forecast_horizon = st.selectbox("⏳ Orizzonte forecast (giorni)", [30, 60, 90, 180], index=1)

    st.divider()
    show_raw = st.toggle("Mostra punti RAW", value=True)
    show_daily = st.toggle("Mostra linea DAILY", value=True)
    prevent_upward_forecast = st.toggle("Blocca forecast crescente", value=True)

    st.divider()
    if st.button("🔄 Forza refresh RENPHO", use_container_width=True):
        load_renpho_csv.clear()
        st.rerun()

# -------------------------
# LOAD DATA
# -------------------------
try:
    renpho = load_renpho_csv(csv_url)
except Exception as e:
    st.error(f"Errore caricamento RENPHO: {e}")
    st.stop()

try:
    manual = load_manual()
except Exception as e:
    st.error(f"Errore caricamento dati manuali da Supabase: {e}")
    st.stop()

df = combine_data(renpho, manual)
df = add_bmi_if_missing(df, height_m)

if df.empty:
    st.warning("Nessun dato disponibile.")
    st.stop()

daily = make_daily_series(df)
daily["ma"] = daily["weight"].rolling(ma_window, min_periods=1).mean()

# -------------------------
# DATE FILTER
# -------------------------
min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "📅 Intervallo analisi",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

df_f = df[(df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].copy()
daily_f = daily[(daily["date"].dt.date >= start_d) & (daily["date"].dt.date <= end_d)].copy()

if df_f.empty:
    st.warning("L'intervallo selezionato non contiene dati.")
    st.stop()

daily_f["ma"] = daily_f["weight"].rolling(ma_window, min_periods=1).mean()

# -------------------------
# OGGI + DOMANI
# -------------------------
today = date.today()
tomorrow = today + timedelta(days=1)
tomorrow_ts = ts_at_midnight(tomorrow)

# -------------------------
# LAST & PREVIOUS MEASURE
# -------------------------
last_row = pick_last_prefer_manual(df_f)
last_dt = pd.to_datetime(last_row["date"])
last_weight = float(last_row["weight"])
last_bmi = float(last_row["bmi"]) if pd.notna(last_row["bmi"]) else np.nan

prev_df = df_f[df_f["date"] < last_dt].sort_values("date")
prev_row = prev_df.iloc[-1] if not prev_df.empty else None

if prev_row is not None:
    prev_weight = float(prev_row["weight"])
    prev_bmi = float(prev_row["bmi"]) if pd.notna(prev_row["bmi"]) else np.nan
else:
    prev_weight, prev_bmi = None, None

delta_bmi = (last_bmi - prev_bmi) if (prev_bmi is not None and np.isfinite(last_bmi) and np.isfinite(prev_bmi)) else None

loss_from_baseline = float(baseline_weight - last_weight)
dist_to_target = float(last_weight - float(target_weight))

if prev_weight is not None:
    loss_prev = float(baseline_weight - prev_weight)
    dist_prev = float(prev_weight - float(target_weight))
    delta_loss = float(loss_from_baseline - loss_prev)
    delta_dist = float(dist_to_target - dist_prev)
else:
    delta_loss, delta_dist = None, None

# -------------------------
# MODEL + FORECAST
# -------------------------
model = fit_best_forecast_model(
    daily_f,
    lookback_days=lookback_days,
    smoothing_window=forecast_smoothing,
)

if model.get("ok", False) and prevent_upward_forecast:
    if model["type"] == "holt_damped" and model.get("recent_slope", 0) > 0:
        model = dict(model)
        model["recent_slope"] = 0.0
    elif model["type"] == "fallback_trend" and model.get("recent_slope", 0) > 0:
        model = dict(model)
        model["recent_slope"] = 0.0

pred_tomorrow = None
if model.get("ok", False):
    tmp_dates, tmp_values = forecast_series(model, start_date=tomorrow_ts, horizon_days=0)
    if len(tmp_values) > 0:
        pred_tomorrow = float(tmp_values[0])

target_date_est, days_to_target = estimate_target_date(
    model,
    target_weight=float(target_weight),
    start_from=today,
    max_horizon_days=365,
) if model.get("ok", False) else (None, None)

# -------------------------
# TABS
# -------------------------
st.title("📉 RENPHO — Weight Monitor")
tab_dash, tab_manual, tab_forecast, tab_data = st.tabs(
    ["📊 Cruscotto", "✍️ Manuale", "🔮 Forecast", "🧾 Dataset"]
)

# -------------------------
# CRUSCOTTO
# -------------------------
with tab_dash:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Ultima misura",
        f"{last_weight:.2f} kg",
        f"{last_dt.strftime('%d %b %H:%M')}",
        delta_color="off",
    )

    c2.metric(
        "BMI (ultima misura)",
        f"{last_bmi:.2f}" if np.isfinite(last_bmi) else "—",
        (format_delta(delta_bmi, 2) + " vs ultima misura") if delta_bmi is not None else "—",
        delta_color="inverse",
    )

    c3.metric(
        "Perdita vs baseline",
        f"{loss_from_baseline:+.2f} kg",
        (format_delta(delta_loss, 2) + " kg vs ultima misura") if delta_loss is not None else "—",
        delta_color="normal",
    )

    c4.metric(
        "Distanza dal target",
        f"{dist_to_target:+.2f} kg",
        (format_delta(delta_dist, 2) + " kg vs ultima misura") if delta_dist is not None else "—",
        delta_color="inverse",
    )

    c5.metric(
        "Data target stimata",
        target_date_est.strftime("%d %b") if target_date_est else "—",
        (f"{days_to_target} giorni (da oggi)") if days_to_target is not None else "—",
    )

    c6.metric(
        f"Forecast {tomorrow.strftime('%d %b')}",
        f"{pred_tomorrow:.2f} kg" if pred_tomorrow is not None else "—",
        "modello Holt smorzato",
    )

    st.markdown("---")
    st.subheader("Trend peso + forecasting")

    fig = go.Figure()

    if show_raw:
        fig.add_trace(go.Scatter(
            x=df_f["date"], y=df_f["weight"],
            mode="markers",
            name="RAW",
            hovertemplate="Data: %{x}<br>Peso: %{y:.2f} kg<extra></extra>",
        ))

    if show_daily:
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["weight"],
            mode="lines+markers",
            name="DAILY (mediana/giorno)",
            hovertemplate="Giorno: %{x}<br>Peso daily: %{y:.2f} kg<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=daily_f["date"], y=daily_f["ma"],
            mode="lines",
            name=f"MA({ma_window}gg)",
            hovertemplate="Giorno: %{x}<br>MA: %{y:.2f} kg<extra></extra>",
        ))

    fig.add_hline(
        y=float(target_weight),
        line_dash="dash",
        annotation_text="🎯 Target",
        annotation_position="bottom right",
    )

    add_vline_robust(
        fig,
        x_dt=last_dt,
        text=f"Ultima misura ({last_dt.strftime('%d %b %H:%M')})",
    )

    if model.get("ok", False) and len(daily_f) >= 2:
        last_daily_dt = daily_f["date"].max().normalize()
        horizon_days = int(forecast_horizon)
        future_dates, y_fore = forecast_series(
            model,
            start_date=last_daily_dt,
            horizon_days=horizon_days,
        )

        fig.add_trace(go.Scatter(
            x=future_dates, y=y_fore,
            mode="lines",
            name=f"Forecast ({horizon_days}g)",
            line=dict(dash="dash"),
            hovertemplate="Data: %{x}<br>Forecast: %{y:.2f} kg<extra></extra>",
        ))

    if pred_tomorrow is not None:
        fig.add_trace(go.Scatter(
            x=[tomorrow_ts.to_pydatetime()],
            y=[pred_tomorrow],
            mode="markers+text",
            name="Forecast domani",
            text=[f"{pred_tomorrow:.2f} kg"],
            textposition="top center",
            hovertemplate="Data: %{x}<br>Forecast: %{y:.2f} kg<extra></extra>",
        ))

    fig.update_layout(
        font=dict(size=11),
        xaxis_title="Data",
        yaxis_title="Peso (kg)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Ultime 10 misure")

    last10 = df_f.sort_values("date", ascending=False).head(10).copy()
    last10["Data"] = last10["date"].dt.strftime("%Y-%m-%d %H:%M")
    last10["Peso (kg)"] = last10["weight"].map(lambda x: f"{x:.2f}")
    last10["BMI"] = last10["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    last10["Origine"] = last10["source"]

    st.dataframe(last10[["Data", "Peso (kg)", "BMI", "Origine"]], use_container_width=True, hide_index=True)

# -------------------------
# MANUALE
# -------------------------
with tab_manual:
    st.subheader("➕ Inserimento manuale (Peso + BMI)")
    st.caption("Se BMI=0 verrà calcolato automaticamente con altezza.")

    with st.form("manual_form", clear_on_submit=True):
        colA, colB, colC, colD = st.columns(4)
        m_date = colA.date_input("Data", value=date.today())
        m_time = colB.time_input("Ora", value=datetime.now().time().replace(second=0, microsecond=0))
        m_weight = colC.number_input("Peso (kg)", min_value=0.0, value=float(last_weight), step=0.1)
        m_bmi = colD.number_input("BMI (0 = auto)", min_value=0.0, value=0.0, step=0.1)

        submitted = st.form_submit_button("✅ Salva", use_container_width=True)
        if submitted:
            try:
                dt = pd.Timestamp(datetime.combine(m_date, m_time))
                bmi_val = float(m_bmi) if float(m_bmi) > 0 else (
                    float(m_weight) / (height_m**2) if height_m > 0 else np.nan
                )

                insert_manual_entry(dt, float(m_weight), bmi_val)
                st.success("Misura salvata in modo permanente.")
                st.rerun()
            except Exception as e:
                st.error(f"Errore salvataggio misura manuale: {e}")

    st.markdown("---")
    st.subheader("🗑️ Cancella misure manuali")

    manual_now = load_manual()
    if manual_now.empty:
        st.info("Nessuna misura manuale salvata.")
    else:
        tmp = manual_now.copy()
        tmp["label"] = (
            tmp["date"].dt.strftime("%Y-%m-%d %H:%M")
            + " | "
            + tmp["weight"].map(lambda x: f"{x:.2f} kg")
        )

        selected_labels = st.multiselect(
            "Seleziona i record da cancellare",
            options=tmp["label"].tolist()
        )

        selected_ids = tmp.loc[tmp["label"].isin(selected_labels), "id"].astype(int).tolist()

        col1, col2 = st.columns(2)
        if col1.button("🗑️ Cancella selezionate", use_container_width=True, type="primary"):
            if selected_ids:
                try:
                    delete_manual_entries_by_id(selected_ids)
                    st.success("Misure selezionate cancellate.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Errore cancellazione: {e}")
            else:
                st.warning("Seleziona almeno un record.")

        if col2.button("⚠️ Cancella TUTTO", use_container_width=True):
            try:
                clear_manual_entries()
                st.success("Archivio manuale azzerato.")
                st.rerun()
            except Exception as e:
                st.error(f"Errore cancellazione totale: {e}")

        st.markdown("---")
        st.subheader("Archivio misure manuali")

        show_manual = tmp.sort_values("date", ascending=False).copy()
        show_manual["Data"] = show_manual["date"].dt.strftime("%Y-%m-%d %H:%M")
        show_manual["Peso (kg)"] = show_manual["weight"].map(lambda x: f"{x:.2f}")
        show_manual["BMI"] = show_manual["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
        show_manual["Origine"] = show_manual["source"]

        st.dataframe(
            show_manual[["Data", "Peso (kg)", "BMI", "Origine"]],
            use_container_width=True,
            hide_index=True,
        )

# -------------------------
# FORECAST
# -------------------------
with tab_forecast:
    st.subheader("Forecast settimanale (sabati) fino al target")

    if not model.get("ok", False):
        st.warning("Modello non disponibile (pochi punti o dati insufficienti).")
    elif not target_date_est:
        st.warning("Data target non stimabile entro 365 giorni.")
    else:
        start_sat = next_saturday(today)
        if start_sat <= today:
            start_sat = start_sat + timedelta(days=7)

        if target_date_est < start_sat:
            st.info(f"Il target è stimato prima del prossimo sabato ({start_sat.strftime('%d %b')}).")
        else:
            st.success(f"Sabati (da {start_sat.strftime('%d %b')}) fino al target stimato: {target_date_est.strftime('%d %b')}")

            rows = []
            s = start_sat
            while s <= target_date_est:
                ts = ts_at_midnight(s)
                future_dates, future_vals = forecast_series(model, start_date=ts, horizon_days=0)
                w_pred = float(future_vals[0]) if len(future_vals) > 0 else np.nan

                rows.append({
                    "Sabato": s.strftime("%d %b"),
                    "Peso previsto (kg)": round(w_pred, 2),
                    "Distanza dal target (kg)": round(w_pred - float(target_weight), 2),
                })
                s += timedelta(days=7)

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if model["type"] == "holt_damped":
            st.caption(
                f"Modello usato: Holt con trend smorzato | volatilità residua stimata: {model.get('resid_std', 0):.2f} kg"
            )
        else:
            st.caption("Modello usato: fallback trend smorzato")

# -------------------------
# DATASET
# -------------------------
with tab_data:
    st.subheader("Dataset (tutte le misure) — più recente in alto")

    out = df_f.sort_values("date", ascending=False).copy()
    out["Data"] = out["date"].dt.strftime("%Y-%m-%d %H:%M")
    out["Peso (kg)"] = out["weight"].map(lambda x: f"{x:.2f}")
    out["BMI"] = out["bmi"].map(lambda x: f"{x:.2f}" if pd.notna(x) and np.isfinite(x) else "")
    out["Origine"] = out["source"]

    st.dataframe(out[["Data", "Peso (kg)", "BMI", "Origine"]], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Scarica CSV (filtrato)",
        data=out[["Data", "Peso (kg)", "BMI", "Origine"]].to_csv(index=False).encode("utf-8"),
        file_name="renpho_dataset_export.csv",
        mime="text/csv",
        use_container_width=True,
    )