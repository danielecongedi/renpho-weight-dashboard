import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
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

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MANUAL_FILE = DATA_DIR / "manual_entries.csv"

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

def safe_add_vline(fig: go.Figure, x, **kwargs):
    """
    Fix per bug/compatibilità plotly+pandas Timestamp: add_vline prova a fare sum(x)
    e scoppia con Timestamp. Converte a datetime python.
    """
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        x = pd.to_datetime(x).to_pydatetime()
    fig.add_vline(x=x, **kwargs)

# -------------------------
# LOADERS
# -------------------------
@st.cache_data(ttl=300, show_spinner="Scaricamento dati RENPHO in corso...")
def load_renpho_csv(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url, header=None)

    # spesso è 1 colonna unica -> split
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
    return df

def load_manual() -> pd.DataFrame:
    if not MANUAL_FILE.exists():
        return pd.DataFrame(columns=["date", "weight", "bmi", "source"])
    df = pd.read_csv(MANUAL_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["source"] = "manual"
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df

def save_manual(df: pd.DataFrame) -> None:
    out = df.copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out.to_csv(MANUAL_FILE, index=False)

def combine_data(renpho_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    if manual_df.empty:
        df = renpho_df.copy()
    else:
        df = pd.concat([renpho_df, manual_df], ignore_index=True)
        # priorità manual in caso di stesso timestamp
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

def day_before_value(daily_df: pd.DataFrame, day: date) -> pd.Series | None:
    prev = day - timedelta(days=1)
    row = daily_df[daily_df["date"].dt.date == prev]
    if row.empty:
        return None
    return row.iloc[-1]

# -------------------------
# MODEL (stabilizzato)
# -------------------------
def fit_linear_model(daily_df: pd.DataFrame, lookback_days: int) -> dict:
    if daily_df.empty or len(daily_df) < 2:
        return {"ok": False, "reason": "no_data"}

    end = daily_df["date"].max()
    start = end - pd.Timedelta(days=lookback_days)
    sub = daily_df[daily_df["date"] >= start].copy()
    if len(sub) < 2:
        sub = daily_df.copy()

    if len(sub) < 7:
        return {"ok": False, "reason": f"too_few_points ({len(sub)})"}

    t0 = sub["date"].min()
    x = (sub["date"] - t0).dt.total_seconds().values / 86400.0
    y = sub["weight"].values

    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    m = float(m)
    b = float(b)

    # clamp ±2 kg/settimana
    max_week = 2.0
    if abs(m * 7.0) > max_week:
        m = np.sign(m) * (max_week / 7.0)

    return {"ok": True, "m": m, "b": b, "t0": t0, "sub": sub}

def predict_weight(model: dict, when: pd.Timestamp) -> float:
    x = (when - model["t0"]).total_seconds() / 86400.0
    return float(model["m"] * x + model["b"])

# -------------------------
# SECRETS
# -------------------------
csv_url = st.secrets.get("CSV_URL", "")
if not csv_url:
    st.error("CSV_URL non impostato. Mettilo nei Secrets.")
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
    lookback_days = st.selectbox("🔍 Finestra modello (giorni)", [30, 45, 60, 90, 120, 180], index=2)
    ma_window = st.selectbox("📈 Media mobile (giorni)", [7, 14, 21, 30], index=0)

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

manual = load_manual()
df = combine_data(renpho, manual)
df = add_bmi_if_missing(df, height_m)

if df.empty:
    st.warning("Nessun dato disponibile.")
    st.stop()

daily = make_daily_series(df)
daily["ma"] = daily["weight"].rolling(ma_window, min_periods=1).mean()

# -------------------------
# DATE FILTER (INTERVALLO ANALISI)
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
# "OGGI" REALE (si aggiorna ogni giorno)
# -------------------------
today = date.today()
tomorrow = today + timedelta(days=1)
tomorrow_ts = ts_at_midnight(tomorrow)

# -------------------------
# CURRENT "LAST" (prefer manual for last day in filtered data)
# -------------------------
last_row = pick_last_prefer_manual(df_f)
last_day = last_row["date"].date()
last_weight = float(last_row["weight"])
last_bmi = float(last_row["bmi"]) if pd.notna(last_row["bmi"]) else np.nan

yesterday_row = day_before_value(daily_f, last_day)
if yesterday_row is not None:
    w_y = float(yesterday_row["weight"])
    bmi_y = float(yesterday_row["bmi"]) if pd.notna(yesterday_row["bmi"]) else np.nan
else:
    w_y, bmi_y = None, None

delta_w = (last_weight - w_y) if w_y is not None else None
delta_bmi = (last_bmi - bmi_y) if (bmi_y is not None and np.isfinite(last_bmi) and np.isfinite(bmi_y)) else None

loss_from_baseline = float(baseline_weight - last_weight)
dist_to_target = float(last_weight - float(target_weight))

loss_y = (baseline_weight - w_y) if w_y is not None else None
delta_loss = (loss_from_baseline - loss_y) if loss_y is not None else None

dist_y = (w_y - float(target_weight)) if w_y is not None else None
delta_dist = (dist_to_target - dist_y) if dist_y is not None else None

# -------------------------
# MODEL + FORECAST
# -------------------------
model = fit_linear_model(daily_f, lookback_days=lookback_days)

if model.get("ok", False) and prevent_upward_forecast and model["m"] > 0:
    model = dict(model)
    model["m"] = 0.0

# stima data target (riferita a "today" come riferimento umano)
target_date_est = None
days_to_target = None
if model.get("ok", False) and model["m"] < 0:
    x_target = (float(target_weight) - model["b"]) / model["m"]
    if np.isfinite(x_target) and x_target >= 0:
        target_date_est = (model["t0"] + pd.Timedelta(days=float(x_target))).date()
        days_to_target = (target_date_est - today).days

# forecast del giorno successivo a "oggi" (sempre, anche se non inserisco misure)
pred_tomorrow = None
if model.get("ok", False):
    pred_tomorrow = predict_weight(model, tomorrow_ts)

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
        f"{last_row['date'].strftime('%d %b %H:%M')}",
        delta_color="off",
    )

    c2.metric(
        "BMI (ultima misura)",
        f"{last_bmi:.2f}" if np.isfinite(last_bmi) else "—",
        (format_delta(delta_bmi, 2) + " vs ieri") if delta_bmi is not None else "—",
        delta_color="inverse",
    )

    c3.metric(
        "Perdita vs baseline",
        f"{loss_from_baseline:+.2f} kg",
        (format_delta(delta_loss, 2) + " kg vs ieri") if delta_loss is not None else "—",
        delta_color="normal",
    )

    c4.metric(
        "Distanza dal target",
        f"{dist_to_target:+.2f} kg",
        (format_delta(delta_dist, 2) + " kg vs ieri") if delta_dist is not None else "—",
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
        "giorno successivo a oggi",
    )

    st.markdown("---")
    st.subheader("Trend peso + forecasting (modello su DAILY)")

    fig = go.Figure()

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=df_f["date"],
                y=df_f["weight"],
                mode="markers",
                name="RAW",
                hovertemplate="Data: %{x}<br>Peso: %{y:.2f} kg<extra></extra>",
            )
        )

    if show_daily:
        fig.add_trace(
            go.Scatter(
                x=daily_f["date"],
                y=daily_f["weight"],
                mode="lines+markers",
                name="DAILY (mediana/giorno)",
                hovertemplate="Giorno: %{x}<br>Peso daily: %{y:.2f} kg<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily_f["date"],
                y=daily_f["ma"],
                mode="lines",
                name=f"MA({ma_window}gg)",
                hovertemplate="Giorno: %{x}<br>MA: %{y:.2f} kg<extra></extra>",
            )
        )

    fig.add_hline(
        y=float(target_weight),
        line_dash="dash",
        annotation_text="🎯 Target",
        annotation_position="bottom right",
    )

    # FIX errore Timestamp: converti a datetime python via helper
    safe_add_vline(
        fig,
        x=last_row["date"],
        line_dash="dot",
        annotation_text=f"Ultima misura ({last_row['date'].strftime('%d %b %H:%M')})",
        annotation_position="top left",
    )

    # forecast (orizzonte selezionabile)
    if model.get("ok", False) and len(daily_f) >= 2:
        last_dt = daily_f["date"].max()
        horizon_days = int(forecast_horizon)
        future_dates = pd.date_range(last_dt, last_dt + pd.Timedelta(days=horizon_days), freq="D")
        y_fore = [predict_weight(model, d) for d in future_dates]

        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=y_fore,
                mode="lines",
                name=f"Forecast ({horizon_days}g)",
                line=dict(dash="dash"),
                hovertemplate="Data: %{x}<br>Forecast: %{y:.2f} kg<extra></extra>",
            )
        )

    # Punto forecast di domani (sempre relativo a "oggi")
    if model.get("ok", False) and pred_tomorrow is not None:
        fig.add_trace(
            go.Scatter(
                x=[tomorrow_ts.to_pydatetime()],
                y=[pred_tomorrow],
                mode="markers+text",
                name="Forecast domani",
                text=[f"{pred_tomorrow:.2f} kg"],
                textposition="top center",
                hovertemplate="Data: %{x}<br>Forecast: %{y:.2f} kg<extra></extra>",
            )
        )

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

    st.dataframe(
        last10[["Data", "Peso (kg)", "BMI", "Origine"]],
        use_container_width=True,
        hide_index=True,
    )

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
            dt = pd.Timestamp(datetime.combine(m_date, m_time))
            bmi_val = float(m_bmi) if float(m_bmi) > 0 else (
                float(m_weight) / (height_m**2) if height_m > 0 else np.nan
            )

            new_row = pd.DataFrame(
                [{
                    "date": dt,
                    "weight": float(m_weight),
                    "bmi": bmi_val,
                    "source": "manual",
                }]
            )

            manual_now = load_manual()
            manual_now = pd.concat([manual_now, new_row], ignore_index=True)
            manual_now = manual_now.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            save_manual(manual_now)

            st.success("Misura salvata. Aggiorno…")
            load_renpho_csv.clear()
            st.rerun()

    st.markdown("---")
    st.subheader("🗑️ Cancella misure manuali")

    manual_now = load_manual()
    if manual_now.empty:
        st.info("Nessuna misura manuale salvata.")
    else:
        tmp = manual_now.copy()
        tmp["date_str"] = tmp["date"].dt.strftime("%Y-%m-%d %H:%M")
        to_delete = st.multiselect("Seleziona i record da cancellare", options=tmp["date_str"].tolist())

        col1, col2 = st.columns(2)
        if col1.button("🗑️ Cancella selezionate", use_container_width=True, type="primary"):
            if to_delete:
                keep = ~tmp["date_str"].isin(to_delete)
                manual_new = tmp.loc[keep, ["date", "weight", "bmi", "source"]].copy()
                save_manual(manual_new)
                st.success("Cancellate. Aggiorno…")
                st.rerun()
            else:
                st.warning("Seleziona almeno un record.")

        if col2.button("⚠️ Cancella TUTTO", use_container_width=True):
            save_manual(pd.DataFrame(columns=["date", "weight", "bmi", "source"]))
            st.success("Manuale azzerato. Aggiorno…")
            st.rerun()

# -------------------------
# FORECAST (sabati fino a target) — rimuovi sabati già passati rispetto a OGGI
# -------------------------
with tab_forecast:
    st.subheader("Forecast settimanale (sabati) fino al target")

    if not model.get("ok", False):
        st.warning("Modello non disponibile (pochi punti o dati insufficienti).")
    elif not target_date_est:
        st.warning("Data target non stimabile: non posso calcolare i sabati fino al target.")
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
                w_pred = predict_weight(model, ts)
                rows.append(
                    {
                        "Sabato": s.strftime("%d %b"),
                        "Peso previsto (kg)": round(w_pred, 2),
                        "Distanza dal target (kg)": round(w_pred - float(target_weight), 2),
                    }
                )
                s += timedelta(days=7)

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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