import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta, date

# -------------------------
# PAGE
# -------------------------
st.set_page_config(
    page_title="RENPHO Weight Monitor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# DEFAULTS
# -------------------------
DEFAULT_BASELINE_DATE = date(2026, 8, 1)  # <- qui: 01/08/2026
DEFAULT_BASELINE_WEIGHT = 112.0
DEFAULT_TARGET_WEIGHT = 72.0

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MANUAL_FILE = DATA_DIR / "manual_entries.csv"


# -------------------------
# HELPERS
# -------------------------
@st.cache_data(ttl=300, show_spinner="Scaricamento dati RENPHO in corso...")
def load_renpho_csv(url: str) -> pd.DataFrame:
    """
    Parser robusto per export RENPHO come il tuo file:
    - niente header
    - ogni riga è una stringa tra virgolette con campi separati da virgola
      "YYYY.MM.DD,HH:MM:SS,81.50,24.6,..."
    Prendiamo solo: data+ora e peso (campo 2).
    """
    raw = pd.read_csv(url, header=None)

    # spesso viene letto come una colonna unica -> split
    if raw.shape[1] == 1:
        s = raw[0].astype(str).str.strip()
        s = s.str.replace(r'^"|"$', "", regex=True)  # rimuove virgolette
        raw = s.str.split(",", expand=True)

    if raw.shape[1] < 3:
        raise ValueError("CSV RENPHO non riconosciuto. Mancano i campi base (data, ora, peso).")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(
        raw[0].astype(str).str.strip() + " " + raw[1].astype(str).str.strip(),
        errors="coerce",
        dayfirst=True
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
        df["__prio"] = df["source"].map({"renpho": 0, "manual": 1}).fillna(0)
        df = df.sort_values(["date", "__prio"]).drop(columns=["__prio"])
        df = df.drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date").reset_index(drop=True)


def fit_linear_model(df: pd.DataFrame, lookback_days: int) -> dict:
    """
    Modello "ML" semplice, spiegabile e stabile: regressione lineare su una finestra temporale.
    """
    if df.empty:
        return {"ok": False}

    end = df["date"].max()
    start = end - pd.Timedelta(days=lookback_days)
    sub = df[df["date"] >= start].copy()
    if len(sub) < 2:
        sub = df.copy()
    if len(sub) < 2:
        return {"ok": False}

    t0 = sub["date"].min()
    x = (sub["date"] - t0).dt.total_seconds().values / 86400.0
    y = sub["weight"].values

    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"ok": True, "m": m, "b": b, "t0": t0, "sub": sub}


def predict_weight(model: dict, when: pd.Timestamp) -> float:
    x = (when - model["t0"]).total_seconds() / 86400.0
    return float(model["m"] * x + model["b"])


def next_saturday(d: date) -> date:
    days_ahead = (5 - d.weekday()) % 7
    return d + timedelta(days=days_ahead)


# -------------------------
# LOAD URL
# -------------------------
csv_url = st.secrets.get("CSV_URL", "")
if not csv_url:
    st.error("CSV_URL non impostato. Mettilo nei Secrets (.streamlit/secrets.toml o Streamlit Cloud).")
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
    lookback_days = st.selectbox("🔍 Finestra modello (giorni)", [30, 45, 60, 90, 120, 180], index=2)
    ma_window = st.selectbox("📈 Media mobile (giorni)", [7, 14, 21, 30], index=0)

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
if df.empty:
    st.warning("Nessun dato disponibile.")
    st.stop()

# date filter
min_date = df["date"].min().date()
max_date = df["date"].max().date()
date_range = st.sidebar.date_input("📅 Intervallo analisi", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

df_f = df[(df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)].copy()
if df_f.empty:
    st.warning("L'intervallo selezionato non contiene dati.")
    st.stop()

df_f["ma"] = df_f["weight"].rolling(ma_window, min_periods=1).mean()

latest = df_f.iloc[-1]
latest_w = float(latest["weight"])
latest_dt = latest["date"]

# perdita positiva = baseline - ultimo
loss_from_baseline = float(baseline_weight - latest_w)
dist_to_target = float(latest_w - float(target_weight))

# ML model
model = fit_linear_model(df_f, lookback_days=lookback_days)
target_date_est = None
target_date_str = "—"

if model.get("ok", False):
    m = model["m"]
    b = model["b"]
    if m < 0:
        x_target = (float(target_weight) - b) / m
        if np.isfinite(x_target) and x_target >= 0:
            target_date_est = (model["t0"] + pd.Timedelta(days=float(x_target))).date()
            target_date_str = target_date_est.strftime("%d %b %Y")

# -------------------------
# TABS
# -------------------------
st.title("📉 RENPHO — Weight Monitor")
tab_dash, tab_manual, tab_forecast, tab_data = st.tabs(["📊 Cruscotto", "✍️ Manuale", "🔮 Forecast & Sabati", "🧾 Dataset"])

# --- DASHBOARD ---
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ultima misura", f"{latest_w:.1f} kg", latest_dt.strftime("%d/%m %H:%M"))

    # delta_color inverse: se "Perdita" è positiva, la vogliamo verde -> NON usare inverse
    # qui mostriamo come delta "Perdita" rispetto a baseline
    c2.metric("Perdita dal 01/08", f"{loss_from_baseline:+.1f} kg", f"baseline {baseline_weight:.1f} kg", delta_color="normal")
    c3.metric("Distanza dal target", f"{dist_to_target:+.1f} kg", f"target {target_weight:.1f} kg", delta_color="inverse")
    c4.metric("Data target stimata (ML)", target_date_str, f"ultimi {lookback_days} gg" if target_date_est else "trend non calante")

    st.markdown("---")
    st.subheader("📈 Trend peso + modello + forecasting")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_f["date"], y=df_f["weight"],
        mode="lines+markers",
        name="Peso",
        hovertemplate="Data: %{x}<br>Peso: %{y:.1f} kg<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=df_f["date"], y=df_f["ma"],
        mode="lines",
        name=f"Media mobile ({ma_window}gg)",
        hovertemplate="Data: %{x}<br>MA: %{y:.1f} kg<extra></extra>"
    ))

    fig.add_hline(
        y=float(target_weight),
        line_dash="dash",
        annotation_text="🎯 Target",
        annotation_position="bottom right"
    )

    if model.get("ok", False):
        sub = model["sub"]
        y_fit = [predict_weight(model, d) for d in sub["date"]]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=y_fit,
            mode="lines",
            name="Trend (ML)",
            line=dict(dash="dot")
        ))

        last_dt = df_f["date"].max()
        if target_date_est:
            horizon_days = (target_date_est - last_dt.date()).days + 10
        else:
            horizon_days = 180
        horizon_days = int(max(30, min(horizon_days, 365)))

        future_dates = pd.date_range(last_dt, last_dt + pd.Timedelta(days=horizon_days), freq="D")
        y_fore = [predict_weight(model, d) for d in future_dates]

        fig.add_trace(go.Scatter(
            x=future_dates, y=y_fore,
            mode="lines",
            name="Forecast",
            line=dict(dash="dash")
        ))

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Peso (kg)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

# --- MANUAL ENTRY ---
with tab_manual:
    st.subheader("➕ Inserimento manuale (Peso + BMI)")
    st.caption("I dati manuali hanno priorità sui RENPHO se stesso timestamp.")

    with st.form("manual_form", clear_on_submit=True):
        colA, colB, colC, colD = st.columns(4)
        m_date = colA.date_input("Data", value=date.today())
        m_time = colB.time_input("Ora", value=datetime.now().time().replace(second=0, microsecond=0))
        m_weight = colC.number_input("Peso (kg)", min_value=0.0, value=float(latest_w), step=0.1)
        m_bmi = colD.number_input("BMI (opzionale)", min_value=0.0, value=0.0, step=0.1, help="Lascia 0 per ignorare.")

        submitted = st.form_submit_button("✅ Salva", use_container_width=True)
        if submitted:
            dt = pd.Timestamp(datetime.combine(m_date, m_time))
            bmi_val = float(m_bmi) if float(m_bmi) > 0 else np.nan

            new_row = pd.DataFrame([{
                "date": dt,
                "weight": float(m_weight),
                "bmi": bmi_val,
                "source": "manual"
            }])

            manual_now = load_manual()
            manual_now = pd.concat([manual_now, new_row], ignore_index=True)
            manual_now = manual_now.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            save_manual(manual_now)

            st.success("Misura salvata. Aggiorno…")
            load_renpho_csv.clear()
            st.rerun()

    st.divider()
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

# --- FORECAST SATURDAYS ---
with tab_forecast:
    st.subheader("🗓️ Tabella sabati fino al target")

    if not model.get("ok", False):
        st.warning("Dati insufficienti per il modello.")
    elif not target_date_est:
        st.warning("Trend non in calo: impossibile stimare data target.")
    else:
        st.success(f"Target {target_weight:.1f} kg stimato per: **{target_date_str}**")

        start_ref = max(date.today(), df_f["date"].max().date())
        sat = next_saturday(start_ref)

        rows = []
        while sat <= target_date_est:
            ts = pd.Timestamp(datetime.combine(sat, datetime.min.time()))
            w_pred = predict_weight(model, ts)
            rows.append({
                "Sabato": sat.strftime("%d %b %Y"),
                "Peso previsto (kg)": round(w_pred, 1),
                "Mancanti (kg)": round(w_pred - float(target_weight), 1)
            })
            sat += timedelta(days=7)

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Nessun sabato nel range: target molto vicino.")

# --- DATASET ---
with tab_data:
    st.subheader("🧾 Dataset (filtrato)")

    show_source = st.toggle("Mostra source", value=True)
    out = df_f.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cols = ["date", "weight", "bmi", "ma", "source"] if show_source else ["date", "weight", "bmi", "ma"]
    cols = [c for c in cols if c in out.columns]

    st.dataframe(out[cols], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Scarica CSV",
        data=out[cols].to_csv(index=False).encode("utf-8"),
        file_name="renpho_dashboard_export.csv",
        mime="text/csv",
        use_container_width=True
    )