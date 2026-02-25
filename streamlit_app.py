import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta, date

# -------------------------
# CONFIGURAZIONE PAGINA
# -------------------------
st.set_page_config(
    page_title="RENPHO Weight Monitor", 
    page_icon="📉", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# COSTANTI
# -------------------------
BASELINE_DATE = date(2024, 8, 1) # Assicurati che l'anno sia corretto
BASELINE_WEIGHT = 112.0
TARGET_WEIGHT = 72.0

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MANUAL_FILE = DATA_DIR / "manual_entries.csv"

# -------------------------
# FUNZIONI DI SUPPORTO (HELPERS)
# -------------------------
@st.cache_data(ttl=300, show_spinner="Scaricamento dati RENPHO in corso...")
def load_renpho_csv(url: str) -> pd.DataFrame:
    """Parser robusto per export RENPHO."""
    raw = pd.read_csv(url, header=None)

    if raw.shape[1] == 1:
        s = raw[0].astype(str).str.strip()
        s = s.str.replace(r'^"|"$', "", regex=True)
        raw = s.str.split(",", expand=True)

    if raw.shape[1] < 3:
        raise ValueError("CSV RENPHO non riconosciuto. Mancano i campi base.")

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
    """Carica i dati inseriti manualmente dal file locale."""
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
    """Salva i dati manuali su file CSV."""
    out = df.copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out.to_csv(MANUAL_FILE, index=False)

def combine_data(renpho_df: pd.DataFrame, manual_df: pd.DataFrame) -> pd.DataFrame:
    """Unisce i dati RENPHO e manuali, dando priorità a quelli manuali in caso di conflitto."""
    if manual_df.empty:
        df = renpho_df.copy()
    else:
        df = pd.concat([renpho_df, manual_df], ignore_index=True)
        df["__prio"] = df["source"].map({"renpho": 0, "manual": 1}).fillna(0)
        df = df.sort_values(["date", "__prio"]).drop(columns=["__prio"])
        df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.sort_values("date").reset_index(drop=True)
    return df

def fit_linear_model(df: pd.DataFrame, lookback_days: int) -> dict:
    """Modello lineare semplice per prevedere il calo di peso."""
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
    x = (sub["date"] - t0).dt.total_seconds().values / 86400.0  # giorni
    y = sub["weight"].values

    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return {"ok": True, "m": m, "b": b, "t0": t0, "sub": sub}

def predict_weight(model: dict, when: pd.Timestamp) -> float:
    """Calcola il peso previsto per una data specifica."""
    x = (when - model["t0"]).total_seconds() / 86400.0
    return float(model["m"] * x + model["b"])

def next_saturday(d: date) -> date:
    """Trova il prossimo sabato rispetto a una data."""
    days_ahead = (5 - d.weekday()) % 7
    return d + timedelta(days=days_ahead)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("⚙️ Impostazioni")
    st.markdown("Gestisci i parametri del modello e dell'obiettivo.")

    target_weight = st.number_input("🎯 Target (kg)", value=float(TARGET_WEIGHT), step=0.5)
    baseline_weight = st.number_input("⚑ Peso iniziale (kg)", value=float(BASELINE_WEIGHT), step=0.5)
    baseline_date = st.date_input("⚑ Data Inizio", value=BASELINE_DATE)

    st.divider()
    
    lookback_days = st.selectbox("🔍 Finestra modello predittivo", [30, 45, 60, 90, 120, 180], index=2, help="Quanti giorni passati usare per calcolare il trend.")
    ma_window = st.selectbox("📈 Media mobile (giorni)", [7, 14, 21, 30], index=0, help="Per smussare le fluttuazioni giornaliere.")

    st.divider()
    if st.button("🔄 Forza refresh dati RENPHO", use_container_width=True):
        load_renpho_csv.clear()
        st.rerun()

# -------------------------
# CARICAMENTO DATI MAIN
# -------------------------
st.title("📉 RENPHO — Weight Monitor")

csv_url = st.secrets.get("CSV_URL", "")
if not csv_url:
    st.error("⚠️ CSV_URL non impostato. Inseriscilo in `.streamlit/secrets.toml` o nelle impostazioni di Streamlit Cloud.")
    st.stop()

try:
    renpho = load_renpho_csv(csv_url)
except Exception as e:
    st.error(f"⚠️ Errore durante il caricamento da RENPHO: {e}")
    st.stop()

manual = load_manual()
df = combine_data(renpho, manual)

if df.empty:
    st.warning("📭 Nessun dato disponibile.")
    st.stop()

# Filtro date (posizionato sotto il titolo per maggiore visibilità o in sidebar)
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

if df_f.empty:
    st.warning("📭 L'intervallo selezionato non contiene dati.")
    st.stop()

# Calcoli metriche
df_f["ma"] = df_f["weight"].rolling(ma_window, min_periods=1).mean()
latest = df_f.iloc[-1]
latest_w = float(latest["weight"])
latest_date = latest["date"]

loss_from_baseline = latest_w - baseline_weight # Sarà negativo (es. -40 kg)
dist_to_target = latest_w - target_weight

# Fitting Modello
model = fit_linear_model(df_f, lookback_days=lookback_days)
target_date_est = None
target_date_str = "—"

if model.get("ok", False):
    m = model["m"]
    b = model["b"]
    if m < 0: # Sta dimagrendo
        x_target = (float(target_weight) - b) / m
        if np.isfinite(x_target) and x_target >= 0:
            target_date_est = (model["t0"] + pd.Timedelta(days=float(x_target))).date()
            target_date_str = target_date_est.strftime("%d %b %Y")

# -------------------------
# LAYOUT A TABS
# -------------------------
tab_dash, tab_manual, tab_forecast, tab_data = st.tabs([
    "📊 Cruscotto", 
    "✍️ Inserimento Manuale", 
    "🔮 Forecast & Sabati", 
    "🧾 Dataset"
])

# --- TAB 1: CRUSCOTTO ---
with tab_dash:
    # Metriche in alto
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ultima misura", f"{latest_w:.1f} kg", f"{latest_date.strftime('%d/%m %H:%M')}")
    
    # delta_color="inverse" colora di verde i numeri negativi (perdita di peso)
    c2.metric("Dal Inizio", f"{latest_w:.1f} kg", f"{loss_from_baseline:+.1f} kg", delta_color="inverse")
    c3.metric("Dist. Target", f"{target_weight:.1f} kg", f"{dist_to_target:+.1f} kg", delta_color="inverse")
    
    # Per la data non usiamo il delta, ma solo le info
    c4.metric("Data Stimata Target", target_date_str, f"Basato su ultimi {lookback_days} gg" if target_date_est else "Trend non calante")

    st.markdown("---")
    st.subheader("📈 Trend del Peso e Previsione")

    # GRAFICO INTERATTIVO CON PLOTLY
    fig = go.Figure()

    # Pesi Reali
    fig.add_trace(go.Scatter(
        x=df_f["date"], y=df_f["weight"], 
        mode='lines+markers', 
        name='Peso Registrato',
        marker=dict(color='#1f77b4', size=5),
        line=dict(width=1, color='rgba(31, 119, 180, 0.4)'),
        hovertemplate='Data: %{x}<br>Peso: %{y:.1f} kg<extra></extra>'
    ))

    # Media Mobile
    fig.add_trace(go.Scatter(
        x=df_f["date"], y=df_f["ma"], 
        mode='lines', 
        name=f'Media Mobile ({ma_window}gg)',
        line=dict(color='#ff7f0e', width=3),
        hovertemplate='Data: %{x}<br>Media: %{y:.1f} kg<extra></extra>'
    ))

    # Target
    fig.add_hline(
        y=float(target_weight), 
        line_dash="dash", 
        line_color="green", 
        annotation_text="🎯 Obiettivo",
        annotation_position="bottom right"
    )

    # Forecast e Fit
    if model.get("ok", False):
        sub = model["sub"]
        y_fit = [predict_weight(model, d) for d in sub["date"]]
        
        # Linea del modello sui dati storici
        fig.add_trace(go.Scatter(
            x=sub["date"], y=y_fit, 
            mode='lines', 
            name='Trend (ML)',
            line=dict(color='#d62728', width=2, dash='dot')
        ))

        # Previsione futura (fino al target o per 180 giorni)
        last_dt = df_f["date"].max()
        horizon_days = (target_date_est - last_dt.date()).days + 10 if target_date_est else 180
        horizon_days = max(30, min(horizon_days, 365)) # Limite tra 30 e 365 giorni per pulizia grafica
        
        future_dates = pd.date_range(last_dt, last_dt + pd.Timedelta(days=horizon_days), freq="D")
        y_fore = [predict_weight(model, d) for d in future_dates]
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=y_fore, 
            mode='lines', 
            name='Forecast',
            line=dict(color='#9467bd', width=2, dash='dash')
        ))

    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Peso (kg)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: INSERIMENTO MANUALE ---
with tab_manual:
    st.subheader("➕ Aggiungi misura manuale")
    st.info("💡 I dati manuali hanno la priorità su quelli RENPHO a parità di orario.")

    with st.form("manual_form", clear_on_submit=True):
        colA, colB, colC, colD = st.columns(4)
        m_date = colA.date_input("Data", value=date.today())
        m_time = colB.time_input("Ora", value=datetime.now().time().replace(second=0, microsecond=0))
        m_weight = colC.number_input("Peso (kg)", min_value=0.0, value=float(latest_w), step=0.1)
        m_bmi = colD.number_input("BMI (Opzionale)", min_value=0.0, value=0.0, step=0.1, help="Lascia 0.0 per ignorare.")

        submitted = st.form_submit_button("✅ Salva misura", use_container_width=True)

        if submitted:
            dt = pd.Timestamp(datetime.combine(m_date, m_time))
            bmi_val = float(m_bmi) if float(m_bmi) > 0 else np.nan

            new_row = pd.DataFrame([{
                "date": dt, "weight": float(m_weight), "bmi": bmi_val, "source": "manual"
            }])

            manual_now = load_manual()
            manual_now = pd.concat([manual_now, new_row], ignore_index=True)
            manual_now = manual_now.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            save_manual(manual_now)

            st.success("Misura salvata! Aggiornamento in corso...")
            load_renpho_csv.clear()
            st.rerun()

    st.divider()
    
    st.subheader("🗑️ Cancella misure manuali")
    manual_now = load_manual()
    
    if manual_now.empty:
        st.caption("Non ci sono misure manuali salvate al momento.")
    else:
        manual_now_disp = manual_now.copy()
        manual_now_disp["date_str"] = manual_now_disp["date"].dt.strftime("%Y-%m-%d %H:%M")
        
        to_delete = st.multiselect("Seleziona i record da rimuovere:", options=manual_now_disp["date_str"].tolist())

        col_del1, col_del2 = st.columns(2)
        if col_del1.button("🗑️ Cancella Selezionate", type="primary", use_container_width=True):
            if to_delete:
                keep_mask = ~manual_now_disp["date_str"].isin(to_delete)
                manual_new = manual_now_disp.loc[keep_mask, ["date", "weight", "bmi", "source"]].copy()
                save_manual(manual_new)
                st.success("Record cancellati. Aggiornamento in corso...")
                st.rerun()
            else:
                st.warning("Seleziona almeno un record per cancellarlo.")

        if col_del2.button("⚠️ Svuota Tutto", use_container_width=True):
            save_manual(pd.DataFrame(columns=["date", "weight", "bmi", "source"]))
            st.success("Archivio manuale azzerato. Aggiornamento in corso...")
            st.rerun()

# --- TAB 3: FORECAST & SABATI ---
with tab_forecast:
    st.subheader("🗓️ Tabella dei Sabati fino al Target")

    if not model.get("ok", False):
        st.warning("⚠️ Dati insufficienti per stimare un modello predittivo.")
    elif not target_date_est:
        st.warning("⚠️ Attualmente il trend non è in discesa. Impossibile calcolare una data di arrivo.")
    else:
        st.success(f"🎯 Con il ritmo attuale, raggiungerai **{target_weight:.1f} kg** stimato per il **{target_date_str}**.")

        start_ref = max(date.today(), df_f["date"].max().date())
        sat = next_saturday(start_ref)
        sats = []
        
        while sat <= target_date_est:
            sats.append(sat)
            sat += timedelta(days=7)

        if not sats:
            st.info("Nessun sabato intermedio trovato. Il target è imminente!")
        else:
            rows = []
            for s in sats:
                ts = pd.Timestamp(datetime.combine(s, datetime.min.time()))
                w_pred = predict_weight(model, ts)
                rows.append({
                    "Sabato": s.strftime("%d %b %Y"), 
                    "Peso Previsto (kg)": round(w_pred, 1), 
                    "Mancanti (kg)": round(w_pred - float(target_weight), 1)
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- TAB 4: DATASET ---
with tab_data:
    col_data1, col_data2 = st.columns([3, 1])
    
    with col_data1:
        st.subheader("🧾 Storico Completo")
    with col_data2:
        show_source = st.toggle("Mostra origine dati", value=True)

    out = df_f.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = ["date", "weight", "bmi", "ma", "source"] if show_source else ["date", "weight", "bmi", "ma"]
    cols = [c for c in cols if c in out.columns]
    
    st.dataframe(out[cols], use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Esporta Dataset (CSV)",
        data=out[cols].to_csv(index=False).encode("utf-8"),
        file_name=f"renpho_export_{date.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )