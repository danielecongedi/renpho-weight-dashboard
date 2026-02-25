import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="RENPHO Weight Dashboard", layout="wide")

# Legge il link dal secrets di Streamlit Cloud
CSV_URL = st.secrets["CSV_URL"]

st.title("📉 RENPHO — Weight Monitor")
st.caption("Dati letti dal CSV pubblicato su Google Sheets.")

@st.cache_data(ttl=300)
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [c.strip().lower() for c in df.columns]

    col_date = next((c for c in ["date", "data", "day", "giorno"] if c in df.columns), None)
    col_weight = next((c for c in ["weight", "peso", "kg", "weight(kg)"] if c in df.columns), None)

    if col_date is None or col_weight is None:
        raise ValueError("Mancano colonne: usa 'date/data' e 'weight/peso'.")

    out = df[[col_date, col_weight]].copy()
    out.rename(columns={col_date: "date", col_weight: "weight"}, inplace=True)

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["weight"] = (
        out["weight"].astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("kg", "", regex=False)
        .str.strip()
    )
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")

    out = out.dropna(subset=["date", "weight"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out

try:
    data = load_data(CSV_URL)
except Exception as e:
    st.error(f"Errore caricamento dati: {e}")
    st.stop()

st.sidebar.header("⚙️ Impostazioni")
min_date = data["date"].min().date()
max_date = data["date"].max().date()

date_range = st.sidebar.date_input(
    "Intervallo date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = min_date, max_date

df = data[(data["date"].dt.date >= start) & (data["date"].dt.date <= end)].copy()

target_weight = st.sidebar.number_input(
    "🎯 Target peso (kg)",
    min_value=0.0,
    value=float(df["weight"].min()),
    step=0.5
)
ma_window = st.sidebar.selectbox("Media mobile", [7, 14, 21, 30], index=0)

st.sidebar.divider()
if st.sidebar.button("🔄 Forza refresh dati"):
    load_data.clear()
    st.rerun()

latest = df.iloc[-1]
latest_w = float(latest["weight"])
first_w = float(df.iloc[0]["weight"])
delta_total = latest_w - first_w

df_last7 = df[df["date"] >= (pd.Timestamp(latest["date"]) - pd.Timedelta(days=7))]
delta_7 = np.nan
if len(df_last7) >= 2:
    delta_7 = float(df_last7.iloc[-1]["weight"] - df_last7.iloc[0]["weight"])

to_target = latest_w - float(target_weight)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ultima misura", f"{latest_w:.1f} kg", f"{delta_7:+.1f} kg (7gg)" if not np.isnan(delta_7) else "—")
c2.metric("Periodo selezionato", f"{first_w:.1f} → {latest_w:.1f} kg", f"{delta_total:+.1f} kg")
c3.metric("Distanza dal target", f"{to_target:+.1f} kg", "↓ meglio" if to_target > 0 else "✅ target raggiunto/oltre")
c4.metric("Ultima data", str(latest["date"].date()), "")

st.divider()

df["ma"] = df["weight"].rolling(ma_window, min_periods=1).mean()

left, right = st.columns([2, 1])

with left:
    st.subheader("Trend peso")
    fig = plt.figure()
    plt.plot(df["date"], df["weight"], marker="o")
    plt.plot(df["date"], df["ma"])
    plt.axhline(float(target_weight), linestyle="--")
    plt.xlabel("Data")
    plt.ylabel("Peso (kg)")
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Distribuzione (periodo)")
    fig2 = plt.figure()
    plt.hist(df["weight"].dropna().values, bins=12)
    plt.xlabel("Peso (kg)")
    plt.ylabel("Frequenza")
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()
st.subheader("Dati")
st.dataframe(
    df[["date", "weight", "ma"]].rename(columns={"ma": f"MA{ma_window}"}),
    use_container_width=True,
    hide_index=True
)