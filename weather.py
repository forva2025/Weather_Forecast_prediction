"""Nimbus: real historical weather intelligence and next-day forecasting."""
from __future__ import annotations

from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Nimbus | Weather Intelligence", page_icon="⛅", layout="wide", initial_sidebar_state="expanded")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FEATURES = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_10m_max", "temp_lag_1", "temp_lag_2", "temp_rolling_7", "day_sin", "day_cos"]


def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
    :root{--navy:#081a2f;--blue:#2274f0;--mint:#43d3a4;--ink:#10243e;--muted:#63758b}
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif} h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.03em}
    [data-testid="stAppViewContainer"]{background:linear-gradient(145deg,#f8fbff 0%,#eef6ff 48%,#f8fbff 100%)}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#071a30 0%,#102c4d 100%)} [data-testid="stSidebar"] *{color:#eaf4ff}
    [data-testid="stSidebar"] input{color:#10243e!important} [data-testid="stSidebar"] .stButton button{background:linear-gradient(90deg,#35c8ed,#4ed6aa);color:#082039;border:0;font-weight:800}
    .block-container{padding-top:1.5rem;padding-bottom:3rem;max-width:1440px}
    .hero{position:relative;overflow:hidden;padding:2.2rem 2.4rem;border-radius:26px;color:white;background:radial-gradient(circle at 82% 12%,rgba(73,210,235,.35),transparent 26%),linear-gradient(120deg,#071b33,#124b82);box-shadow:0 18px 50px rgba(18,65,110,.18);margin-bottom:1.2rem}
    .hero:after{content:'☀';position:absolute;right:4%;top:-34px;font-size:130px;color:rgba(255,255,255,.08)}
    .eyebrow{color:#63e5d0;font-weight:800;text-transform:uppercase;letter-spacing:.14em;font-size:.76rem}.hero h1{font-size:clamp(2.2rem,4vw,4rem);margin:.35rem 0 .55rem;line-height:1.03}.hero p{color:#d5e9fb;max-width:720px;font-size:1.06rem;margin:0}
    .status-pill{display:inline-block;padding:.36rem .7rem;border:1px solid rgba(116,236,209,.35);background:rgba(67,211,164,.12);border-radius:99px;color:#a8f2df;font-size:.78rem;margin-top:1rem}
    [data-testid="stMetric"]{background:rgba(255,255,255,.86);border:1px solid #dce9f7;padding:1rem 1.1rem;border-radius:18px;box-shadow:0 8px 25px rgba(42,77,110,.06)}
    [data-testid="stMetricLabel"]{color:#60738a}[data-testid="stMetricValue"]{color:#0d2949;font-family:'Manrope',sans-serif}div[data-testid="stTabs"] button{font-weight:700}div[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:#2274f0}
    .info-card{padding:1.1rem 1.2rem;border-radius:16px;background:#fff;border:1px solid #dce9f7;min-height:142px}.info-card .icon{font-size:1.5rem}.info-card h4{color:#173455;margin:.35rem 0}.info-card p{color:#63758b;margin:0;font-size:.9rem}
    .section-kicker{color:#2274f0;font-weight:800;font-size:.74rem;letter-spacing:.12em;text-transform:uppercase}.footer{text-align:center;color:#74869b;padding:2rem 0 .25rem;font-size:.84rem}.stAlert{border-radius:14px}
    </style>""", unsafe_allow_html=True)


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_city(city: str) -> dict:
    response = requests.get(GEOCODING_URL, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=20)
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ValueError(f'No location found for "{city}". Try a nearby major city.')
    p = results[0]
    return {"name": p["name"], "country": p.get("country", ""), "admin1": p.get("admin1", ""), "latitude": p["latitude"], "longitude": p["longitude"]}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_history(latitude: float, longitude: float, start: date, end: date) -> pd.DataFrame:
    fields = ["weather_code", "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
    response = requests.get(ARCHIVE_URL, params={"latitude": latitude, "longitude": longitude, "start_date": start.isoformat(), "end_date": end.isoformat(), "daily": ",".join(fields), "timezone": "auto", "models": "era5_seamless"}, timeout=45)
    response.raise_for_status()
    payload = response.json()
    if "daily" not in payload:
        raise ValueError(payload.get("reason", "No observations were returned."))
    frame = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame[frame.columns.difference(["date"])] = frame[frame.columns.difference(["date"])].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["temperature_2m_mean"]).reset_index(drop=True)


def weather_label(code):
    if pd.isna(code): return "Unknown"
    code = int(code)
    if code == 0: return "Clear"
    if code in (1, 2, 3): return "Cloudy"
    if code in (45, 48): return "Fog"
    if code in (51, 53, 55, 56, 57): return "Drizzle"
    if code in (61, 63, 65, 66, 67, 80, 81, 82): return "Rain"
    if code in (71, 73, 75, 77, 85, 86): return "Snow"
    if code in (95, 96, 99): return "Thunderstorm"
    return "Mixed"


def build_features(raw):
    frame = raw.sort_values("date").copy()
    frame["temp_lag_1"] = frame["temperature_2m_mean"].shift(1)
    frame["temp_lag_2"] = frame["temperature_2m_mean"].shift(2)
    frame["temp_rolling_7"] = frame["temperature_2m_mean"].shift(1).rolling(7).mean()
    day = frame["date"].dt.dayofyear
    frame["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    frame["day_cos"] = np.cos(2 * np.pi * day / 365.25)
    frame["target_next_day"] = frame["temperature_2m_mean"].shift(-1)
    return frame.dropna(subset=FEATURES + ["target_next_day"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def train_model(frame):
    split = int(len(frame) * .8)
    train, test = frame.iloc[:split], frame.iloc[split:].copy()
    model = RandomForestRegressor(n_estimators=350, min_samples_leaf=3, max_features=.8, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["target_next_day"])
    test["prediction"] = model.predict(test[FEATURES])
    test["baseline"] = test["temperature_2m_mean"]
    metrics = {"mae": mean_absolute_error(test["target_next_day"], test["prediction"]), "rmse": mean_squared_error(test["target_next_day"], test["prediction"]) ** .5, "r2": r2_score(test["target_next_day"], test["prediction"]), "baseline_mae": mean_absolute_error(test["target_next_day"], test["baseline"]), "train_rows": len(train), "test_rows": len(test)}
    return model, test, metrics


def style_chart(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.7)", font=dict(family="DM Sans", color="#36516f"), title_font=dict(family="Manrope", color="#102f50"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hoverlabel=dict(bgcolor="#0b2745", font_color="white"))
    fig.update_xaxes(showgrid=False, linecolor="#d8e4ef")
    fig.update_yaxes(gridcolor="#e5edf5", zeroline=False)
    return fig


inject_styles()
with st.sidebar:
    st.markdown("## ⛅ Nimbus")
    st.caption("HISTORICAL WEATHER INTELLIGENCE")
    st.markdown("---")
    with st.form("location_form"):
        city = st.text_input("Location", value="New York", help="Enter a city or town.")
        start_date = st.date_input("History starts", value=date(2021, 1, 1), max_value=date.today() - timedelta(days=7))
        end_date = st.date_input("History ends", value=date(2025, 12, 31), max_value=date.today() - timedelta(days=5))
        submitted = st.form_submit_button("Load weather history", use_container_width=True)
    st.markdown("---")
    st.markdown("**Data source**")
    st.caption("Open-Meteo Historical Weather API · ERA5-Seamless · daily resolution")
    st.markdown("**Model**")
    st.caption("Random Forest · chronological 80/20 validation · seasonal + lag features")

if "query" not in st.session_state:
    st.session_state.query = {"city": "New York", "start": date(2021, 1, 1), "end": date(2025, 12, 31)}
if submitted:
    if start_date >= end_date: st.sidebar.error("The start date must be earlier than the end date.")
    elif (end_date - start_date).days < 365: st.sidebar.error("Choose at least one year so the model can learn seasonality.")
    else: st.session_state.query = {"city": city.strip(), "start": start_date, "end": end_date}
query = st.session_state.query

st.markdown("""<section class="hero"><div class="eyebrow">Weather intelligence, made clear</div><h1>Learn from the atmosphere.<br>Forecast what comes next.</h1><p>Explore real historical conditions, discover seasonal patterns, and estimate tomorrow's mean temperature with an explainable machine-learning workflow.</p><span class="status-pill">● LIVE DATA PIPELINE &nbsp;·&nbsp; ERA5-SEAMLESS</span></section>""", unsafe_allow_html=True)
try:
    with st.spinner("Collecting observations and training the forecast model…"):
        place = geocode_city(query["city"])
        raw = fetch_history(place["latitude"], place["longitude"], query["start"], query["end"])
        model_frame = build_features(raw)
        if len(model_frame) < 300: raise ValueError("Fewer than 300 complete observations were returned. Choose a longer range.")
        model, test, metrics = train_model(model_frame)
except (requests.RequestException, ValueError) as exc:
    st.error(f"Unable to prepare the dashboard: {exc}")
    st.info("Check your internet connection, location spelling, and selected date range, then try again.")
    st.stop()

latest = raw.iloc[-1]
label = ", ".join(dict.fromkeys(x for x in [place["name"], place["admin1"], place["country"]] if x))
st.markdown('<div class="section-kicker">Selected station area</div>', unsafe_allow_html=True)
st.markdown(f"## {label}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Latest mean", f"{latest['temperature_2m_mean']:.1f} °C", latest["date"].strftime("%d %b %Y"))
m2.metric("Period average", f"{raw['temperature_2m_mean'].mean():.1f} °C")
m3.metric("Total precipitation", f"{raw['precipitation_sum'].sum():,.0f} mm")
m4.metric("Model MAE", f"{metrics['mae']:.2f} °C", f"{metrics['baseline_mae']-metrics['mae']:+.2f} °C vs baseline")
m5.metric("Validation R²", f"{metrics['r2']:.3f}", f"{metrics['test_rows']:,} test days")

overview, forecast, patterns, data_tab, methods = st.tabs(["Overview", "Forecast lab", "Climate patterns", "Data explorer", "Methodology"])
with overview:
    st.markdown("### Conditions over time")
    period = st.segmented_control("Visible period", ["6 months", "1 year", "All history"], default="1 year")
    days = {"6 months": 183, "1 year": 365}.get(period)
    visible = raw.tail(days) if days else raw
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=visible.date, y=visible.temperature_2m_max, name="Daily high", line=dict(color="#f6a43a", width=1.2)))
    fig.add_trace(go.Scatter(x=visible.date, y=visible.temperature_2m_min, name="Daily low", line=dict(color="#4a91f2", width=1.2), fill="tonexty", fillcolor="rgba(49,133,235,.10)"))
    fig.add_trace(go.Scatter(x=visible.date, y=visible.temperature_2m_mean.rolling(14).mean(), name="14-day mean", line=dict(color="#102f50", width=3)))
    fig.update_layout(title="Temperature range and smoothed trend", yaxis_title="Temperature (°C)")
    st.plotly_chart(style_chart(fig), use_container_width=True)
    left, right = st.columns([1.65, 1])
    with left:
        monthly = raw.set_index("date").resample("MS").agg({"precipitation_sum":"sum", "relative_humidity_2m_mean":"mean"}).reset_index()
        combo = make_subplots(specs=[[{"secondary_y":True}]])
        combo.add_bar(x=monthly.date, y=monthly.precipitation_sum, name="Precipitation", marker_color="#39bcd8", secondary_y=False)
        combo.add_scatter(x=monthly.date, y=monthly.relative_humidity_2m_mean, name="Humidity", line=dict(color="#6f58d9", width=2), secondary_y=True)
        combo.update_yaxes(title_text="Precipitation (mm)", secondary_y=False); combo.update_yaxes(title_text="Humidity (%)", range=[0,100], secondary_y=True); combo.update_layout(title="Monthly moisture profile")
        st.plotly_chart(style_chart(combo, 385), use_container_width=True)
    with right:
        counts = raw.weather_code.map(weather_label).value_counts().rename_axis("Condition").reset_index(name="Days")
        donut = px.pie(counts, values="Days", names="Condition", hole=.64, color_discrete_sequence=["#2274f0", "#43d3a4", "#ffc45c", "#8e75e8", "#54c8e8", "#ef7189"]); donut.update_layout(title="Observed conditions")
        st.plotly_chart(style_chart(donut, 385), use_container_width=True)

with forecast:
    st.markdown("### Tomorrow temperature estimator")
    st.caption("Adjust the latest known conditions. Nimbus combines them with lagged temperatures and annual seasonality.")
    c1,c2,c3,c4 = st.columns(4)
    in_temp = c1.number_input("Today's mean (°C)", -50., 55., float(round(latest.temperature_2m_mean,1)), .5)
    in_humidity = c2.slider("Mean humidity (%)", 0, 100, int(round(latest.relative_humidity_2m_mean)))
    in_wind = c3.number_input("Max wind (km/h)", 0., 200., float(round(latest.wind_speed_10m_max,1)), .5)
    in_rain = c4.number_input("Precipitation (mm)", 0., 500., float(round(latest.precipitation_sum,1)), .5)
    next_date = latest.date + pd.Timedelta(days=1); doy = next_date.dayofyear
    row = pd.DataFrame([{"temperature_2m_mean":in_temp, "temperature_2m_min":float(latest.temperature_2m_min), "temperature_2m_max":float(latest.temperature_2m_max), "relative_humidity_2m_mean":in_humidity, "precipitation_sum":in_rain, "wind_speed_10m_max":in_wind, "temp_lag_1":float(raw.iloc[-2].temperature_2m_mean), "temp_lag_2":float(raw.iloc[-3].temperature_2m_mean), "temp_rolling_7":float(raw.iloc[-8:-1].temperature_2m_mean.mean()), "day_sin":np.sin(2*np.pi*doy/365.25), "day_cos":np.cos(2*np.pi*doy/365.25)}])
    predicted = float(model.predict(row[FEATURES])[0])
    st.markdown(f"""<div style="margin:1.2rem 0;padding:1.5rem 1.7rem;border-radius:20px;background:linear-gradient(110deg,#102f50,#196ea4);color:white"><div style="color:#8ee6d0;font-weight:800;letter-spacing:.1em;font-size:.72rem">MODEL ESTIMATE · {next_date:%d %B %Y}</div><div style="font-family:Manrope;font-size:3rem;font-weight:800;line-height:1.2">{predicted:.1f} °C</div><div style="color:#d1e6f5">Expected daily mean · typical validation error ±{metrics['mae']:.1f} °C</div></div>""", unsafe_allow_html=True)
    tail = test.tail(180); comparison = go.Figure()
    comparison.add_trace(go.Scatter(x=tail.date+pd.Timedelta(days=1), y=tail.target_next_day, name="Observed", line=dict(color="#102f50", width=2.5)))
    comparison.add_trace(go.Scatter(x=tail.date+pd.Timedelta(days=1), y=tail.prediction, name="Predicted", line=dict(color="#2274f0", width=2, dash="dot")))
    comparison.update_layout(title="Out-of-sample performance · latest 180 test days", yaxis_title="Mean temperature (°C)")
    st.plotly_chart(style_chart(comparison), use_container_width=True)
    a,b,c,d = st.columns(4); a.metric("Mean absolute error",f"{metrics['mae']:.2f} °C"); b.metric("Root mean squared error",f"{metrics['rmse']:.2f} °C"); c.metric("Persistence baseline",f"{metrics['baseline_mae']:.2f} °C"); d.metric("Relative improvement",f"{100*(metrics['baseline_mae']-metrics['mae'])/metrics['baseline_mae']:.1f}%")

with patterns:
    st.markdown("### Seasonal fingerprint")
    seasonal = raw.assign(month=raw.date.dt.month_name().str[:3], year=raw.date.dt.year); order=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    heat = seasonal.groupby(["year","month"])["temperature_2m_mean"].mean().reset_index().pivot(index="year",columns="month",values="temperature_2m_mean").reindex(columns=order)
    heatmap = px.imshow(heat, aspect="auto", color_continuous_scale=[[0,"#2376c9"],[.5,"#f7f5d7"],[1,"#f05e3b"]], labels=dict(color="°C")); heatmap.update_layout(title="Monthly mean temperature by year",xaxis_title="",yaxis_title="Year")
    st.plotly_chart(style_chart(heatmap), use_container_width=True)
    names={"temperature_2m_mean":"Today's mean","temperature_2m_min":"Today's low","temperature_2m_max":"Today's high","relative_humidity_2m_mean":"Humidity","precipitation_sum":"Precipitation","wind_speed_10m_max":"Wind","temp_lag_1":"1-day lag","temp_lag_2":"2-day lag","temp_rolling_7":"7-day mean","day_sin":"Season (sin)","day_cos":"Season (cos)"}
    importance=pd.DataFrame({"Feature":[names[x] for x in FEATURES],"Importance":model.feature_importances_}).sort_values("Importance")
    bar=px.bar(importance,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=["#bceaf2","#2274f0"]); bar.update_layout(title="What drives the model?",coloraxis_showscale=False)
    st.plotly_chart(style_chart(bar),use_container_width=True)

with data_tab:
    st.markdown("### Inspect and export observations")
    st.caption(f"{len(raw):,} real daily observations · {raw.date.min():%d %b %Y} to {raw.date.max():%d %b %Y}")
    display=raw.copy(); display["condition"]=display.weather_code.map(weather_label); display.columns=[x.replace("_"," ").title() for x in display.columns]
    st.dataframe(display.sort_values("Date",ascending=False),use_container_width=True,hide_index=True,height=430)
    st.download_button("Download historical data (CSV)",raw.to_csv(index=False).encode(),f"{place['name'].lower().replace(' ','_')}_weather.csv","text/csv")

with methods:
    st.markdown("### A forecasting workflow you can audit")
    cards=[("🌍","Real observations","Daily ERA5-Seamless historical fields come directly from Open-Meteo—no hand-written sample rows."),("🧭","Time-aware validation","The oldest 80% trains the model; the newest 20% tests it. Future records never leak into training."),("🌲","Nonlinear model","A 350-tree random forest learns weather interactions, lag effects, and the annual seasonal cycle."),("📏","Honest benchmarks","MAE, RMSE, R², and a persistence baseline put the model's errors in context.")]
    for col,(icon,title,body) in zip(st.columns(4),cards): col.markdown(f'<div class="info-card"><div class="icon">{icon}</div><h4>{title}</h4><p>{body}</p></div>',unsafe_allow_html=True)
    st.markdown("#### Pipeline"); st.code("City → Geocoding → Historical API → Feature engineering → Chronological split → Random Forest → Evaluation",language="text")
    st.caption(f"Training samples: {metrics['train_rows']:,} · Test samples: {metrics['test_rows']:,} · Coordinates: {place['latitude']:.3f}, {place['longitude']:.3f}")

st.markdown('<div class="footer">Built with Streamlit · Open-Meteo · scikit-learn · Plotly &nbsp; | &nbsp; Data remains attributable to Open-Meteo and its source models.</div>',unsafe_allow_html=True)
