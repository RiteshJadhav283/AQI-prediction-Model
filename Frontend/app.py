"""
AQI Prediction Dashboard — Streamlit Application
Elegant, interactive dashboard for Air Quality Index analysis & prediction (India 2015-2020)
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark glassmorphic theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f2027 40%, #1a1a2e 100%);
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,255,150,0.15);
}
[data-testid="metric-container"] label {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f4ff !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

h1 { color: #00ff99 !important; font-weight: 700 !important; }
h2 { color: #80dfff !important; font-weight: 600 !important; }
h3 { color: #b0c4de !important; font-weight: 500 !important; }

.section-divider {
    height: 2px;
    background: linear-gradient(90deg, #00ff99, #00b4d8, transparent);
    border: none;
    margin: 2rem 0;
    border-radius: 2px;
}

.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(16px);
    margin-bottom: 1rem;
}

.aqi-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
}

div[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
}
div[data-baseweb="tab"] {
    border-radius: 8px !important;
    color: rgba(255,255,255,0.6) !important;
    font-weight: 500 !important;
}
div[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(0,255,153,0.15) !important;
    color: #00ff99 !important;
}

label[data-testid="stWidgetLabel"] {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.85rem !important;
}

.stButton button {
    background: linear-gradient(135deg, #00c776, #00b4d8);
    color: #000 !important;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.8rem;
    transition: all 0.25s;
    font-size: 0.95rem;
}
.stButton button:hover {
    opacity: 0.88;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,199,118,0.4);
}

section[data-testid="stSidebar"] .stButton button {
    padding: 0.35rem 0.4rem !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    min-width: 0 !important;
}
section[data-testid="stSidebar"] .stButton button p {
    font-size: 0.76rem !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,255,153,0.3); border-radius: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "Dataset" / "AQI" / "city_day.csv"

AQI_COLORS = {
    "Good": "#00e400",
    "Satisfactory": "#92d14f",
    "Moderate": "#ffff00",
    "Poor": "#ff7e00",
    "Very Poor": "#ff0000",
    "Severe": "#8f3f97",
    "Unknown": "#888888",
}

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["MonthName"] = df["Month"].map(MONTHS)
    df_clean = df.dropna(subset=["AQI"]).drop_duplicates()
    return df, df_clean


@st.cache_data(show_spinner=False)
def build_models(df_clean):
    df_model = df_clean.copy()
    le = LabelEncoder()
    df_model["City_enc"] = le.fit_transform(df_model["City"])
    df_model = df_model.drop(columns=["AQI_Bucket", "Date", "MonthName"], errors="ignore")
    df_model = df_model.fillna(df_model.median(numeric_only=True))

    feature_cols = ["City_enc", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
                    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "Year", "Month", "Day"]
    X = df_model[feature_cols]
    y = df_model["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        if name == "Linear Regression":
            model.fit(X_train_sc, y_train)
            preds = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        mse  = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, preds)
        results[name] = {
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4),
            "y_test": y_test.values,
            "y_pred": preds,
            "model": model,
        }

    rf_model = results["Random Forest"]["model"]
    feat_imp = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return results, feat_imp, le, scaler, feature_cols, X_train


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def aqi_category(val):
    if val <= 50:   return "Good"
    if val <= 100:  return "Satisfactory"
    if val <= 200:  return "Moderate"
    if val <= 300:  return "Poor"
    if val <= 400:  return "Very Poor"
    return "Severe"


def aqi_color(val):
    return AQI_COLORS.get(aqi_category(val), "#888")


def plotly_dark_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#c8d6e5", family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(t=50, b=40, l=50, r=30),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🌿  Loading data & training models…"):
    df_raw, df = load_data()
    model_results, feat_imp, le, scaler, feature_cols, X_train_ref = build_models(df)

cities = sorted(df["City"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 AQI Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "📊  EDA", "🤖  ML Models", "🔮  Predict AQI"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("### 🔧 Filters")
    
    # Quick selection controls for cities
    city_preset = st.radio(
        "Quick City Selection",
        ["All (26)", "Top 10", "Top 5", "Custom"],
        horizontal=True,
        index=2,
        label_visibility="collapsed",
    )

    if city_preset == "All (26)":
        pre_sel = list(cities)
    elif city_preset == "Top 10":
        pre_sel = list(cities[:10])
    elif city_preset == "Top 5":
        pre_sel = list(cities[:5])
    else:
        pre_sel = st.session_state.get("selected_cities", cities[:5])

    sel_cities = st.multiselect(
        "Cities",
        options=cities,
        default=pre_sel,
        key="selected_cities" if city_preset == "Custom" else None,
        help=f"Select individual cities or use the preset pills above (Total: {len(cities)} cities)"
    )

    year_range = st.slider("Year Range", 2015, 2020, (2015, 2020))

    st.markdown("---")
    st.caption("Data: CPCB / Kaggle — India 2015-2020  \n26 cities • 24,850 records")

df_f = df[df["City"].isin(sel_cities) & df["Year"].between(*year_range)] if sel_cities else df[df["Year"].between(*year_range)]



# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Overview":
    st.markdown("# 🌍  Air Quality Index — India")
    st.markdown("### Real-time analytics & machine learning insights for 26 Indian cities (2015–2020)")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋  Records", f"{len(df_f):,}")
    with col2:
        st.metric("🏙️  Cities", len(df_f["City"].unique()))
    with col3:
        mean_aqi = df_f["AQI"].mean()
        st.metric("📈  Avg AQI", f"{mean_aqi:.0f}")
    with col4:
        st.metric("⬆️  Max AQI", f"{df_f['AQI'].max():.0f}")
    with col5:
        st.metric("⬇️  Min AQI", f"{df_f['AQI'].min():.0f}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### 🗺️  Average AQI by City")
        city_avg = (
            df_f.groupby("City")["AQI"]
            .mean()
            .reset_index()
            .sort_values("AQI", ascending=True)
        )
        fig_city = px.bar(
            city_avg, x="AQI", y="City", orientation="h",
            color="AQI",
            color_continuous_scale=[[0,"#00e400"],[0.25,"#ffff00"],[0.5,"#ff7e00"],[0.75,"#ff0000"],[1,"#8f3f97"]],
            template="plotly_dark",
        )
        fig_city.update_layout(**plotly_dark_layout(), coloraxis_showscale=False, height=500)
        st.plotly_chart(fig_city, use_container_width=True)

    with right:
        st.markdown("#### 🥧  AQI Bucket Distribution")
        bucket_counts = df_f["AQI_Bucket"].value_counts().reset_index()
        bucket_counts.columns = ["Bucket", "Count"]
        fig_pie = px.pie(
            bucket_counts, names="Bucket", values="Count",
            color="Bucket", color_discrete_map=AQI_COLORS,
            hole=0.45, template="plotly_dark",
        )
        fig_pie.update_layout(**{**plotly_dark_layout(), "height": 280, "legend": dict(orientation="h", y=-0.1)})
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 📅  AQI Trend (Monthly Avg)")
        monthly = df_f.groupby(["Year", "Month"])["AQI"].mean().reset_index()
        monthly["Date"] = pd.to_datetime(monthly[["Year", "Month"]].assign(day=1))
        monthly = monthly.sort_values("Date")
        fig_trend = px.line(monthly, x="Date", y="AQI", template="plotly_dark",
                            color_discrete_sequence=["#00ff99"])
        fig_trend.update_layout(**plotly_dark_layout(), height=220, xaxis_title="", yaxis_title="AQI")
        fig_trend.update_traces(line_width=2)
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("#### 🌡️  City-Month AQI Heatmap")
    pivot = df_f.pivot_table(values="AQI", index="City", columns="Month", aggfunc="mean").round(0)
    pivot.columns = [MONTHS[c] for c in pivot.columns]
    fig_hm = px.imshow(
        pivot,
        color_continuous_scale=[[0,"#00e400"],[0.2,"#ffff00"],[0.4,"#ff7e00"],[0.7,"#ff0000"],[1,"#8f3f97"]],
        aspect="auto", template="plotly_dark",
    )
    fig_hm.update_layout(**plotly_dark_layout(), height=420)
    st.plotly_chart(fig_hm, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊  EDA":
    st.markdown("# 📊  Exploratory Data Analysis")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlations", "Pollutants", "Seasonal"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### AQI Distribution")
            fig_hist = px.histogram(df_f, x="AQI", nbins=60,
                                    color_discrete_sequence=["#00b4d8"],
                                    template="plotly_dark", marginal="box")
            mean_v = df_f["AQI"].mean()
            median_v = df_f["AQI"].median()
            fig_hist.add_vline(x=mean_v, line_dash="dash", line_color="#ff6b6b",
                               annotation_text=f"Mean: {mean_v:.0f}")
            fig_hist.add_vline(x=median_v, line_dash="dot", line_color="#ffd93d",
                               annotation_text=f"Median: {median_v:.0f}")
            fig_hist.update_layout(**plotly_dark_layout(), height=420)
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.markdown("#### Summary Statistics")
            stats = df_f["AQI"].describe().round(2)
            for k, v in stats.items():
                label = {"count":"Count","mean":"Mean","std":"Std Dev","min":"Min",
                         "25%":"Q1","50%":"Median","75%":"Q3","max":"Max"}.get(k, k)
                st.metric(label, f"{v:,.2f}")

        st.markdown("#### Boxplot — AQI by City")
        fig_box = px.box(df_f.sort_values("AQI", ascending=False),
                         x="City", y="AQI", color="City",
                         template="plotly_dark", notched=True)
        fig_box.update_layout(**plotly_dark_layout(), height=450,
                               showlegend=False, xaxis_tickangle=-40)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        st.markdown("#### 🔗  Pearson Correlation Matrix")
        num_cols = ["AQI"] + POLLUTANTS
        corr_mat = df_f[num_cols].dropna().corr().round(2)
        fig_corr = px.imshow(corr_mat, color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, text_auto=True, template="plotly_dark")
        fig_corr.update_layout(**plotly_dark_layout(), height=550)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### 🔎  Scatter: Pollutant vs AQI")
        sel_poll = st.selectbox("Choose pollutant", POLLUTANTS)
        sample_df = df_f.dropna(subset=[sel_poll, "AQI"]).sample(min(3000, len(df_f)))
        fig_scatter = px.scatter(sample_df, x=sel_poll, y="AQI", color="City",
                                 opacity=0.55, trendline="ols", template="plotly_dark")
        fig_scatter.update_layout(**plotly_dark_layout(), height=430)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.markdown("#### 📦  Pollutant Levels by City (Median)")
        poll_city = df_f.groupby("City")[POLLUTANTS].median().round(2)
        sel_polls = st.multiselect("Pollutants", POLLUTANTS, default=["PM2.5", "PM10", "NO2"])
        if sel_polls:
            poll_melt = poll_city[sel_polls].reset_index().melt(id_vars="City", var_name="Pollutant", value_name="Median")
            fig_poll = px.bar(poll_melt, x="City", y="Median", color="Pollutant",
                              barmode="group", template="plotly_dark",
                              color_discrete_sequence=px.colors.qualitative.Bold)
            fig_poll.update_layout(**plotly_dark_layout(), height=450, xaxis_tickangle=-40)
            st.plotly_chart(fig_poll, use_container_width=True)

        st.markdown("#### 📈  Pollutant Trend Over Time")
        sel_poll2 = st.selectbox("Pollutant", POLLUTANTS, key="poll_trend")
        poll_time = df_f.groupby(["Year","Month"])[sel_poll2].mean().reset_index()
        poll_time["Date"] = pd.to_datetime(poll_time[["Year","Month"]].assign(day=1))
        poll_time = poll_time.sort_values("Date")
        fig_pt = px.area(poll_time, x="Date", y=sel_poll2, template="plotly_dark",
                         color_discrete_sequence=["#00ff99"])
        fig_pt.update_layout(**plotly_dark_layout(), height=320)
        st.plotly_chart(fig_pt, use_container_width=True)

    with tab4:
        st.markdown("#### 🍂  Seasonal AQI Pattern (Monthly Average)")
        seasonal = df_f.groupby("Month")["AQI"].mean().reset_index()
        seasonal["MonthName"] = seasonal["Month"].map(MONTHS)
        fig_sea = px.line(seasonal, x="MonthName", y="AQI", markers=True,
                          template="plotly_dark", color_discrete_sequence=["#00b4d8"])
        fig_sea.update_traces(line_width=3, marker_size=8)
        fig_sea.update_layout(**plotly_dark_layout(), height=380,
                               xaxis=dict(categoryorder="array", categoryarray=list(MONTHS.values())))
        st.plotly_chart(fig_sea, use_container_width=True)

        st.markdown("#### 📆  Year-over-Year AQI")
        yoy = df_f.groupby(["Year","Month"])["AQI"].mean().reset_index()
        yoy["MonthName"] = yoy["Month"].map(MONTHS)
        fig_yoy = px.line(yoy, x="MonthName", y="AQI", color="Year",
                          markers=True, template="plotly_dark",
                          color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_yoy.update_layout(**plotly_dark_layout(), height=400,
                               xaxis=dict(categoryorder="array", categoryarray=list(MONTHS.values())))
        st.plotly_chart(fig_yoy, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖  ML Models":
    st.markdown("# 🤖  Machine Learning — Model Performance")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    metrics_df = pd.DataFrame(
        {k: {m: v for m, v in v.items() if m not in ["y_test", "y_pred", "model"]}
         for k, v in model_results.items()}
    ).T.reset_index().rename(columns={"index": "Model"})

    best_model_name = metrics_df.loc[metrics_df["R2 Score"].idxmax(), "Model"]

    col_kpi = st.columns(4)
    icons = {"Linear Regression": "📐", "Decision Tree": "🌳", "Random Forest": "🌲", "Gradient Boosting": "🚀"}
    for i, (_, row) in enumerate(metrics_df.iterrows()):
        with col_kpi[i]:
            badge = " ⭐" if row["Model"] == best_model_name else ""
            st.metric(f"{icons.get(row['Model'], '🤖')} {row['Model']}{badge}",
                      f"R² = {row['R2 Score']:.4f}",
                      delta=f"RMSE = {row['RMSE']:.2f}", delta_color="inverse")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab_m1, tab_m2, tab_m3, tab_m4 = st.tabs(
        ["📊  Metrics", "📉  Actual vs Predicted", "🌟  Feature Importance", "🔍  Error Analysis"])

    with tab_m1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### RMSE (↓ lower is better)")
            fig_rmse = px.bar(metrics_df, x="Model", y="RMSE", color="Model",
                              template="plotly_dark",
                              color_discrete_sequence=px.colors.qualitative.Bold)
            for _, row in metrics_df.iterrows():
                fig_rmse.add_annotation(x=row["Model"], y=row["RMSE"]+0.5,
                                         text=str(row["RMSE"]), showarrow=False,
                                         font=dict(color="white", size=12))
            fig_rmse.update_layout(**plotly_dark_layout(), height=380, showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
        with c2:
            st.markdown("#### R² Score (↑ higher is better)")
            fig_r2 = px.bar(metrics_df, x="Model", y="R2 Score", color="Model",
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Bold)
            fig_r2.update_layout(**plotly_dark_layout(), height=380, showlegend=False,
                                  yaxis=dict(range=[0.75, 1.0]))
            for _, row in metrics_df.iterrows():
                fig_r2.add_annotation(x=row["Model"], y=row["R2 Score"]+0.002,
                                       text=str(row["R2 Score"]), showarrow=False,
                                       font=dict(color="white", size=12))
            st.plotly_chart(fig_r2, use_container_width=True)

        st.markdown("#### 📋  Full Metrics")
        st.dataframe(metrics_df.set_index("Model"), use_container_width=True)

    with tab_m2:
        sel_model = st.selectbox("Select model", list(model_results.keys()))
        y_test = model_results[sel_model]["y_test"]
        y_pred = model_results[sel_model]["y_pred"]
        idx = np.random.choice(len(y_test), min(1500, len(y_test)), replace=False)
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(x=y_test[idx], y=y_pred[idx], mode="markers",
                                     marker=dict(color="#00b4d8", opacity=0.45, size=5), name="Predictions"))
        lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        fig_avp.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                     line=dict(color="#ff6b6b", dash="dash", width=2), name="Perfect Fit"))
        fig_avp.update_layout(**plotly_dark_layout(), height=480,
                               xaxis_title="Actual AQI", yaxis_title="Predicted AQI",
                               title=f"{sel_model} — Actual vs Predicted")
        st.plotly_chart(fig_avp, use_container_width=True)

    with tab_m3:
        st.markdown("#### 🌟  Random Forest Feature Importances")
        fi_df = feat_imp.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale=["#00b4d8", "#00ff99"],
                        template="plotly_dark")
        fig_fi.update_layout(**plotly_dark_layout(), height=460, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab_m4:
        sel_model2 = st.selectbox("Select model", list(model_results.keys()), key="ea")
        y_test2 = model_results[sel_model2]["y_test"]
        y_pred2 = model_results[sel_model2]["y_pred"]
        residuals = y_test2 - y_pred2
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Residuals Distribution")
            fig_res = px.histogram(x=residuals, nbins=60,
                                   color_discrete_sequence=["#f7b731"],
                                   template="plotly_dark",
                                   labels={"x": "Residual (Actual - Predicted)"})
            fig_res.add_vline(x=0, line_dash="dash", line_color="#ff6b6b")
            fig_res.update_layout(**plotly_dark_layout(), height=350)
            st.plotly_chart(fig_res, use_container_width=True)
        with c2:
            st.markdown("#### Residuals vs Predicted")
            fig_rv = px.scatter(x=y_pred2, y=residuals, opacity=0.45,
                                color_discrete_sequence=["#a29bfe"],
                                template="plotly_dark",
                                labels={"x": "Predicted AQI", "y": "Residual"})
            fig_rv.add_hline(y=0, line_dash="dash", line_color="#ff6b6b")
            fig_rv.update_layout(**plotly_dark_layout(), height=350)
            st.plotly_chart(fig_rv, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — PREDICT AQI
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮  Predict AQI":
    st.markdown("# 🔮  Predict Your AQI")
    st.markdown("Enter pollutant values to get a real-time AQI prediction.")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    use_model = st.selectbox("Select model", list(model_results.keys()), index=2)

    c_form, c_result = st.columns([2, 1])

    with c_form:
        st.markdown("### 🧪  Pollutant Inputs")
        with st.form("prediction_form"):
            city_sel = st.selectbox("City", cities)
            year_sel = st.selectbox("Year", list(range(2015, 2026)), index=5)
            month_sel = st.selectbox("Month", list(MONTHS.values()))
            day_sel = st.slider("Day", 1, 31, 15)
            st.markdown("---")
            g1, g2, g3 = st.columns(3)
            with g1:
                pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 65.0, step=1.0)
                no   = st.number_input("NO (µg/m³)", 0.0, 500.0, 12.0, step=1.0)
                nh3  = st.number_input("NH3 (µg/m³)", 0.0, 200.0, 16.0, step=1.0)
                benz = st.number_input("Benzene (µg/m³)", 0.0, 100.0, 1.5, step=0.1)
            with g2:
                pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1500.0, 100.0, step=1.0)
                no2  = st.number_input("NO2 (µg/m³)", 0.0, 500.0, 30.0, step=1.0)
                co   = st.number_input("CO (mg/m³)", 0.0, 100.0, 10.0, step=0.5)
                tol  = st.number_input("Toluene (µg/m³)", 0.0, 500.0, 10.0, step=0.5)
            with g3:
                nox  = st.number_input("NOx (µg/m³)", 0.0, 500.0, 40.0, step=1.0)
                so2  = st.number_input("SO2 (µg/m³)", 0.0, 500.0, 20.0, step=1.0)
                o3   = st.number_input("O3 (µg/m³)", 0.0, 500.0, 50.0, step=1.0)
                xyl  = st.number_input("Xylene (µg/m³)", 0.0, 200.0, 2.0, step=0.1)
            submitted = st.form_submit_button("🚀  Predict AQI", use_container_width=True)

    with c_result:
        if submitted:
            month_num = {v: k for k, v in MONTHS.items()}[month_sel]
            try:
                city_enc = le.transform([city_sel])[0]
            except Exception:
                city_enc = 0

            input_data = pd.DataFrame([[
                city_enc, pm25, pm10, no, no2, nox, nh3, co, so2, o3, benz, tol, xyl,
                year_sel, month_num, day_sel
            ]], columns=feature_cols)

            mdl = model_results[use_model]["model"]
            if use_model == "Linear Regression":
                pred = mdl.predict(scaler.transform(input_data))[0]
            else:
                pred = mdl.predict(input_data)[0]
            pred = max(0, round(pred, 1))

            cat = aqi_category(pred)
            color = aqi_color(pred)

            st.markdown("### 🎯  Prediction Result")
            st.markdown(
                f"""<div class="glass-card" style="text-align:center; border-color:{color}55; box-shadow: 0 0 30px {color}33;">
                    <p style="font-size:1rem; color:rgba(255,255,255,0.6); margin:0;">Predicted AQI</p>
                    <p style="font-size:4rem; font-weight:800; color:{color}; margin:0.3rem 0;">{pred}</p>
                    <span class="aqi-badge" style="background:{color}22; color:{color}; border: 1.5px solid {color};">{cat}</span>
                </div>""",
                unsafe_allow_html=True,
            )

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                gauge={
                    "axis": {"range": [0, 500], "tickcolor": "#c8d6e5"},
                    "bar": {"color": color, "thickness": 0.25},
                    "steps": [
                        {"range": [0, 50],   "color": "#00e40033"},
                        {"range": [50, 100], "color": "#92d14f33"},
                        {"range": [100, 200],"color": "#ffff0033"},
                        {"range": [200, 300],"color": "#ff7e0033"},
                        {"range": [300, 400],"color": "#ff000033"},
                        {"range": [400, 500],"color": "#8f3f9733"},
                    ],
                },
                title={"text": "AQI Gauge", "font": {"color": "#c8d6e5"}},
                number={"font": {"color": color}},
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d6e5",
                                    height=280, margin=dict(t=60, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            advice = {
                "Good": ("😊", "#00e400", "Air quality is good. Enjoy outdoor activities!"),
                "Satisfactory": ("🙂", "#92d14f", "Air is acceptable. Sensitive individuals should limit prolonged outdoor exertion."),
                "Moderate": ("😐", "#ffff00", "Breathing discomfort for sensitive groups."),
                "Poor": ("😷", "#ff7e00", "Health effects possible. Limit outdoor activities."),
                "Very Poor": ("🤧", "#ff0000", "Health alert! Avoid outdoor activities."),
                "Severe": ("☠️", "#8f3f97", "Emergency conditions. Stay indoors!"),
            }
            em, col, msg = advice.get(cat, ("❓", "#888", ""))
            st.markdown(
                f"""<div class="glass-card" style="border-color:{col}44; margin-top:1rem;">
                    <p style="font-size:1.5rem; margin:0;">{em} <span style="color:{col}; font-weight:600;">{cat}</span></p>
                    <p style="color:rgba(255,255,255,0.7); font-size:0.88rem; margin-top:0.4rem;">{msg}</p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """<div class="glass-card" style="text-align:center; margin-top:3rem; opacity:0.6;">
                    <p style="font-size:3rem;">🔮</p>
                    <p style="color:rgba(255,255,255,0.5);">Fill in the form and click<br><b>Predict AQI</b></p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📏  AQI Scale Reference")
    scale_df = pd.DataFrame({
        "Category": ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"],
        "AQI Range": ["0–50","51–100","101–200","201–300","301–400","401–500"],
        "Health Impact": [
            "Minimal impact",
            "Minor breathing discomfort to sensitive people",
            "Breathing discomfort to people with lung/heart disease",
            "Breathing discomfort to most on prolonged exposure",
            "Respiratory illness on prolonged exposure",
            "Health impacts even for healthy persons",
        ],
    })
    st.dataframe(scale_df, use_container_width=True, hide_index=True)
