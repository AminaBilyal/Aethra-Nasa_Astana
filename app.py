# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Aethra — Urban Sustainability Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLES ----------------
st.markdown("""
<style>
    .stApp { background-color: #0b0c10; color: #e6f2ff; }
    .big-title { color: #66fcf1; font-size:36px; font-weight:700; text-align:center; }
    .subtitle { color: #cfeefc; text-align:center; margin-bottom:20px; }
    .card { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; }
    .small { color: #a8cfe6; font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🌍 Aethra — Urban Sustainability Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Using NASA Earth observation data + ML to help planners design healthier cities (Astana demo)</div>", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
@st.cache_data
def load_csv(path="astana_analysis.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    # fallback: synthetic demo data
    np.random.seed(42)
    districts = ["Almaty", "Saryarka", "Yesil", "Baikonur", "Nura"]
    rows = []
    for d in districts:
        for i in range(60):
            lat = 51.10 + np.random.rand()*0.2
            lon = 71.30 + np.random.rand()*0.2
            pm25 = np.random.normal(20 + districts.index(d)*3, 6)
            temp = np.random.normal(10, 8)
            hum = np.clip(60 + 20*np.sin(i/10)+np.random.normal(0,5), 20, 95)
            pop = 150000 + districts.index(d)*50000
            green = max(3, 12 - districts.index(d)*2 + np.random.normal(0,1))
            traffic = 40000 + districts.index(d)*15000 + np.random.normal(0,8000)
            month = (i % 12) + 1
            rows.append([d, lat, lon, pm25, temp, hum, pop, green, traffic, month])
    df_demo = pd.DataFrame(rows, columns=["district","latitude","longitude","pm25","temperature","humidity","population","green_spaces","traffic_volume","month"])
    return df_demo

def try_load_model(path="pollution_model.pkl"):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Cannot load model from {path}: {e}")
            return None
    else:
        return None

def plot_avg_pm25(df):
    fig, ax = plt.subplots(figsize=(6,4))
    order = df.groupby("district")["pm25"].mean().sort_values(ascending=False)
    sns.barplot(x=order.values, y=order.index, palette="rocket", ax=ax)
    ax.set_xlabel("Avg PM2.5 (µg/m³)")
    ax.set_ylabel("")
    ax.set_title("Average PM2.5 by District")
    st.pyplot(fig)

def plot_monthly_pattern(df):
    fig, ax = plt.subplots(figsize=(6,3))
    monthly = df.groupby("month")["pm25"].mean()
    sns.lineplot(x=monthly.index, y=monthly.values, marker="o", ax=ax, color="#f39c12")
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Monthly Average PM2.5")
    ax.set_xticks(range(1,13))
    st.pyplot(fig)

def plot_feature_importance(model, feature_names):
    if model is None:
        st.info("Feature importance will appear here when a compatible model is uploaded.")
        return
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        fig, ax = plt.subplots(figsize=(6,3))
        order = np.argsort(fi)
        sns.barplot(x=fi[order], y=np.array(feature_names)[order], palette="viridis", ax=ax)
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.info("Model does not expose feature_importances_. Try a tree-based model for importance plot.")

# ---------------- LOAD DATA & MODEL ----------------
df = load_csv()
model = try_load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls & Scenario")
tab_choice = st.sidebar.radio("Go to", ["Overview","Map","ML Model","Insights","About"])
districts = sorted(df['district'].unique().tolist())
selected_district = st.sidebar.selectbox("District (map focus)", districts)
show_heat = st.sidebar.checkbox("Heatmap", True)
filter_level = st.sidebar.selectbox("Filter by PM2.5 level", ["All","Low (<15)","Moderate (15-25)","High (>=25)"])

# ---------------- OVERVIEW TAB ----------------
if tab_choice == "Overview":
    st.header("Overview")
    st.markdown("""
    **Aethra** demonstrates how urban planners can use NASA Earth observation data (AOD, NDVI, etc.) 
    combined with local socio-economic data to identify pollution hotspots and prioritize green infrastructure.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_pm = df['pm25'].mean()
        st.metric("City average PM2.5", f"{avg_pm:.1f} µg/m³")
    with col2:
        max_pm = df.groupby("district")["pm25"].mean().max()
        worst = df.groupby("district")["pm25"].mean().idxmax()
        st.metric("Worst district (avg)", f"{worst}: {max_pm:.1f}")
    with col3:
        rows = len(df)
        st.metric("Data points", f"{rows}")
    st.markdown("---")
    st.subheader("Quick Visuals")
    c1, c2 = st.columns(2)
    with c1:
        plot_avg_pm25(df)
    with c2:
        plot_monthly_pattern(df)

# ---------------- MAP TAB ----------------
elif tab_choice == "Map":
    st.header("Interactive Map — Pollution & Vegetation")
    st.markdown("Use the controls in the sidebar to filter or toggle heatmap. Click markers for details.")
    # Filter by level
    df_map = df.copy()
    if filter_level != "All":
        if filter_level == "Low (<15)":
            df_map = df_map[df_map["pm25"] < 15]
        elif filter_level == "Moderate (15-25)":
            df_map = df_map[(df_map["pm25"] >= 15) & (df_map["pm25"] < 25)]
        else:
            df_map = df_map[df_map["pm25"] >= 25]

    center = [df_map["latitude"].mean(), df_map["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB dark_matter")

    # heatmap
    if show_heat and not df_map.empty:
        heat_data = df_map[["latitude","longitude","pm25"]].values.tolist()
        HeatMap(heat_data, radius=25, blur=20, max_zoom=13, min_opacity=0.3).add_to(m)

    # district markers
    for _, r in df_map.iterrows():
        color = "green" if r["pm25"] < 15 else "orange" if r["pm25"] < 25 else "red"
        popup_html = f"""<b>{r['district']}</b><br>PM2.5: {r['pm25']:.1f} µg/m³<br>
                         Temp: {r['temperature']:.1f}°C<br>Humidity: {r['humidity']:.0f}%<br>
                         Green km²: {r['green_spaces']:.1f}<br>Traffic: {int(r['traffic_volume'])}"""
        folium.CircleMarker(location=[r["latitude"], r["longitude"]],
                            radius=7, color=color, fill=True, fill_opacity=0.8,
                            popup=popup_html).add_to(m)

    # add layer control and show
    folium.LayerControl().add_to(m)
    st_data = st_folium(m, width=1000, height=600)

# ---------------- ML MODEL TAB ----------------
elif tab_choice == "ML Model":
    st.header("Machine Learning — Predict PM2.5")
    st.markdown("Use the model to predict PM2.5 for a scenario (or upload your model file named 'pollution_model.pkl').")
    # scenario inputs
    st.subheader("Scenario inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        temp_in = st.number_input("Temperature (°C)", value=float(df['temperature'].median()))
        hum_in = st.number_input("Humidity (%)", value=float(df['humidity'].median()))
    with col2:
        pop_in = st.number_input("Population", value=int(df['population'].median()))
        green_in = st.number_input("Green spaces (km²)", value=float(df['green_spaces'].median()))
    with col3:
        traffic_in = st.number_input("Traffic volume", value=int(df['traffic_volume'].median()))
        month_in = st.slider("Month", 1, 12, 6)

    # try to predict
    if model is None:
        st.warning("No model loaded. Upload 'pollution_model.pkl' in the repo root to enable predictions.")
    else:
        features = np.array([[temp_in, hum_in, pop_in, green_in, traffic_in, month_in]])
        try:
            pred = model.predict(features)[0]
            st.markdown(f"### 🔮 Predicted PM2.5: **{pred:.2f} µg/m³**")
            if pred >= 25:
                st.error("⚠️ High pollution predicted — prioritize green infrastructure & traffic measures.")
            elif pred >= 15:
                st.warning("🟠 Moderate pollution — consider targeted greening.")
            else:
                st.success("🟢 Low pollution — maintain current green coverage.")
            # feature importance if available
            st.markdown("#### Feature importance (if available):")
            plot_feature_importance(model, ["temperature","humidity","population","green_spaces","traffic_volume","month"])
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # show model diagnostics: sample predicted vs actual for a few rows
    if model is not None:
        st.markdown("---")
        st.subheader("Model sample predictions vs actual")
        sample_df = df.sample(8, random_state=42)
        X_sample = sample_df[["temperature","humidity","population","green_spaces","traffic_volume","month"]]
        try:
            preds = model.predict(X_sample)
            sample_df = sample_df.copy()
            sample_df["predicted_pm25"] = preds
            st.dataframe(sample_df[["district","pm25","predicted_pm25"]].reset_index(drop=True))
        except Exception:
            st.info("Could not compute sample predictions with the loaded model.")

# ---------------- INSIGHTS TAB ----------------
elif tab_choice == "Insights":
    st.header("Insights & Recommendations")
    st.markdown("""
    **How urban planners can use Aethra:**  
    1. Identify high-risk zones (heatmap + markers).  
    2. Simulate scenarios (change traffic, add green space) and see predicted PM2.5.  
    3. Prioritize interventions: parks, traffic restrictions, monitoring.  
    4. Integrate satellite NDVI and AOD to track vegetation and aerosol trends.
    """)
    # priority table
    priority = (df.groupby("district")["pm25"].mean().sort_values(ascending=False).reset_index().rename(columns={"pm25":"avg_pm25"}))
    priority["priority"] = priority["avg_pm25"].rank(method="dense", ascending=False).astype(int)
    st.subheader("Priority districts (higher = more urgent)")
    st.table(priority)

# ---------------- ABOUT TAB ----------------
elif tab_choice == "About":
    st.header("About Aethra")
    st.markdown("""
    **Aethra** — a demo tool that shows how NASA Earth observation data combined with local metrics (population, traffic, green spaces)
    can support decision-making for healthier cities.  

    **Team:** Amina (lead)  
    **Technologies:** Streamlit, Folium, scikit-learn, NASA datasets (AOD/NDVI), WorldPop  
    """)
    st.markdown("---")
    st.markdown("**How to cite / reproduce:**\n\n1. Provide `astana_analysis.csv` in repo root.\n2. Provide `pollution_model.pkl` in repo root (trained RandomForest or compatible regressor).\n3. Deploy this app on Streamlit Cloud.")
    st.markdown("")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:#9dded8;'>Built for NASA Space Apps — Data Pathways to Healthy Cities</div>", unsafe_allow_html=True)
