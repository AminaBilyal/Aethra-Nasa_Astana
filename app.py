# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="NASA Hackathon Project",
    layout="wide",
    page_icon="🚀",
)

# --- HEADER ---
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:10px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg" width="80">
        <h1 style="color:white;">NASA Hackathon: Data Pathways to Healthy Cities</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---", unsafe_allow_html=True)

# --- DARK THEME ---
st.markdown(
    """
    <style>
    .main {background-color: #0c0c0c; color: white;}
    .stButton>button {background-color: #1a1a1a; color: white;}
    .stSelectbox>div>div {background-color: #1a1a1a; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- LOAD OR CREATE DATA ---
@st.cache_data
def create_astana_data():
    districts = {
        'Almaty': {'lat': 51.15, 'lon': 71.45, 'population': 350000, 'green_spaces': 12},
        'Saryarka': {'lat': 51.20, 'lon': 71.40, 'population': 280000, 'green_spaces': 8},
        'Yesil': {'lat': 51.18, 'lon': 71.35, 'population': 320000, 'green_spaces': 15},
        'Nura': {'lat': 51.12, 'lon': 71.50, 'population': 190000, 'green_spaces': 10},
        'Baikonur': {'lat': 51.22, 'lon': 71.48, 'population': 150000, 'green_spaces': 9}
    }
    all_data = []
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    for district, info in districts.items():
        for i, date in enumerate(dates):
            base_temp = 5 + 25 * np.sin(2 * np.pi * i / 365)
            if district == 'Saryarka':
                temp = base_temp + np.random.normal(2,1.5)
                pm25 = np.random.normal(28,6)
                traffic = np.random.normal(80000,10000)
            elif district == 'Almaty':
                temp = base_temp + np.random.normal(1,2)
                pm25 = np.random.normal(22,5)
                traffic = np.random.normal(95000,15000)
            elif district == 'Yesil':
                temp = base_temp + np.random.normal(-1,1)
                pm25 = np.random.normal(15,3)
                traffic = np.random.normal(60000,8000)
            else:
                temp = base_temp + np.random.normal(0,1.5)
                pm25 = np.random.normal(20,4)
                traffic = np.random.normal(50000,7000)
            humidity = 60 + 20*np.sin(2*np.pi*i/365 + np.pi/2) + np.random.normal(0,5)
            park_need = max(0, (pm25 - 10) * info['population']/10000)
            all_data.append({
                'date': date,
                'district': district,
                'temperature': round(temp,1),
                'humidity': max(30,min(95,round(humidity,1))),
                'pm25': max(5,round(pm25,1)),
                'population': info['population'],
                'green_spaces': info['green_spaces'],
                'traffic_volume': max(10000, traffic),
                'park_need_index': round(park_need,1),
                'latitude': info['lat'],
                'longitude': info['lon'],
                'month': date.month
            })
    return pd.DataFrame(all_data)

df = create_astana_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
selected_month = st.sidebar.slider("Select month:", 1, 12, 1)
selected_pm_levels = st.sidebar.multiselect(
    "Select pollution levels to display:",
    ["Low (<15)", "Moderate (15-25)", "High (>25)"],
    default=["Low (<15)", "Moderate (15-25)", "High (>25)"]
)

filtered_df = df[df['month']==selected_month]

# --- INTERACTIVE MAP ---
st.subheader(f"Astana Pollution Map: Month {selected_month}")

m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="CartoDB dark_matter")

# Add heatmap for all points
heat_data = [[row['latitude'], row['longitude'], row['pm25']] for index,row in filtered_df.iterrows()]
HeatMap(heat_data, min_opacity=0.5, radius=25, blur=15, max_val=50).add_to(m)

# CircleMarkers by pollution levels
for index,row in filtered_df.iterrows():
    if row['pm25']<15 and "Low (<15)" in selected_pm_levels:
        color = "green"
    elif 15<=row['pm25']<=25 and "Moderate (15-25)" in selected_pm_levels:
        color = "orange"
    elif row['pm25']>25 and "High (>25)" in selected_pm_levels:
        color = "red"
    else:
        continue
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['district']} PM2.5: {row['pm25']}"
    ).add_to(m)

st_data = st_folium(m, width=800, height=500)

# --- ML MODEL ---
st.subheader("ML Model: PM2.5 Prediction")
features = ['temperature', 'humidity', 'population', 'green_spaces', 'traffic_volume', 'month']
X = df[features]
y = df['pm25']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.write(f"Model Mean Absolute Error: {mae:.2f} µg/m³")

# Feature importance plot
st.subheader("Feature Importance")
importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=True)
fig, ax = plt.subplots(figsize=(6,4))
ax.barh(importances['feature'], importances['importance'], color='cyan')
ax.set_title("Feature Importance for PM2.5 Prediction")
st.pyplot(fig)

# --- ADDITIONAL GRAPHS ---
st.subheader("Pollution & Green Spaces Analysis")
fig2, axes = plt.subplots(1,2, figsize=(12,4))

pm25_means = df.groupby('district')['pm25'].mean().sort_values(ascending=False)
axes[0].bar(pm25_means.index, pm25_means.values, color='orange')
axes[0].set_title("Average PM2.5 by District")
axes[0].set_ylabel("PM2.5 µg/m³")
axes[0].tick_params(axis='x', rotation=45)

green_pm25 = df.groupby('district').agg({'green_spaces':'first','pm25':'mean'})
axes[1].scatter(green_pm25['green_spaces'], green_pm25['pm25'], color='lime', s=100)
for i,row in green_pm25.iterrows():
    axes[1].annotate(i, (row['green_spaces'], row['pm25']))
axes[1].set_xlabel("Green Spaces (km²)")
axes[1].set_ylabel("PM2.5")
axes[1].set_title("Green Spaces vs PM2.5")

st.pyplot(fig2)

# --- RECOMMENDATIONS ---
st.subheader("Urban Planning Recommendations")
worst = df.groupby('district')['pm25'].mean().idxmax()
best = df.groupby('district')['pm25'].mean().idxmin()
st.markdown(f"**Most polluted district:** {worst} → Actions: Air quality monitoring, Green infrastructure, Traffic management")
st.markdown(f"**Cleanest district:** {best} → Study and replicate successful strategies")
top_parks = df.groupby('district')['park_need_index'].mean().sort_values(ascending=False)
st.markdown(f"**Immediate Park Development Needed:** {top_parks.index[0]}, {top_parks.index[1]}")

st.success("🚀 Interactive demo ready! All zones and months selectable, ML model and map integrated.")
