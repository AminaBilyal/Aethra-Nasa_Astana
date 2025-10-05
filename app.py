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
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="NASA Hackathon: Healthy Cities", layout="wide", page_icon="🌍")

# --- Header ---
st.markdown("""
<div style="text-align:center;">
    <h1>🌍 NASA Hackathon: Data Pathways to Healthy Cities</h1>
    <h3>Environmental Monitoring & Urban Planning</h3>
    <p><b>Project by:</b> Amina Bilyalova</p>
</div>
""", unsafe_allow_html=True)

# Optional: NASA logo
st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/nasa-logo-web-rgb.png", width=150)

# --- Sidebar for CSV upload ---
st.sidebar.header("Upload CSV file")
csv_file = st.sidebar.file_uploader("Upload CSV file with districts data", type=["csv"])
if csv_file is None:
    st.warning("Please upload your CSV file to continue.")
    st.stop()

df = pd.read_csv(csv_file)

# --- Data preprocessing ---
df['month'] = pd.to_datetime(df['date']).dt.month

# --- ML model ---
features = ['temperature', 'humidity', 'population', 'green_spaces', 'traffic_volume', 'month']
X = df[features]
y = df['pm25']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

df['predicted_pm25'] = model.predict(X)

# --- ML Output ---
st.subheader("ML Predictions (PM2.5)")
st.dataframe(df[['date','district','pm25','predicted_pm25']].head(10))

# --- Visualizations ---
st.subheader("Visualizations")
fig, axes = plt.subplots(2, 2, figsize=(15,12))

# PM2.5 by district
pm25_data = df.groupby('district')['pm25'].mean()
axes[0,0].bar(pm25_data.index, pm25_data.values, color='tomato')
axes[0,0].set_title('Average PM2.5 by District')
axes[0,0].set_ylabel('PM2.5 µg/m³')

# Green spaces vs PM2.5
green_pm25 = df.groupby('district').agg({'green_spaces':'first','pm25':'mean'})
axes[0,1].scatter(green_pm25['green_spaces'], green_pm25['pm25'], s=100, color='green')
for i, row in green_pm25.iterrows():
    axes[0,1].annotate(i, (row['green_spaces'], row['pm25']))
axes[0,1].set_title('Green Spaces vs PM2.5')
axes[0,1].set_xlabel('Green Spaces (km²)')
axes[0,1].set_ylabel('PM2.5')

# Seasonal PM2.5 patterns
monthly_pm25 = df.groupby('month')['pm25'].mean()
axes[1,0].plot(monthly_pm25.index, monthly_pm25.values, marker='o', linewidth=2, color='blue')
axes[1,0].set_title('Seasonal PM2.5 Patterns')
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('PM2.5')

# Feature importance
feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance', ascending=True)
axes[1,1].barh(feature_importance['feature'], feature_importance['importance'], color='purple')
axes[1,1].set_title('Feature Importance for PM2.5 Prediction')

plt.tight_layout()
st.pyplot(fig)

# --- Interactive Folium Map ---
st.subheader("🌆 Astana Pollution Heatmap")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="CartoDB positron")

heat_data = [[row['latitude'], row['longitude'], row['pm25']] for _, row in df.iterrows()]
HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(m)

# District markers
for _, row in df.groupby('district').first().iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"{row['district']} District\nPM2.5: {df[df['district']==row['district']]['pm25'].mean():.1f}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

st_folium(m, width=900, height=550)

# --- Recommendations ---
st.subheader("📌 Urban Planning Recommendations")
worst_district = df.groupby('district')['pm25'].mean().idxmax()
best_district = df.groupby('district')['pm25'].mean().idxmin()
st.markdown(f"""
- Focus green infrastructure and air quality monitoring in **{worst_district}**.
- Study best practices in **{best_district}** for sustainable city planning.
- Monitor seasonal PM2.5 peaks and traffic hotspots.
- Use ML predictions to plan park development and traffic management.
""")

st.markdown("---")
st.markdown("<p style='text-align:center;'>🌐 Interactive demo by <b>Amina Bilyalova</b> | NASA Hackathon 2025</p>", unsafe_allow_html=True)
