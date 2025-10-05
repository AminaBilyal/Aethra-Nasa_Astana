import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import joblib
from folium.plugins import HeatMap

st.set_page_config(page_title="NASA Hackathon Project", layout="wide")

# ---- Sidebar for file uploads ----
st.sidebar.header("Upload your data files")
csv_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
pkl_file = st.sidebar.file_uploader("Upload ML model file (.pkl)", type=["pkl"])

st.title("NASA Hackathon: Data Pathways to Healthy Cities 🌍")
st.markdown("""
### Environmental Monitoring for Sustainable Cities
This project analyzes urban air quality and green zones using NASA Earth observation data.  
We demonstrate how urban planners can develop smart strategies to maintain both wellbeing and the environment.
""")

st.markdown("**Project by:** Amina Bilyalova and team  \n**Technologies:** Python, Streamlit, Folium, ML, NASA Open Data")

# ---- Load CSV ----
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.success("CSV loaded successfully!")
else:
    st.warning("Upload your CSV file to continue.")
    st.stop()

# ---- Load ML model ----
if pkl_file is not None:
    model = joblib.load(pkl_file)
    st.success("ML model loaded successfully!")
else:
    st.warning("Upload your ML model (.pkl) to continue.")
    st.stop()

# ---- Sidebar filters ----
st.sidebar.header("Filters")
districts = df['district'].unique()
selected_districts = st.sidebar.multiselect("Select districts", districts, default=list(districts))

df_filtered = df[df['district'].isin(selected_districts)]

# ---- Display ML predictions ----
st.subheader("ML Predictions for PM2.5")
features = ['temperature', 'humidity', 'population', 'green_spaces', 'traffic_volume', 'month']
X = df_filtered[features]
y_pred = model.predict(X)
df_filtered['predicted_pm25'] = y_pred
st.dataframe(df_filtered[['date','district','pm25','predicted_pm25']].head(10))

# ---- Visualization 1: Graphs ----
st.subheader("Visualizations")

fig, axes = plt.subplots(2, 2, figsize=(15,12))

# Average PM2.5 by district
pm25_data = df_filtered.groupby('district')['pm25'].mean()
axes[0,0].bar(pm25_data.index, pm25_data.values, color='tomato')
axes[0,0].set_title('Average PM2.5 by District')
axes[0,0].set_ylabel('PM2.5 µg/m³')

# Green spaces vs PM2.5
green_pm25 = df_filtered.groupby('district').agg({'green_spaces':'first','pm25':'mean'})
axes[0,1].scatter(green_pm25['green_spaces'], green_pm25['pm25'], s=100, color='green')
for i, row in green_pm25.iterrows():
    axes[0,1].annotate(i, (row['green_spaces'], row['pm25']))
axes[0,1].set_title('Green Spaces vs PM2.5')
axes[0,1].set_xlabel('Green Spaces (km²)')
axes[0,1].set_ylabel('PM2.5')

# Seasonal PM2.5
df_filtered['month'] = pd.to_datetime(df_filtered['date']).dt.month
monthly_pm25 = df_filtered.groupby('month')['pm25'].mean()
axes[1,0].plot(monthly_pm25.index, monthly_pm25.values, marker='o', color='blue')
axes[1,0].set_title('Seasonal PM2.5 Patterns')
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('PM2.5')

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)
axes[1,1].barh(feature_importance['feature'], feature_importance['importance'], color='purple')
axes[1,1].set_title('Feature Importance for PM2.5 Prediction')

plt.tight_layout()
st.pyplot(fig)

# ---- Visualization 2: Folium map ----
st.subheader("Interactive Map of Astana Pollution Levels")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="CartoDB positron")

# Heatmap
heat_data = [[row['latitude'], row['longitude'], row['pm25']] for index, row in df_filtered.iterrows()]
HeatMap(heat_data, radius=25, max_zoom=13).add_to(m)

# Optional: add district markers
for index, row in df_filtered.groupby('district').first().iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"{index} District",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

st_data = st_folium(m, width=800, height=500)

st.markdown("""
---
### Recommendations
- Focus green infrastructure in districts with highest PM2.5
- Use ML predictions to plan park development and traffic management
- Monitor seasonal patterns for effective urban planning
""")
