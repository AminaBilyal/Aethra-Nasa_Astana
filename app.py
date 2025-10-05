import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="NASA Hackathon Project", layout="wide")

st.title("NASA Hackathon 2025: Data Pathways to Healthy Cities 🌍")
st.markdown("""
### Environmental Monitoring for Sustainable Cities
Автор: Amina Bilyalova  
Проект демонстрирует, как урбанист может использовать NASA Earth observation data для анализа загрязнений и зелёных зон, чтобы развивать города устойчиво и безопасно.
""")

# --- Generate data like in Colab ---
@st.cache_data
def create_data():
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
            temp = base_temp + np.random.normal(0, 2)
            pm25 = np.random.normal(20, 5)
            humidity = 60 + 20 * np.sin(2 * np.pi * i / 365 + np.pi/2) + np.random.normal(0, 5)
            traffic = np.random.normal(50000, 10000)
            park_need = max(0, (pm25 - 10) * info['population'] / 10000)

            all_data.append({
                'date': date,
                'district': district,
                'temperature': round(temp, 1),
                'humidity': max(30, min(95, round(humidity, 1))),
                'pm25': max(5, round(pm25, 1)),
                'population': info['population'],
                'green_spaces': info['green_spaces'],
                'traffic_volume': max(10000, traffic),
                'park_need_index': round(park_need, 1),
                'latitude': info['lat'],
                'longitude': info['lon']
            })
    return pd.DataFrame(all_data)

df = create_data()

# --- ML Model ---
@st.cache_data
def train_model(df):
    df_ml = df.copy()
    df_ml['day_of_year'] = df_ml['date'].dt.dayofyear
    df_ml['month'] = df_ml['date'].dt.month
    features = ['temperature', 'humidity', 'population', 'green_spaces', 'traffic_volume', 'month']
    X = df_ml[features]
    y = df_ml['pm25']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return model, mae

model, mae = train_model(df)
st.subheader("ML Model Accuracy")
st.info(f"Mean Absolute Error: {mae:.2f} µg/m³ for PM2.5 prediction")

# --- Plots ---
st.subheader("Urban Pollution & Environmental Data Visualizations")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. PM2.5 by district
pm25_data = df.groupby('district')['pm25'].mean().sort_values(ascending=False)
axes[0,0].bar(pm25_data.index, pm25_data.values, color='tomato')
axes[0,0].axhline(25, color='red', linestyle='--', label='Unhealthy')
axes[0,0].axhline(15, color='orange', linestyle='--', label='Moderate')
axes[0,0].set_title('Average PM2.5 by District')
axes[0,0].legend()

# 2. Green spaces vs pollution
green_pm25 = df.groupby('district').agg({'green_spaces': 'first', 'pm25': 'mean'})
axes[0,1].scatter(green_pm25['green_spaces'], green_pm25['pm25'], s=120)
for i, row in green_pm25.iterrows():
    axes[0,1].annotate(i, (row['green_spaces'], row['pm25']), xytext=(5,5), textcoords='offset points')
axes[0,1].set_xlabel('Green Spaces (km²)')
axes[0,1].set_ylabel('PM2.5')
axes[0,1].set_title('Green Spaces vs Pollution')

# 3. Seasonal PM2.5
df['month'] = df['date'].dt.month
monthly_pm25 = df.groupby('month')['pm25'].mean()
axes[1,0].plot(monthly_pm25.index, monthly_pm25.values, marker='o', linewidth=2, color='teal')
axes[1,0].set_title('Seasonal PM2.5 Patterns')
axes[1,0].set_xlabel('Month')
axes[1,0].set_ylabel('PM2.5')

# 4. Feature importance
features = ['temperature', 'humidity', 'population', 'green_spaces', 'traffic_volume', 'month']
importances = model.feature_importances_
axes[1,1].barh(features, importances, color='purple')
axes[1,1].set_title('Feature Importance for PM2.5 Prediction')

plt.tight_layout()
st.pyplot(fig)

# --- Folium Map with HeatMap ---
st.subheader("Interactive Pollution Map (Astana)")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="CartoDB positron")
heat_data = [[row['latitude'], row['longitude'], row['pm25']] for index, row in df.iterrows()]
HeatMap(heat_data, radius=20, blur=15, max_zoom=13).add_to(m)
st_folium(m, width=800, height=500)

# --- Recommendations ---
st.subheader("Urban Planning Recommendations")
worst = df.groupby('district')['pm25'].mean().idxmax()
best = df.groupby('district')['pm25'].mean().idxmin()
st.markdown(f"""
**Priority District for Action:** {worst}  
Actions: Monitor air quality, increase green infrastructure, traffic management  

**Best Practice District:** {best}  
Actions: Study and replicate successful strategies  

**Immediate Park Development Needed:**  
- {df.groupby('district')['park_need_index'].mean().sort_values(ascending=False).index[0]}  
- {df.groupby('district')['park_need_index'].mean().sort_values(ascending=False).index[1]}  
""")

st.success("✅ Demo project loaded successfully with interactive map, ML predictions, and visualizations")
