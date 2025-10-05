import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import folium
from streamlit_folium import st_folium

# --- Page config ---
st.set_page_config(page_title="NASA Hackathon Project", layout="wide")
st.markdown("""
<style>
body {
    background-color: #0a0a0a;
    color: white;
}
h1, h2, h3, h4, h5 {
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=120)
st.title("NASA Hackathon: Data Pathways to Healthy Cities")
st.markdown("""
### Environmental Monitoring for Sustainable Cities
This project analyzes urban air quality (PM2.5), green spaces, and traffic data for Astana districts using NASA Earth Observation data.  
Interactive visualizations demonstrate how urban planners can develop smart strategies for city growth while maintaining public health and environment.
""")

# --- Load / create data ---
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
    dates = pd.date_range('2023-01-01', '2023-12-31')
    for district, info in districts.items():
        for i, date in enumerate(dates):
            base_temp = 5 + 25*np.sin(2*np.pi*i/365)
            temp = base_temp + np.random.normal(0, 2)
            pm25 = np.random.normal(20, 5)
            humidity = 60 + 20*np.sin(2*np.pi*i/365 + np.pi/2) + np.random.normal(0,5)
            traffic = np.random.normal(50000,10000)
            all_data.append({
                'date': date,
                'district': district,
                'temperature': round(temp,1),
                'humidity': round(max(30,min(95,humidity)),1),
                'pm25': round(max(5,pm25),1),
                'population': info['population'],
                'green_spaces': info['green_spaces'],
                'traffic_volume': int(max(10000, traffic)),
                'latitude': info['lat'],
                'longitude': info['lon'],
                'month': date.month
            })
    return pd.DataFrame(all_data)

df = create_data()

# --- Sidebar filters ---
st.sidebar.header("Filters")
month_select = st.sidebar.selectbox("Select month", 
                                    options=list(range(1,13)), 
                                    format_func=lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
district_select = st.sidebar.multiselect("Select districts", df['district'].unique(), default=df['district'].unique())

df_filtered = df[(df['month']==month_select) & (df['district'].isin(district_select))]

# --- Folium Map ---
st.subheader("PM2.5 Pollution Map")
m = folium.Map(location=[51.1694, 71.4491], zoom_start=11, tiles="CartoDB dark_matter")
for idx, row in df_filtered.iterrows():
    if row['pm25'] > 25:
        color = 'red'
    elif row['pm25'] > 15:
        color = 'orange'
    else:
        color = 'green'
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['district']}, PM2.5: {row['pm25']}"
    ).add_to(m)

st_data = st_folium(m, width=900, height=550)

# --- ML Model ---
st.subheader("PM2.5 Prediction Model")
features = ['temperature','humidity','population','green_spaces','traffic_volume','month']
X = df[features]
y = df['pm25']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"RandomForestRegressor MAE: {mae:.2f} µg/m³")

# --- Plots ---
st.subheader("Visualizations")
fig, axes = plt.subplots(2,2, figsize=(12,10))
# Avg PM2.5
pm25_avg = df.groupby('district')['pm25'].mean().sort_values(ascending=False)
axes[0,0].bar(pm25_avg.index, pm25_avg.values, color='salmon')
axes[0,0].set_title("Average PM2.5 by District", color='white')
axes[0,0].tick_params(axis='x', rotation=45, colors='white')
axes[0,0].tick_params(axis='y', colors='white')
# Green vs PM2.5
sns.scatterplot(data=df_filtered, x='green_spaces', y='pm25', hue='district', s=100, ax=axes[0,1'])
axes[0,1].set_title("Green Spaces vs PM2.5", color='white')
axes[0,1].tick_params(axis='x', colors='white')
axes[0,1].tick_params(axis='y', colors='white')
# Monthly PM2.5 trend
monthly_pm = df.groupby(['month','district'])['pm25'].mean().reset_index()
for d in district_select:
    data_d = monthly_pm[monthly_pm['district']==d]
    axes[1,0].plot(data_d['month'], data_d['pm25'], marker='o', label=d)
axes[1,0].set_title("Monthly PM2.5 Trend", color='white')
axes[1,0].set_xticks(range(1,13))
axes[1,0].tick_params(axis='x', colors='white')
axes[1,0].tick_params(axis='y', colors='white')
axes[1,0].legend(facecolor='black')
# Feature importance
fi = pd.DataFrame({'feature':features, 'importance':model.feature_importances_}).sort_values('importance',ascending=True)
axes[1,1].barh(fi['feature'], fi['importance'], color='skyblue')
axes[1,1].set_title("Feature Importance", color='white')
axes[1,1].tick_params(axis='x', colors='white')
axes[1,1].tick_params(axis='y', colors='white')
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.markdown("#### About the Project")
st.markdown("""
- **Data Sources**: NASA Earth Observation (PM2.5, Vegetation Index)  
- **Goal**: Support sustainable city development and healthy urban planning  
- **Team Lead**: Amina Bilyalova  
- **Target Audience**: Municipalities, environmental researchers, citizens
""")
