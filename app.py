import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide", page_title="Bird Population & Climate Impact")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("Occurance & Climate Data.xlsx")
    return df

df = load_data()

st.title("🐦 Impact of Climate Change on Bird Populations")

# Sidebar filters
st.sidebar.title("🌍 Impact of Climate Change on Birds")
st.sidebar.markdown("Download or explore the dataset and begin your analysis.")

with open("Occurance & Climate Data.xlsx", "rb") as file:
    st.sidebar.download_button(
        label="📥 Download Dataset",
        data=file,
        file_name="Occurance & Climate Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

selected_species = st.sidebar.selectbox("Choose Bird Species", sorted(df["Bird_Species"].unique()))
year_range = st.sidebar.slider("Year Range", int(df["Year"].min()), int(df["Year"].max()), (1980, 2010))

df_filtered = df[(df["Bird_Species"] == selected_species) & (df["Year"].between(*year_range))]

st.markdown(f"### Data for **{selected_species}** from {year_range[0]} to {year_range[1]}")

# Line plot of population trend
fig1, ax1 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_filtered, x="Year", y="Population", marker="o", ax=ax1)
ax1.set_title(f"Population Trend for {selected_species}")
st.pyplot(fig1)

# Correlation heatmap
st.markdown("#### 🔍 Correlation with Population")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.heatmap(df_filtered.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# ML prediction model
st.markdown("### 🤖 Predict Future Bird Population Based on Climate Factors")

features = ["Temperature", "Precipitation", "Shift_km", "Traffic", "Year"]
target = "Population"
model_df = df[df["Bird_Species"] == selected_species]

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"**Model Performance for {selected_species}:**")
st.write(f"• R² Score: {r2:.3f}")
st.write(f"• RMSE: {rmse:.2f}")

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()

st.markdown("#### 📊 Feature Importance")
fig3, ax3 = plt.subplots()
importances.plot(kind='barh', ax=ax3, color='teal')
ax3.set_title("Which Factors Affect Bird Population?")
st.pyplot(fig3)

# User input for prediction
st.markdown("### 📈 Predict Population Under New Climate Scenario")

col1, col2, col3 = st.columns(3)
temp_input = col1.slider("Temperature (°C)", float(df["Temperature"].min()), float(df["Temperature"].max()), 25.0)
precip_input = col2.slider("Precipitation (mm)", float(df["Precipitation"].min()), float(df["Precipitation"].max()), 1500.0)
shift_input = col3.slider("Shift Distance (km)", float(df["Shift_km"].min()), float(df["Shift_km"].max()), 10.0)

col4, col5 = st.columns(2)
traffic_input = col4.slider("Traffic Index", float(df["Traffic"].min()), float(df["Traffic"].max()), 2000.0)
year_input = col5.slider("Prediction Year", 2025, 2100, 2030)

input_data = pd.DataFrame([[temp_input, precip_input, shift_input, traffic_input, year_input]], columns=features)
predicted_population = rf_model.predict(input_data)[0]

st.success(f"📌 Predicted Population for {selected_species} in {year_input}: **{int(predicted_population)} birds**")

# Conservation insight
st.markdown("### 🌿 Conservation Insight")
important = importances.idxmax()
st.info(f"**Key Factor for {selected_species}:** {important} — Consider monitoring this for effective conservation planning.")

# Optional map
st.markdown("### 🗺️ Geographical Sightings Map")
map_df = df[df["Bird_Species"] == selected_species]
map_fig = px.scatter_mapbox(map_df,
                            lat="Latitude", lon="Longitude",
                            color="Population",
                            size="Population",
                            hover_name="Country",
                            mapbox_style="carto-positron",
                            zoom=1,
                            height=400)
st.plotly_chart(map_fig, use_container_width=True)

# =======================================
# 🧭 SPECIES CLASSIFICATION / ALERT SYSTEM
# =======================================

st.markdown("## 🚨 Species Trend Classification & Alerts")

# Trend classification function
@st.cache_data
def classify_species_trends(data):
    results = []
    for species in data["Bird_Species"].unique():
        sub_df = data[data["Bird_Species"] == species]
        if len(sub_df) < 10:
            continue
        X = sub_df[["Year"]]
        y = sub_df["Population"]
        lr = LinearRegression()
        lr.fit(X, y)
        slope = lr.coef_[0]
        results.append({
            "Species": species,
            "Trend": "Declining" if slope < 0 else "Stable/Increasing",
            "Slope": slope,
            "SampleSize": len(sub_df)
        })
    return pd.DataFrame(results).sort_values("Slope")

trend_df = classify_species_trends(df)

# Show summary metrics
n_declining = trend_df[trend_df["Trend"] == "Declining"].shape[0]
n_stable = trend_df[trend_df["Trend"] == "Stable/Increasing"].shape[0]
st.write(f"**📉 Declining Species:** {n_declining}")
st.write(f"**📈 Stable/Increasing Species:** {n_stable}")

# Show interactive table
st.markdown("### 📋 Species Trend Table")
selected_trend = st.selectbox("Filter by Trend", ["All", "Declining", "Stable/Increasing"])

if selected_trend != "All":
    filtered_trend_df = trend_df[trend_df["Trend"] == selected_trend]
else:
    filtered_trend_df = trend_df

st.dataframe(filtered_trend_df.style.background_gradient(cmap="coolwarm", subset=["Slope"]))

# Highlight top 5 at risk
st.markdown("### 🔴 Top 5 Most At-Risk Species (by negative slope)")
top_declining = trend_df[trend_df["Trend"] == "Declining"].head(5)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=top_declining, y="Species", x="Slope", palette="Reds_r", ax=ax)
ax.set_title("Top 5 Declining Species (Negative Population Slope)")
st.pyplot(fig)
