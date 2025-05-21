# bird_climate_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_excel("Occurance & Climate Data.xlsx")

# Basic overview
print("\n--- Dataset Overview ---")
print(df.info())
print(df.describe())

# -------------------------------
# 1. Exploratory Data Analysis
# -------------------------------

# Population trend over time
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Year', y='Population', estimator='mean', ci=None)
plt.title("Average Bird Population Over Years")
plt.xlabel("Year")
plt.ylabel("Population")
plt.tight_layout()
plt.savefig("population_trend.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Between Variables")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Temperature vs Population (scatter)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Temperature', y='Population', hue='Bird_Species', legend=False)
plt.title("Temperature vs Population")
plt.tight_layout()
plt.savefig("temp_vs_population.png")
plt.close()

# -------------------------------
# 2. Geospatial Visualization
# -------------------------------
fig = px.scatter_geo(df,
                     lat='Latitude',
                     lon='Longitude',
                     color='Population',
                     hover_name='Bird_Species',
                     animation_frame='Year',
                     title='Bird Population by Location Over Time')
fig.write_html("geo_population_map.html")

# -------------------------------
# 3. Predictive Modeling
# -------------------------------
# Features and target
features = ['Temperature', 'Precipitation', 'Shift_km', 'Traffic', 'Year']
X = df[features]
y = df['Population']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel Mean Squared Error: {mse:.2f}")

# -------------------------------
# 4. Feature Importance
# -------------------------------
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 6))
feature_importance.plot(kind='bar', title='Feature Importance')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nAll plots and the HTML map are saved in the project folder.")
