import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Simulation
np.random.seed(42)
n_records = 1000

data = {
    'Route_Distance_km': np.random.uniform(50, 1000, n_records),
    'Shipment_Volume_kg': np.random.uniform(10, 5000, n_records),
    'Vehicle_Type': np.random.choice(['Van', 'Truck', 'Trailer'], n_records, p=[0.4, 0.4, 0.2])
}
df = pd.DataFrame(data)

# Simulate dependent variables with some realistic noise
df['Transportation_Cost_USD'] = (df['Route_Distance_km'] * 0.5) + (df['Shipment_Volume_kg'] * 0.05) + np.random.normal(50, 20, n_records)
df['Delivery_Time_Days'] = (df['Route_Distance_km'] / 200) + np.random.gamma(shape=2.0, scale=1.0, size=n_records)

# 2. Exploratory Data Analysis (EDA)
print("--- Central Tendencies and Summary Statistics ---")
print(df.describe())

print("\n--- Correlation Matrix ---")
# Exclude categorical data for correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.corr())

# 3. Visualizations
sns.set_theme(style="whitegrid")

# Figure 1: Distribution of Delivery Times
plt.figure(figsize=(10, 6))
sns.histplot(df['Delivery_Time_Days'], kde=True, bins=30, color='blue')
plt.title('Distribution of Delivery Times')
plt.xlabel('Delivery Time (Days)')
plt.ylabel('Frequency')
plt.savefig('delivery_time_distribution.png')
plt.show()

# Figure 2: Route Distance vs. Transportation Cost
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Route_Distance_km', y='Transportation_Cost_USD', hue='Vehicle_Type', data=df, alpha=0.7)
plt.title('Route Distance vs. Transportation Cost')
plt.xlabel('Route Distance (km)')
plt.ylabel('Transportation Cost (USD)')
plt.savefig('distance_vs_cost.png')
plt.show()

# Figure 3: Shipment Volume by Vehicle Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Vehicle_Type', y='Shipment_Volume_kg', data=df, palette='Set2')
plt.title('Shipment Volume Distribution by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Shipment Volume (kg)')
plt.savefig('volume_by_vehicle.png')
plt.show()

# Figure 4: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Logistics Variables')
plt.savefig('correlation_heatmap.png')
plt.show()
