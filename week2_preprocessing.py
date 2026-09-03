import pandas as pd
import numpy as np

def preprocess_logistics_data(filepath):
    # 1. Load Data
    print("Loading raw logistics dataset...")
    df = pd.read_csv(filepath)
    
    # 2. Handling Missing Values
    # Drop rows where critical timestamps are missing
    df.dropna(subset=['dispatch_timestamp', 'delivery_timestamp'], inplace=True)
    
    # Impute missing weights with the median weight
    median_weight = df['package_weight_lbs'].median()
    df['package_weight_lbs'].fillna(median_weight, inplace=True)
    
    # 3. Outlier Detection (IQR Method for distance)
    Q1 = df['distance_miles'].quantile(0.25)
    Q3 = df['distance_miles'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out extreme distance anomalies and impossible negative distances
    df = df[(df['distance_miles'] >= max(0, lower_bound)) & (df['distance_miles'] <= upper_bound)]
    
    # 4. Normalization and Feature Engineering
    # Convert weight to kilograms
    df['package_weight_kg'] = df['package_weight_lbs'] * 0.453592
    
    # Min-Max Scaling for Traffic Index (assuming original scale is 0-100)
    df['traffic_index_scaled'] = (df['traffic_index'] - df['traffic_index'].min()) / \
                                 (df['traffic_index'].max() - df['traffic_index'].min())
    
    print("Data preprocessing complete. Clean dataset ready for EDA.")
    return df

# Example usage:
# clean_df = preprocess_logistics_data('raw_logistics_data.csv')
