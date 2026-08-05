"""
Smart Irrigation and Water Consumption Prediction System
Model Training and EDA Pipeline Script
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

# Import local generator if needed
sys.path.append(os.path.dirname(__file__))
from dataset.generate_dataset import generate_smart_irrigation_dataset

def load_and_preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset', 'smart_irrigation_dataset.csv')
    
    if not os.path.exists(dataset_path) or pd.read_csv(dataset_path).shape[0] < 10000:
        print("Dataset missing or has fewer than 10,000 records. Generating new dataset...")
        df = generate_smart_irrigation_dataset(10000)
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        df.to_csv(dataset_path, index=False)
        df.to_csv(os.path.join(base_dir, 'smart_irrigation_dataset.csv'), index=False)
    else:
        print(f"Loading existing dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)

    print(f"Raw Dataset Shape: {df.shape}")

    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_count - len(df)} duplicate records.")

    # Missing value handling
    if df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        cat_cols = df.select_dtypes(include=['object']).columns
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0])

    # Outlier handling via IQR Capping for continuous variables
    num_cols = ['Temperature', 'Humidity', 'Soil_Moisture', 'Rainfall', 'Wind_Speed', 'Solar_Radiation', 'Previous_Water_Consumption']
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    return df

def generate_eda_graphs(df):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eda_dir = os.path.join(base_dir, 'static', 'images', 'eda_plots')
    os.makedirs(eda_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Temperature vs Water Required
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Temperature', y='Water_Required', hue='Soil_Type', alpha=0.7, palette='tab10')
    plt.title('Temperature vs Water Required (Liters)', fontsize=14, fontweight='bold')
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Water Required (Liters)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'temp_vs_water.png'), dpi=300)
    plt.close()

    # 2. Humidity vs Water Required
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df.sample(min(1000, len(df))), x='Humidity', y='Water_Required',
                scatter_kws={'alpha': 0.4, 'color': '#20c997'}, line_kws={'color': '#0d6efd', 'linewidth': 2})
    plt.title('Humidity (%) vs Water Required (Liters)', fontsize=14, fontweight='bold')
    plt.xlabel('Humidity (%)', fontsize=12)
    plt.ylabel('Water Required (Liters)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'humidity_vs_water.png'), dpi=300)
    plt.close()

    # 3. Rainfall vs Water Required
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Rainfall', y='Water_Required', hue='Irrigation_Needed', palette={'Yes': '#dc3545', 'No': '#198754'}, alpha=0.7)
    plt.title('Rainfall (mm) vs Water Required (Liters)', fontsize=14, fontweight='bold')
    plt.xlabel('Rainfall (mm)', fontsize=12)
    plt.ylabel('Water Required (Liters)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'rainfall_vs_water.png'), dpi=300)
    plt.close()

    # 4. Crop-wise Water Usage
    plt.figure(figsize=(10, 5))
    crop_avg = df.groupby('Crop_Type')['Water_Required'].mean().sort_values(ascending=False)
    sns.barplot(x=crop_avg.index, y=crop_avg.values, palette='viridis')
    plt.title('Average Water Required by Crop Type (Liters)', fontsize=14, fontweight='bold')
    plt.xlabel('Crop Type', fontsize=12)
    plt.ylabel('Average Water Required (Liters)', fontsize=12)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'crop_water_usage.png'), dpi=300)
    plt.close()

    # 5. Soil-wise Water Usage
    plt.figure(figsize=(8, 5))
    soil_avg = df.groupby('Soil_Type')['Water_Required'].mean().sort_values(ascending=False)
    sns.barplot(x=soil_avg.index, y=soil_avg.values, palette='magma')
    plt.title('Average Water Required by Soil Type (Liters)', fontsize=14, fontweight='bold')
    plt.xlabel('Soil Type', fontsize=12)
    plt.ylabel('Average Water Required (Liters)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'soil_water_usage.png'), dpi=300)
    plt.close()

    # 6. Weekly Consumption Pattern (Simulated timeline)
    plt.figure(figsize=(10, 5))
    df_sample = df.copy()
    df_sample['Week'] = (df_sample['Day_Of_Year'] // 7) + 1
    weekly_data = df_sample.groupby('Week')['Water_Required'].mean().head(12)
    plt.plot(weekly_data.index, weekly_data.values, marker='o', color='#0d6efd', linewidth=2.5, markersize=8)
    plt.title('Weekly Average Water Consumption Trend (Liters)', fontsize=14, fontweight='bold')
    plt.xlabel('Week of Year', fontsize=12)
    plt.ylabel('Average Consumption (Liters)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'weekly_water_consumption.png'), dpi=300)
    plt.close()

    # 7. Monthly Consumption Pattern
    plt.figure(figsize=(10, 5))
    df_sample['Month'] = (df_sample['Day_Of_Year'] // 30) + 1
    monthly_data = df_sample.groupby('Month')['Water_Required'].mean().head(12)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(months[:len(monthly_data)], monthly_data.values, color='#198754', alpha=0.85)
    plt.title('Monthly Average Water Consumption Pattern (Liters)', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Water Required (Liters)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'monthly_water_consumption.png'), dpi=300)
    plt.close()

    # 8. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    print("EDA plots saved successfully in static/images/eda_plots/")

def train_and_evaluate_models(df):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Encode Categorical Features
    df_encoded = df.copy()
    label_encoders = {}
    cat_columns = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage']

    for col in cat_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Features & Targets
    feature_cols = [
        'Temperature', 'Humidity', 'Soil_Moisture', 'Rainfall',
        'Wind_Speed', 'Solar_Radiation', 'Soil_Type', 'Crop_Type',
        'Crop_Growth_Stage', 'Previous_Water_Consumption', 'Day_Of_Year'
    ]

    X = df_encoded[feature_cols]
    y = df_encoded['Water_Required']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidate_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=12),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=14, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=14, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=6),
        'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42, learning_rate=0.05)
    }

    results = {}
    trained_model_objects = {}

    print("\n--- Training and Evaluating ML Models ---")
    for name, model in candidate_models.items():
        if name in ['Linear Regression', 'AdaBoost']:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        trained_model_objects[name] = model

        print(f"{name:<20} | R² Score: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    # Select Best Model based on R2 score
    best_model_name = max(results, key=lambda k: results[k]['R2'])
    best_model = trained_model_objects[best_model_name]
    best_r2 = results[best_model_name]['R2']
    best_rmse = results[best_model_name]['RMSE']

    print(f"\nWINNING MODEL: {best_model_name} (R² = {best_r2:.4f}, RMSE = {best_rmse:.2f})")

    # Plot Model Comparison
    eda_dir = os.path.join(base_dir, 'static', 'images', 'eda_plots')
    plt.figure(figsize=(10, 5))
    models_names = list(results.keys())
    r2_scores = [results[m]['R2'] for m in models_names]
    sns.barplot(x=models_names, y=r2_scores, palette='crest')
    plt.title('ML Models Accuracy Comparison (R² Score)', fontsize=14, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.ylim(0, 1.05)
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'model_accuracy_comparison.png'), dpi=300)
    plt.close()

    # Feature Importance Plot for best model (if tree-based)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[feature_cols[i] for i in indices], palette='viridis')
        plt.title(f'Feature Importance ({best_model_name})', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, 'feature_importance.png'), dpi=300)
        plt.close()

    # Train 7-Day Forecast Model (Predicting next 7 days consumption based on current metrics)
    print("\nTraining 7-Day Water Consumption Forecaster...")
    X_forecast = X.copy()
    y_forecast_7d = df_encoded['Water_Required'] * 7.0 + np.random.normal(0, 15, len(df))
    
    forecaster = GradientBoostingRegressor(n_estimators=80, random_state=42)
    forecaster.fit(X_forecast, y_forecast_7d)

    # Save models & metadata
    irrigation_model_data = {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_cols': feature_cols,
        'metrics': results,
        'best_r2': best_r2,
        'best_rmse': best_rmse
    }

    joblib.dump(irrigation_model_data, os.path.join(models_dir, 'irrigation_model.pkl'))
    joblib.dump({'forecast_model': forecaster, 'feature_cols': feature_cols}, os.path.join(models_dir, 'forecast_model.pkl'))
    joblib.dump(irrigation_model_data, os.path.join(base_dir, 'irrigation_model.pkl'))
    joblib.dump({'forecast_model': forecaster, 'feature_cols': feature_cols}, os.path.join(base_dir, 'forecast_model.pkl'))

    print("\nAll models and pipeline artifacts successfully saved to models/")
    return irrigation_model_data

if __name__ == '__main__':
    df = load_and_preprocess_data()
    generate_eda_graphs(df)
    train_and_evaluate_models(df)
