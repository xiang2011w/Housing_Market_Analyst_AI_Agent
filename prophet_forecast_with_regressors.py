import pandas as pd
from prophet import Prophet
import os
from datetime import datetime
from typing import Tuple

def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Load the cleaned data."""
    # Load the data with the correct separator
    data = pd.read_csv(file_path, sep='\t')
    
    # Debug: Print the columns to verify
    print("Columns in the dataset:", data.columns.tolist())
    
    # Ensure the zip code column is present
    if 'zip code' not in data.columns:
        raise ValueError("Zip code column is missing from the dataset.")
    
    return data

def prepare_data_for_prophet(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Prophet model."""
    # Create a date column from year and quarter
    data['ds'] = pd.to_datetime(data['year'].astype(str) + '-' + (data['quarter'] * 3 - 2).astype(str) + '-01')
    
    # Drop unnecessary columns, but keep zip_code
    data = data.drop(columns=['year', 'quarter'])
    
    # Check for NaN values and drop them
    data.dropna(subset=['ds'], inplace=True)
    
    return data

def split_data(data: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and test sets based on a split date."""
    train_data = data[data['ds'] < split_date]
    test_data = data[data['ds'] >= split_date]
    return train_data, test_data

def forecast_with_zip_code(data):
    """Generate forecasts for each zip code."""
    forecasts = []
    for zip_code, group in data.groupby('zip code'):
        # Check if there's enough data to train the model
        if group['ds'].nunique() < 2:  # Ensure there's more than one unique date
            print(f"Not enough data for zip code {zip_code}. Skipping.")
            continue
        
        # Forecast for each metric
        forecast_dict = {'ds': [], 'zip code': [], 'median_ppsf': [], 'median_sale_price': [], 'new_listings': [], 'inventory': []}
        for metric in ['median_ppsf', 'median_sale_price', 'new_listings', 'inventory']:
            model = Prophet()
            group['y'] = group[metric]
            model.fit(group[['ds', 'y']])
            
            # Create future DataFrame for the desired period
            future = model.make_future_dataframe(periods=13, freq='M')  # Predicting from Oct 2024 to Oct 2025
            forecast = model.predict(future)
            
            # Store only future predictions
            future_forecast = forecast[(forecast['ds'] > group['ds'].max()) & (forecast['ds'] <= '2025-10-31')]
            forecast_dict['ds'] = future_forecast['ds']
            forecast_dict['zip code'] = zip_code
            forecast_dict[metric] = future_forecast['yhat'].clip(lower=0).round(2)
        
        forecast_df = pd.DataFrame(forecast_dict)
        
        # Streamline median_ppsf variation
        if 'median_ppsf' in forecast_df.columns:
            forecast_df['median_ppsf'] = forecast_df['median_ppsf'].rolling(window=2).apply(
                lambda x: x[0] if abs(x[1] - x[0]) / x[0] > 0.05 else x[1], raw=True
            ).fillna(method='bfill')
        
        # Replace 0 values with forward or backward fill
        for col in ['median_ppsf', 'median_sale_price']:
            forecast_df[col] = forecast_df[col].replace(0, method='ffill').replace(0, method='bfill')
        
        forecasts.append(forecast_df)
    
    if forecasts:
        full_forecast = pd.concat(forecasts)
        full_forecast['year'] = full_forecast['ds'].dt.year
        full_forecast['month'] = full_forecast['ds'].dt.month
        full_forecast = full_forecast[['year', 'month', 'zip code', 'median_ppsf', 'median_sale_price', 'new_listings', 'inventory']]
        full_forecast.to_csv('processed_data/forecast_metrics_with_zip.csv', index=False)
    else:
        print("No forecasts generated due to insufficient data.")

def main():
    # Define file paths
    processed_data_folder = 'processed_data'
    # Use the smaller dataset for Massachusetts
    cleaned_data_path = os.path.join(processed_data_folder, 'cleaned_ma_data.tsv')
    
    # Load the cleaned data
    data = load_cleaned_data(cleaned_data_path)
    
    # Prepare data for Prophet
    prophet_data = prepare_data_for_prophet(data)
    
    # Split the data into training and test sets
    split_date = '2024-01-01'  # Example split date
    train_data, test_data = split_data(prophet_data, split_date)
    
    # Generate forecasts by zip code
    forecast_with_zip_code(prophet_data)

if __name__ == "__main__":
    main() 