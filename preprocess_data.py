import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load the TSV file into a DataFrame."""
    data = pd.read_csv(file_path, sep='\t')
    print("Columns in the dataset:", data.columns)  # Print column names
    return data


def rename_and_format_zip_code(data: pd.DataFrame) -> pd.DataFrame:
    """Rename 'region' column to 'zip code' and format the values to keep only digits."""
    if 'region' in data.columns:
        # Rename the column
        data.rename(columns={'region': 'zip code'}, inplace=True)
        
        # Format the values to keep only digits
        data['zip code'] = data['zip code'].str.extract(r'(\d+)', expand=False)
    
    return data

def convert_to_datetime(data: pd.DataFrame) -> pd.DataFrame:
    """Convert period_begin and period_end columns to datetime format."""
    data['period_begin'] = pd.to_datetime(data['period_begin'], errors='coerce')
    data['period_end'] = pd.to_datetime(data['period_end'], errors='coerce')
    return data

def derive_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derive year and quarter from period_begin."""
    data['year'] = data['period_begin'].dt.year
    data['quarter'] = data['period_begin'].dt.quarter
    return data

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with the most recent available data based on 'period_begin'."""
    # Sort data by 'period_begin' to ensure chronological order
    data.sort_values(by='period_begin', inplace=True)
    
    # Fill missing values with the most recent available data
    data.fillna(method='ffill', inplace=True)
    
    return data

def aggregate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Group data by zip_code and aggregate relevant metrics."""
    # Select relevant columns
    columns_to_keep = [
        'zip code', 'year', 'quarter', 'state', 'state_code', 'property_type',
        'median_sale_price', 'median_list_price', 'median_ppsf', 'median_list_ppsf',
        'homes_sold', 'pending_sales', 'new_listings', 'inventory'
    ]
    
    # Ensure all necessary columns are present
    data = data[columns_to_keep]
    
    # Group and aggregate data
    aggregated_data = data.groupby(['zip code', 'year', 'quarter', 'state', 'state_code', 'property_type']).agg({
        'median_sale_price': 'mean',
        'median_list_price': 'mean',
        'median_ppsf': 'mean',
        'median_list_ppsf': 'mean',
        'homes_sold': 'sum',
        'pending_sales': 'sum',
        'new_listings': 'sum',
        'inventory': 'mean'
    }).reset_index()
    
    return aggregated_data

def main():
    # Define file paths
    raw_data_folder = 'raw data'
    processed_data_folder = 'processed_data'
    input_file_path = os.path.join(raw_data_folder, 'zip_code_market_tracker.tsv')
    cleaned_data_path = os.path.join(processed_data_folder, 'cleaned_data.tsv')

    # Ensure the processed data directory exists
    os.makedirs(processed_data_folder, exist_ok=True)

    # Load the data
    data = load_data(input_file_path)
    print("loading data complete")

    # Rename and format the 'region' column
    data = rename_and_format_zip_code(data)
    print("rename and format zip code complete")

    # Convert period columns to datetime
    data = convert_to_datetime(data)
    print("convert to datetime complete")

    # Derive time features
    data = derive_time_features(data)
    print("derive time features complete")

    # Fill missing values
    data = fill_missing_values(data)
    print("fill missing values complete")

    # Aggregate metrics
    data = aggregate_metrics(data)
    print("aggregate metrics complete")

    # Save the cleaned data
    data.to_csv(cleaned_data_path, sep='\t', index=False)
    print(f"Cleaned data saved to: {cleaned_data_path}")

    print("Data processing complete. Cleaned data saved.")

if __name__ == "__main__":
    main() 