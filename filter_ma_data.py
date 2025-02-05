import pandas as pd
import os

def filter_state_data(input_file_path: str, output_file_path: str, state_code: str = 'MA'):
    """Filter data for a specific state and save to a new file."""
    # Load the data
    data = pd.read_csv(input_file_path, sep='\t')
    
    # Filter for the specified state code
    ma_data = data[data['state_code'] == state_code]
    
    # Save the filtered data
    ma_data.to_csv(output_file_path, sep='\t', index=False)
    print(f"Filtered data saved to: {output_file_path}")

def main():
    # Define file paths
    processed_data_folder = 'processed_data'
    input_file_path = os.path.join(processed_data_folder, 'cleaned_data.tsv')
    output_file_path = os.path.join(processed_data_folder, 'cleaned_ma_data.tsv')
    
    # Filter and save data for Massachusetts
    filter_state_data(input_file_path, output_file_path)

if __name__ == "__main__":
    main() 