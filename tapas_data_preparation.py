# This file loads forecast_metrics_with_zip.csv after ML model Prophet generated forecast data. 
# This file then prepares it as prompt text for potential question answering tasks. 
# It also demonstrates how to load and initialize the TAPAS model and tokenizer from the Hugging Face library.
import pandas as pd
# Import the TapasTokenizer and TapasForQuestionAnswering classes from the Hugging Face Transformers library
from transformers import TapasTokenizer, TapasForQuestionAnswering

def load_forecast_data(file_path: str) -> pd.DataFrame:
    """Load the forecast data."""
    return pd.read_csv(file_path)

def prepare_training_data(data: pd.DataFrame) -> list:
    """Prepare the data for training."""
    training_data = []
    for _, row in data.iterrows():
        prompt = f"Forecast for zip code {row['zip code']} in {row['year']}-{row['month']}: "
        prompt += f"Median PPSF: {row['median_ppsf']}, Median Sale Price: {row['median_sale_price']}, "
        prompt += f"New Listings: {row['new_listings']}, Inventory: {row['inventory']}."
        training_data.append(prompt)
    return training_data

def main():
    # Load and prepare the forecast data
    forecast_data = load_forecast_data('processed_data/forecast_metrics_with_zip.csv')
    training_data = prepare_training_data(forecast_data)

    # Initialize the tokenizer from the pre-trained TAPAS model
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
    # Initialize the model from the pre-trained TAPAS model
    model = TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")

if __name__ == "__main__":
    main() 