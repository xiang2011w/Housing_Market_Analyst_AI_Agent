import warnings
# Suppress the tensor copy construction warning from Transformers.
warnings.filterwarnings("ignore", message="To copy construct from a tensor,")

import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import re
import spacy

# Load spaCy's English model.
# (Install with: pip install spacy && python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Global dictionaries for month conversion.
month_name_to_number = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
month_number_to_name = {v: k for k, v in month_name_to_number.items()}

# Additional dictionary for abbreviated month names.
month_abbr = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December"
}

def parse_natural_query(query):
    """
    Extract key details from a natural language query.
    Uses regex to:
      - Extract the zip code from patterns like "zip code 12345", "zip 12345", or "12345 zip".
      - Extract all 4-digit numbers and choose the first one that is not equal to the zip code as the year.
      - Search for a full or abbreviated month name.
    Returns a tuple: (year, month, zip_code)
    """
    # Attempt to extract zip code using several patterns.
    zip_code = None
    # Pattern: "zip code 12345"
    zip_match = re.search(r'zip\s*code\s*(\d+)', query, re.IGNORECASE)
    if zip_match:
        zip_code = int(zip_match.group(1))
    # Pattern: "12345 zip"
    if zip_code is None:
        zip_match = re.search(r'(\d{4,5})\s*zip', query, re.IGNORECASE)
        if zip_match:
            zip_code = int(zip_match.group(1))
    # Pattern: "zip 12345" (without "code")
    if zip_code is None:
        zip_match = re.search(r'zip\s*(\d+)', query, re.IGNORECASE)
        if zip_match:
            zip_code = int(zip_match.group(1))
            
    # Now extract all 4-digit numbers and choose the first one not equal to the zip_code as the year.
    year = None
    numbers = re.findall(r'\b(\d{4})\b', query)
    for num in numbers:
        if zip_code is None or int(num) != zip_code:
            year = int(num)
            break

    # Extract month: check for full month names first, then abbreviations.
    month = None
    for full in month_name_to_number:
        if full.lower() in query.lower():
            month = full
            break
    if month is None:
        for abbr, full in month_abbr.items():
            if abbr in query.lower():
                month = full
                break
                
    return year, month, zip_code

def load_forecast_data(file_path: str) -> pd.DataFrame:
    """Load the forecast data from CSV and ensure key columns are numeric."""
    data = pd.read_csv(file_path)
    data['zip code'] = data['zip code'].astype(int)
    data['month'] = data['month'].astype(int)
    print("Loaded data shape:", data.shape)
    return data

def answer_query(df, query):
    """
    Return the corresponding table value for queries that include one of the keys:
      - "median price per square foot" (or its variants)
      - "median sale price"
      - "new listings"
      - "inventory"
    The query must include a specific zip code and year.
    For queries about the highest sale price, the month is computed automatically.
    """
    year, month, zip_code = parse_natural_query(query)
    lower_query = query.lower()
    
    # If the query asks about the highest sale price, ignore month.
    if "highest sale price" in lower_query or "highest median_sale_price" in lower_query:
        if not year:
            return "Please specify a year in your query."
        if zip_code is None:
            zip_code = 1001  # Default value.
        result = df[(df['year'] == year) & (df['zip code'] == zip_code)]
        if not result.empty:
            best_row = result.loc[result['median_sale_price'].idxmax()]
            month_num = int(best_row['month'])
            mname = month_number_to_name.get(month_num, str(month_num))
            return (f"In {year}, {mname} had the highest median sale price "
                    f"(${best_row['median_sale_price']:.2f}) with an inventory level of {best_row['inventory']:.2f}.")
        else:
            return "No data found for the given query."
    
    # Otherwise, determine which column is being requested.
    column = None
    if any(x in lower_query for x in ["median price per square foot", "median price per sq ft", "median ppsf", "mppsf"]):
        column = "median_ppsf"
    elif "median sale price" in lower_query:
        column = "median_sale_price"
    elif "new listings" in lower_query:
        column = "new_listings"
    elif "inventory" in lower_query:
        column = "inventory"
    
    if column:
        if year and month and zip_code:
            month_num = month_name_to_number.get(month)
            if not month_num:
                return "Month not recognized."
            result = df[(df['year'] == year) & (df['month'] == month_num) & (df['zip code'] == zip_code)]
            if not result.empty:
                value = result.iloc[0][column]
                if column in ["median_ppsf", "median_sale_price"]:
                    return f"The {column.replace('_', ' ')} for zip code {zip_code} in {month} {year} is ${value:.2f}."
                else:
                    return f"The {column.replace('_', ' ')} for zip code {zip_code} in {month} {year} is {value}."
            else:
                return "No data found for the given query."
        else:
            return "Could not extract all required details (year, month, zip code) from your query."
    
    return "Query format not recognized."

def generate_market_report_with_flan(df, year, zip_code, flan_generator):
    """
    Generate a professional market analysis report using FLAN-T5.
    This function computes key metrics from the data and builds a prompt for FLAN-T5
    to generate a report. The prompt now includes several example reports to guide the generation.
    """
    # Filter the data by year and zip code.
    if zip_code:
        data = df[(df['year'] == year) & (df['zip code'] == zip_code)]
        if data.empty:
            return f"No data available for zip code {zip_code} in {year}."
    else:
        data = df[df['year'] == year]
        if data.empty:
            return f"No data available for {year}."
    
    # Compute key metrics.
    avg_ppsf = data['median_ppsf'].mean()
    avg_sale_price = data['median_sale_price'].mean()
    avg_new_listings = data['new_listings'].mean()
    avg_inventory = data['inventory'].mean()
    
    lowest_inventory_row = data.loc[data['inventory'].idxmin()]
    highest_sale_price_row = data.loc[data['median_sale_price'].idxmax()]
    
    lowest_inventory_month = month_number_to_name.get(int(lowest_inventory_row['month']))
    highest_sale_price_month = month_number_to_name.get(int(highest_sale_price_row['month']))
    
    # Build a detailed data summary.
    summary = (
        f"For {year} in zip code {zip_code}: Average price per square foot is ${avg_ppsf:.2f}, "
        f"median sale price is ${avg_sale_price:.2f}, average new listings per month are {avg_new_listings:.2f} units, "
        f"and average inventory is {avg_inventory:.2f} units. The lowest inventory of {lowest_inventory_row['inventory']} "
        f"units occurred in {lowest_inventory_month}, and the highest median sale price of ${highest_sale_price_row['median_sale_price']:.2f} "
        f"was recorded in {highest_sale_price_month}."
    )
    
    # Include several detailed example reports in the prompt.
    examples = (
        "Example 1:\n"
        "For 2025 in zip code 1001, the housing market has remained robust with an average price per square foot of approximately $245.00 "
        "and a median sale price near $360,000. Inventory was at its lowest in December, indicating strong buyer demand. Over the next two months, "
        "we expect new listings to gradually increase as sellers respond to seasonal market shifts. This influx may ease the tight supply, leading "
        "to more balanced negotiations. Buyers may benefit from improved affordability, while sellers should prepare for a slight moderation in price growth.\n\n"
        "Example 2:\n"
        "In zip code 1001 during 2025, the market has shown stability with an average price per square foot of about $243.50 and a median sale price around $350,000. "
        "Summer months witnessed a spike in sale prices driven by low inventory. Looking ahead, analysts predict that the next couple of months will see a moderate "
        "increase in available homes as seasonal trends reverse. This anticipated rise in inventory could help stabilize prices further, offering potential opportunities for buyers "
        "to negotiate and for sellers to recalibrate their expectations.\n\n"
        "Example 3:\n"
        "The 2025 housing market in zip code 1001 continues to be competitive. Current data indicates an average price per square foot of roughly $250.00 and a median sale price "
        "of approximately $370,000. With inventory dipping during the late fall, the market has favored sellers. However, early indicators suggest that the coming months will bring "
        "an uptick in new listings as market activity normalizes. This anticipated increase in supply is expected to relieve pricing pressure slightly, creating a more balanced market environment."
    )
    
    # Construct the enhanced prompt.
    prompt = (
        f"You are a highly experienced real estate market analyst. Based on the following data summary and examples, "
        f"generate a detailed, professional, and insightful market analysis report for {year} in zip code {zip_code}. "
        f"Your report should analyze current trends, provide predictions for the next couple of months, and offer actionable recommendations for buyers and sellers. "
        f"Use a natural and narrative tone.\n\n"
        f"Data summary:\n{summary}\n\n"
        f"Examples:\n{examples}\n\n"
        f"Now generate the market analysis report."
    )
    
    # Generate the report using FLAN-T5.
    flan_output = flan_generator(prompt, max_length=512, do_sample=True, temperature=0.7)[0]['generated_text']
    return flan_output

def query_model(question: str, data: pd.DataFrame, tokenizer, model):
    """
    Query the TAPAS model with a natural language question using the provided table.
    (This is used as a fallback if the custom parsing does not match the query schema.)
    """
    table = data.astype(str)
    inputs = tokenizer(table=table, queries=[question], padding="max_length", return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    outputs = model(**inputs)
    predicted_answer_coordinates, _ = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(), outputs.logits_aggregation.detach()
    )
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            row, col = coordinates[0]
            answers.append(table.iat[row, col])
        elif len(coordinates) > 1:
            answers.append(", ".join([table.iat[row, col] for row, col in coordinates]))
        else:
            answers.append("No answer found")
    return answers

def main():
    forecast_data = load_forecast_data('processed_data/forecast_metrics_with_zip.csv')
    
    # Load TAPAS for question answering.
    tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
    tapas_model = TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")
    
    # Load the fine-tuned FLAN-T5 model for report generation.
    flan_tokenizer = AutoTokenizer.from_pretrained("./flan_t5_finetuned_market")
    flan_model = AutoModelForSeq2SeqLM.from_pretrained("./flan_t5_finetuned_market")
    flan_generator = pipeline("text2text-generation", model=flan_model, tokenizer=flan_tokenizer)
    
    print("Welcome to the Housing Market Analyst AI agent System!")
    print("\nExample queries you can try:")
    print("  - What was the median price per square foot in zip code 1001 for February 2025?")
    print("  - Which month in 2025 had the highest sale price and lowest inventory in zip code 1001?")
    print("  - Generate a market report for 2025 in zip code 1001")
    
    while True:
        print("\nEnter your query (or type 'exit' to quit):")
        question = input("Query: ").strip()
        if question.lower() == 'exit':
            break
        
        if "report" in question.lower():
            year, _, zip_code = parse_natural_query(question)
            if year:
                report = generate_market_report_with_flan(forecast_data, year, zip_code, flan_generator)
                print("\nMarket Analysis Report:")
                print(report)
            else:
                print("Could not extract a year from your query. Please include a year (e.g., 2025).")
        else:
            response = answer_query(forecast_data, question)
            if response == "Query format not recognized.":
                answers = query_model(question, forecast_data, tapas_tokenizer, tapas_model)
                print("Answer:", answers)
            else:
                print("Answer:", response)

if __name__ == "__main__":
    main()
