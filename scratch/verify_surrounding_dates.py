import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(project_root, "data/processed/curated_historical_sentiment.csv")

def main():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    coal_df = df[df['Ticker'] == 'COALINDIA.NS']
    
    # Check surrounding dates for the sample dates
    test_dates = [
        ('2024-06-02', '2024-06-03', '2024-06-04'),
        ('2024-08-11', '2024-08-12', '2024-08-13'),
        ('2024-11-19', '2024-11-20', '2024-11-21'),
        ('2024-12-25', '2024-12-26', '2024-12-27'),
        ('2025-02-20', '2025-02-21', '2025-02-22'),
    ]
    
    print("Surrounding dates check in curated_historical_sentiment.csv:")
    for prev_d, curr_d, next_d in test_dates:
        row_prev = coal_df[coal_df['Date'] == prev_d]
        row_curr = coal_df[coal_df['Date'] == curr_d]
        row_next = coal_df[coal_df['Date'] == next_d]
        
        avail_prev = row_prev['Sentiment_Available'].values[0] if not row_prev.empty else 'N/A'
        avail_curr = row_curr['Sentiment_Available'].values[0] if not row_curr.empty else 'N/A'
        avail_next = row_next['Sentiment_Available'].values[0] if not row_next.empty else 'N/A'
        
        print(f"Date: {curr_d} (Available: {avail_curr}) | Prev Date {prev_d} (Available: {avail_prev}) | Next Date {next_d} (Available: {avail_next})")

if __name__ == '__main__':
    main()
