import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

# Read the medal table (164 countries x 30 Olympics)
medal_df = pd.read_csv('medal_data.csv', index_col=0)  # Rows: countries, Columns: Years

# Read the event table (30 Olympics x 48 sports)
event_df = pd.read_csv('event_data.csv', index_col=0)  # Rows: Years, Columns: Sports

# Optional: create a fake participant data table for demo
# Shape: (Countries x Years x Sports) — simulate participant numbers
# You should replace this with real participant data
countries = medal_df.index
years = medal_df.columns
sports = event_df.columns

# Fake participant data: random integers between 0 and 100
participant_data = {
    (country, year, sport): np.random.randint(0, 100)
    for country in countries
    for year in years
    for sport in sports
}

# Output rows
output_rows = []

# Loop through each country
for country in countries:
    # Get country's medal vector over 30 Olympics
    country_medals = medal_df.loc[country].values

    # Initialize list for each sport's Spearman correlation
    spearman_scores = {}

    # For each sport, calculate correlation with medal trend
    for sport in sports:
        # Get the sport's event count vector over 30 Olympics
        sport_events = event_df[sport].values

        # Calculate Spearman correlation
        corr, _ = spearmanr(country_medals, sport_events)

        # Save if correlation is valid (not NaN)
        if not np.isnan(corr):
            spearman_scores[sport] = corr

    # Find advantaged sports: correlation > 0.8
    advantaged_sports = [sport for sport, corr in spearman_scores.items() if corr > 0.8]

    # For each year, collect advantaged sports and sum participants
    for year in years:
        year_sports = advantaged_sports
        total_participants = sum(
            participant_data.get((country, year, sport), 0)
            for sport in year_sports
        )
        output_rows.append({
            'Country': country,
            'Year': year,
            'Advantaged_Sports': ', '.join(year_sports),
            'Participants': total_participants
        })

# Create output DataFrame
output_df = pd.DataFrame(output_rows)

# Save to CSV
output_df.to_csv('advantaged_sports_per_country.csv', index=False)
