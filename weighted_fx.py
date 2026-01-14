# %% Import Libraries
import pandas as pd
import os
import numpy as np
from collections import defaultdict

# %% User Input Area
currency_pairs = [
    ["USD", "RWF"],
    ["RWF", "UGX"],
    ["USD", "NGN"]
]

start_date = "2026-01-01"
end_date = "2026-01-07"

# %% Set path to data
script_folder = os.path.dirname(os.path.abspath(__file__))
path = script_folder + "\\data"
dates = pd.date_range(start=start_date, end=end_date)
dfs = []

# %% Import and Clean Data
for d in dates:
    date_str = d.strftime("%Y-%m-%d")
    file_path = f"{path}/DSR_ONAFRIQ_{date_str}.csv" 
    df_daily = pd.read_csv(file_path)
    dfs.append(df_daily)

# Stack them on top of each other
df = pd.concat(dfs, ignore_index=True)

# %% Add column for just date
df['dateprocessed'] = pd.to_datetime(df['dateprocessed'], errors='coerce')
df['processed_date'] = df['dateprocessed'].dt.date
cols = list(df.columns)
cols.insert(2, cols.pop(cols.index('processed_date')))
df = df[cols]  

# Filter for successful transactions only
# And ignores non cross-border transactions
df = df[df['status'] == 'Success']
# df = df[df["s_fx"] != df["r_fx"]]

# %% Keeping only relevant transactions
bases = {pair[0] for pair in currency_pairs}
quotes = {pair[1] for pair in currency_pairs}
all_pair_currencies = bases | quotes

df = df[
    df["s_fx"].isin(quotes) |
    df["r_fx"].isin(quotes)
]

# Map base currencies to all their quotes
base_to_quote = defaultdict(set)
for base, quote in currency_pairs:
    base_to_quote[base].add(quote)

# %% Keeping only relevant columns
keep_column = [
    "datecreated",
    "dateprocessed",
    "processed_date",
    "status",
    "s_fx",
    "s_amount_s_fx",
    "r_fx",
    "r_amount_r_fx",
    "client_fx_rate",
    "s_fx_to_usd",
    "r_fx_to_usd",
    "s_amount_usd",
    "r_amount_usd"
]
df = df[keep_column]

#%% Calculate the relevant FX rate for each transaction
def classify_row(row):
    s, r = row["s_fx"], row["r_fx"]
    # ignore same-currency
    if s == r:
        return None, None 

    # Case 1: s_fx is base, r_fx is quote
    if s in base_to_quote and r in base_to_quote[s]:
        rate = row["client_fx_rate"]
        pair = f"{s}{r}"
        
    # Case 2: s_fx is quote, r_fx is base
    elif r in base_to_quote and s in base_to_quote[r]:
        rate = 1 / row["client_fx_rate"]
        pair = f"{r}{s}"
        
    # Case 3: s_fx is non-pair, r_fx is quote
    elif r in quotes:
        rate = row["client_fx_rate"] * row["s_fx_to_usd"]
        pair = f"USD{r}"
        
    # Case 4: s_fx is quote, r_fx is non-pair
    elif s in quotes:
        rate = (1 / row["client_fx_rate"]) * row["r_fx_to_usd"]
        pair = f"USD{s}"
    
    # ignore anything that doesn't fit
    else:
        return None, None 

    return rate, pair

df[["standardised_rate", "standard_pair"]] = df.apply(
    lambda row: pd.Series(classify_row(row)),
    axis=1
)

# Drop rows that were ignored
df = df.dropna(subset=["standardised_rate"]).copy()

# %% Calculating transaction weighted FX rate for each transaction
pair_cols = df["standard_pair"].dropna().unique().tolist()
pair_data = pd.DataFrame(np.nan, index=df.index, columns=pair_cols)

for col in pair_cols:
    mask = df["standard_pair"] == col
    pair_data.loc[mask, col] = df.loc[mask, "standardised_rate"] * df.loc[mask, "s_amount_usd"]

df = pd.concat([df, pair_data], axis=1)

# %% Aggregating Transaction weighted FX rates
# Daily transaction weighted FX
daily_trx_fx = (
    df
    .groupby("processed_date")[pair_cols]
    .sum(min_count=1)
    .reset_index()
)

# Total transaction weighted FX
total_trx_fx = (
    df[pair_cols]
    .sum(min_count=1)
)

# Converting daily FX to long format
daily_trx_fx = daily_trx_fx.melt(
    id_vars="processed_date",
    var_name="standard_pair",
    value_name="trx_fx"
)

# %% Calculating Total and Daily Transaction Volume (by sent USD amount)
daily_trx_vol = (
    df
    .groupby(["processed_date", "standard_pair"], as_index=False)
    .agg({"s_amount_usd": "sum"})
)

total_trx_vol = (
    df
    .groupby(["standard_pair"], as_index=False)
    .agg({"s_amount_usd": "sum"})
)

# %% Calculating daily weighted-average FX rate
daily_weighted_fx = pd.merge(
    daily_trx_fx,
    daily_trx_vol,
    on=['processed_date', 'standard_pair'],
    how='left',
)

daily_weighted_fx['weighted_rate'] = daily_weighted_fx['trx_fx'] / daily_weighted_fx['s_amount_usd']

# %% Calculating total weighted-average FX rate
# Convert total transaction FX to DataFrame
total_trx_fx = total_trx_fx.reset_index()
total_trx_fx.columns = ["standard_pair", "trx_fx"]  # rename columns

# Merge with total_trx_vol to get total USD per pair
total_weighted_fx = pd.merge(
    total_trx_fx,
    total_trx_vol,
    on="standard_pair",
    how="left"
)

# Calculate weighted FX
total_weighted_fx["weighted_rate"] = total_weighted_fx["trx_fx"] / total_weighted_fx["s_amount_usd"]
