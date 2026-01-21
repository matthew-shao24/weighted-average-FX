# %% Import Libraries
import pandas as pd
import os
import numpy as np
from collections import defaultdict
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# %% User Input Area
currency_pairs = [
    ["USD", "RWF"],
    ["RWF", "UGX"],
    ["USD", "NGN"]
]

start_date = "2026-01-01"
end_date = "2026-01-18"

# %% Import from SQL
password = quote_plus("zqL4HmvHY8Mi&bap")
engine = create_engine(
    f"mysql+mysqlconnector://matthew.shao:{password}@mfsa-reversal-master-rds-prod.cvzkhfo7oqvr.us-east-1.rds.amazonaws.com:3306/onafriq_forex_adapter"
)

df = pd.read_sql(
    f"""
    SELECT *
    FROM transaction_details
    WHERE transaction_processed_date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY transaction_processed_date ASC
    """,
    engine
)

# %% Import FX blotter
script_folder = os.path.dirname(os.path.abspath(__file__))
blotter_path = script_folder + "\\data\\fx_blotter_master_2025_v2.xlsx"
fx_blotter = pd.read_excel(blotter_path, sheet_name="curbi_fxrates")

# %% Add column for just date
df['transaction_processed_date'] = pd.to_datetime(df['transaction_processed_date'], errors='coerce')
df['processed_date'] = df['transaction_processed_date'].dt.date

# %% Keeping only relevant transactions
bases = {pair[0] for pair in currency_pairs}
quotes = {pair[1] for pair in currency_pairs}
all_pair_currencies = bases | quotes

df = df[
    df["send_currency"].isin(quotes) |
    df["receive_currency"].isin(quotes)
]

df = df[df['send_currency'] != df['receive_currency']]

# Map base currencies to all their quotes
base_to_quote = defaultdict(set)
for base, quote in currency_pairs:
    base_to_quote[base].add(quote)

# %% Keeping only relevant columns
keep_column = [
    "transaction_processed_date",
    "processed_date",
    "send_amount",
    "receive_amount",
    "send_currency",
    "receive_currency",
    "rate"
]
df = df[keep_column]

# %% Adding on US rate
fx_usd = fx_blotter[['date', 'tocurrency', 'mid']].copy()
df['processed_date'] = pd.to_datetime(df['processed_date'])
fx_usd['date'] = pd.to_datetime(fx_usd['date'])

# adding usd rates to send currencies
fx_usd.rename(columns={
    'date': 'processed_date',
    'tocurrency': 'currency',
    'mid': 'fx_rate'
}, inplace=True)

# Merge send_currency
df = df.merge(
    fx_usd[['processed_date', 'currency', 'fx_rate']],
    how='left',
    left_on=['processed_date', 'send_currency'],
    right_on=['processed_date', 'currency']
)
df.rename(columns={'fx_rate': 's_fx_to_usd'}, inplace=True)
df.drop(columns=['currency'], inplace=True)

# Merge receive_currency
df = df.merge(
    fx_usd[['processed_date', 'currency', 'fx_rate']],
    how='left',
    left_on=['processed_date', 'receive_currency'],
    right_on=['processed_date', 'currency']
)
df.rename(columns={'fx_rate': 'r_fx_to_usd'}, inplace=True)
df.drop(columns=['currency'], inplace=True)

df['s_fx_to_usd'] = df['s_fx_to_usd'].fillna(1)
df['r_fx_to_usd'] = df['r_fx_to_usd'].fillna(1)

# %% Calculate the relevant FX rate for each transaction
def classify_row(row):
    s, r = row["send_currency"], row["receive_currency"]

    # Ignore same-currency transactions
    if s == r:
        return None, None 

    # Case 1: s_fx is base, r_fx is quote
    if s in base_to_quote and r in base_to_quote[s]:
        rate = row["rate"]  # use the original client rate
        pair = f"{s}{r}"

    # Case 2: s_fx is quote, r_fx is base
    elif r in base_to_quote and s in base_to_quote[r]:
        rate = 1 / row["rate"]
        pair = f"{r}{s}"

    # Case 3: s_fx is non-pair, r_fx is quote
    elif r in quotes:
        rate = row["rate"] * row["s_fx_to_usd"]
        pair = f"USD{r}"

    # Case 4: s_fx is quote, r_fx is non-pair
    elif s in quotes:
        rate = (1 / row["rate"]) * row["r_fx_to_usd"]
        pair = f"USD{s}"

    # Ignore anything that doesn't fit
    else:
        return None, None 

    return rate, pair

# Apply row-by-row
df[["standardised_rate", "standard_pair"]] = df.apply(
    lambda row: pd.Series(classify_row(row)),
    axis=1
)

# Drop rows that were ignored (None rates)
df = df.dropna(subset=["standardised_rate"]).copy()

# %% Calculating transaction weighted FX rate for each transaction
pair_cols = df["standard_pair"].dropna().unique().tolist()
pair_data = pd.DataFrame(np.nan, index=df.index, columns=pair_cols)
df["s_amount_usd"] = df["send_amount"] * df["s_fx_to_usd"]

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
daily_weighted_fx.to_csv("output/daily_weighted_fx.csv")

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
total_weighted_fx.to_csv("output/total_weighted_fx.csv")