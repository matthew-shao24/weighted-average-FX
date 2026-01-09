# %% Import Libraries
import pandas as pd
import os

# %% Set path to data
current_folder = os.getcwd()
path = current_folder + "\\data"
# path = "C:/Users/MatthewShao/OneDrive - Onafriq Limited/Documents/Transaction Engine/data"
dates = pd.date_range(start="2026-01-01", end="2026-01-07")
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
df["processed_date"] = df["dateprocessed"].dt.date
cols = list(df.columns)
cols.insert(2, cols.pop(cols.index('processed_date')))
df = df[cols]  

# %% Filter for successful transactions only
df = df[df['status'] == 'Success']

# %% Keeping only relevant transaction
keep_pairs = {
    frozenset(["USD", "RWF"]),
    frozenset(["RWF", "UGX"]),
    frozenset(["USD", "NGN"]),
}

pairs = df[["s_fx", "r_fx"]].apply(frozenset, axis=1)
df = df[pairs.isin(keep_pairs)]

# %% Calculating Weighted Average FX Rate
# simplifying by keeping only relevant columns
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
df['implied_rate'] = df['r_amount_r_fx'] / df['s_amount_s_fx']
cols = list(df.columns)
cols.insert(9, cols.pop(cols.index('implied_rate')))
df = df[cols]

df.to_csv("data\\test.csv", index=False)

# %% Weighted Average for standard pairs
directional_list = [tuple(pair) for pair in keep_pairs]

standard_df = df[df[["s_fx", "r_fx"]].apply(tuple, axis=1).isin(directional_list)]

standard_df['received_volume_act'] = standard_df['s_amount_s_fx'] * standard_df['client_fx_rate']
standard_df['received_volume_imp'] = standard_df['s_amount_s_fx'] * standard_df['implied_rate']

#Daily transaction volumes for each pair
daily_trans = (
    standard_df
    .groupby(["processed_date", "s_fx", "r_fx"], as_index=False)
    .agg({"s_amount_s_fx": "sum"})
)
#Total transaction volumes for each pair
total_trans = (
    standard_df
    .groupby(["s_fx", "r_fx"], as_index=False)
    .agg({"s_amount_s_fx": "sum"})
)

# %% Weighted Average for inverse pairs
inverse_df = df[~df[["s_fx", "r_fx"]].apply(tuple, axis=1).isin(directional_list)]

inverse_df['received_volume_act'] = inverse_df['s_amount_s_fx'] / inverse_df['client_fx_rate']
inverse_df['received_volume_imp'] = inverse_df['s_amount_s_fx'] /inverse_df['implied_rate']

#Daily transaction volumes for each pair
daily_inverse_trans = (
    inverse_df
    .groupby(["processed_date", "s_fx", "r_fx"], as_index=False)
    .agg({"s_amount_s_fx": "sum"})
)
#Total transaction volumes for each pair
total_inverse_trans = (
    inverse_df
    .groupby(["s_fx", "r_fx"], as_index=False)
    .agg({"s_amount_s_fx": "sum"})
)

# %% merging the data sets
combined_df = pd.concat([standard_df, inverse_df], ignore_index=True)

daily_volume = (
    combined_df
    .groupby(["processed_date", "s_fx", "r_fx"], as_index=False)
    .agg({"received_volume_act": "sum"})
)

combined_df['pair'] = combined_df.apply(lambda x: frozenset([x['s_fx'], x['r_fx']]), axis=1)

daily_corridor_sum = (
    combined_df
    .groupby(['processed_date', 'pair'], as_index=False)
    .agg({'received_volume_act': 'sum'})
)

daily_corridor_sum['pair'] = daily_corridor_sum['pair'].apply(
    lambda x: next((p for p in directional_list if set(p) == set(x)), x)
)

daily_corridor_sum['processed_date'] = pd.to_datetime(daily_corridor_sum['processed_date']).dt.date
# %% Combining the standard and inverse transaction-weighted FX rates

# Standard trades
daily_trans['pair'] = daily_trans.apply(
    lambda x: tuple([x['s_fx'], x['r_fx']]), axis=1
)

# Inverse trades
daily_inverse_trans['pair'] = daily_inverse_trans.apply(
    lambda x: tuple([x['s_fx'], x['r_fx']]), axis=1
)

# Standard trades
daily_standard_sum = (
    daily_trans
    .groupby(['processed_date', 'pair'], as_index=False)
    .agg({'s_amount_s_fx': 'sum'})
)

# Inverse trades
daily_inverse_sum = (
    daily_inverse_trans
    .groupby(['processed_date', 'pair'], as_index=False)
    .agg({'s_amount_s_fx': 'sum'})
)

# %% Treat bi-directional trade as the same
combined_daily = pd.concat([daily_standard_sum, daily_inverse_sum], ignore_index=True)

combined_daily['pair'] = combined_daily['pair'].apply(
    lambda x: next((p for p in directional_list if set(p) == set(x)), x)
)

total_daily_pair = (
    combined_daily
    .groupby(['processed_date', 'pair'], as_index=False)
    .agg({'s_amount_s_fx': 'sum'})
)
total_daily_pair['processed_date'] = pd.to_datetime(total_daily_pair['processed_date']).dt.date

# %% Calculating Daily Weighted FX Rate
daily_weighted_fx = pd.merge(
    total_daily_pair,
    daily_corridor_sum,
    on=['processed_date', 'pair'],
    how='left',  # keeps all rows from total_daily_pair
)

daily_weighted_fx['weighted_rate'] = daily_weighted_fx['received_volume_act'] / daily_weighted_fx['s_amount_s_fx']

# %% Whole Period Weighted FX
total_corridor_sum = (
    daily_corridor_sum
    .groupby(['pair'], as_index=False)
    .agg({'received_volume_act': 'sum'})
)

total_pair = (
    total_daily_pair
    .groupby(['pair'], as_index=False)
    .agg({'s_amount_s_fx': 'sum'})
)

total_weighted_fx = pd.merge(
    total_pair,
    total_corridor_sum,
    on=['pair'],
    how='left',  # keeps all rows from total_daily_pair
)

total_weighted_fx['weighted_rate'] = total_weighted_fx['received_volume_act'] / total_weighted_fx['s_amount_s_fx']

# %% Print Daily FX Rate
daily_weighted_fx
daily_weighted_fx.to_csv("output/daily_weighted_fx.csv")

# %% Print Total FX Rate
total_weighted_fx
total_weighted_fx.to_csv("output/period_weighted_fx.csv")

# %%
