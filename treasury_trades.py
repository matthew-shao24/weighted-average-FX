#%% Import libraries
from simple_salesforce import Salesforce
import salesforce_reporting
import pandas as pd
import numpy as np
from datetime import date

# %% Enter Password
SF_USERNAME = 'dcousins@mfsafrica.com'
SF_PASSWORD = 'Bryant08@@@@!!!!!'
SF_SECURITY_TOKEN = 'vAPGUohRVVDXICwdAwR4IT0by'
SF_REPORT_ID = '00OPz00000ROC0TMAX'

# %% Defining Helper functions
def _to_num(s):
    if isinstance(s, pd.Series):
        return pd.to_numeric(s.str.replace(',', '', regex=True), errors='coerce')
    else:
        try:
            return float(str(s).replace(',', ''))
        except:
            return np.nan
    
FINAL_COLS = [
    "date", "currency1", "currency2", "amount1", "amount2",
    "fx_rate", "corridor", "currency_pair"
]

# %%
def fetch_treasury_trades_for_date(target_date: date) -> pd.DataFrame:
    """Fetch and normalise Treasury FX trades for a single value date."""
    print(f"[TREASURY] Fetching Salesforce report for {target_date}...")

    # Authenticate + fetch report
    sf = Salesforce(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        security_token=SF_SECURITY_TOKEN,
    )
    report = sf.restful(f'analytics/reports/{SF_REPORT_ID}')

    parser = salesforce_reporting.ReportParser(report)
    rows = list(parser.records())
    detail_cols = report['reportMetadata']['detailColumns']
    labels = [
        report['reportExtendedMetadata']['detailColumnInfo'][c]['label']
        for c in detail_cols
    ]
    df = pd.DataFrame(rows, columns=labels)

    # Remove duplicate trades by trade name (if column exists)
    if "FxTrade: Trade Name" in df.columns:
        df = df.drop_duplicates(subset=["FxTrade: Trade Name"])

    # Helper to find columns by suffix
    def col(suffix: str) -> str:
        return next(
            c for c in df.columns
            if c.strip().endswith(suffix)
        )

    # Resolve needed columns
    c_valdate = col("ValueDate")
    c_ccy1    = col("Currency1")
    c_ccy2    = col("Currency2")
    c_amt1    = col("Amount1")
    c_amt2    = col("Amount2")
    c_rate    = col("Rate")
    c_cpty    = col("Counterparty")

    out = pd.DataFrame({
        "date":      pd.to_datetime(df[c_valdate], errors="coerce", dayfirst=True),
        "currency1": df[c_ccy1].astype(str).str.strip(),
        "currency2": df[c_ccy2].astype(str).str.strip(),
        "amount1":   _to_num(df[c_amt1]),
        "amount2":   _to_num(df[c_amt2]),
        "fx_rate":   _to_num(df[c_rate]),
        "corridor": (
            df[c_cpty]
            .astype(str)
            .str.strip()
            .str.replace(r"[_\s]+$", "", regex=True)
            + "_treasury"
        ),
    })

    # Filter to target date
    out = out[out["date"].dt.date == target_date].copy()
    if out.empty:
        print(f"[TREASURY] No trades found for {target_date}.")
        return pd.DataFrame(columns=FINAL_COLS)

    # Put sold currency on RHS; invert rate when flipped
    flip = out["amount1"] < 0

    out = out.assign(
        currency1=np.where(flip, out["currency2"], out["currency1"]),
        amount1=np.where(flip, out["amount2"], out["amount1"]),
        currency2=np.where(flip, out["currency1"], out["currency2"]),
        amount2=np.where(flip, out["amount1"], out["amount2"]),
        fx_rate=np.where(
            flip,
            np.where(out["fx_rate"] != 0, 1.0 / out["fx_rate"], np.nan),
            out["fx_rate"],
        ),
    )

    # Currency pair & final column order
    out["currency_pair"] = out["currency1"] + out["currency2"]

    # Ensure just the columns we care about
    out = out[FINAL_COLS]

    # Normalise date to pure date objects
    out["date"] = pd.to_datetime(out["date"]).dt.date

    print(f"[TREASURY] {len(out)} rows for {target_date}.")
    return out
# %% Pulling the data
target_date = date(2026, 1, 27)
trades_df = fetch_treasury_trades_for_date(target_date)
trades_df.head()
# %%
