"""
Geo Incrementality Test — Step 1 & 2: Setup, Load, Clean, Validate
==================================================================
Loads the workbook, cleans all four sheets, validates the panel,
and exports clean DataFrames for downstream analysis.

Data issues found and handled:
  1. State codes: leading/trailing spaces, mixed case (32 raw -> 16 clean)
  2. Exact-duplicate rows: 20 duplicate date+state pairs (all identical values)
  3. GA net_revenue_usd: 8 NaN values (Feb 2-26) + 4 zeros (Apr 13-16 feed incident)
  4. NV negatives: 3 negative paid_social_spend, 2 negative orders
  5. revenue_feed_complete = 0: 4 rows (GA Apr 13-16)
  6. NC;SC in operations calendar needs parsing
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ============================================================
# CONFIGURATION
# ============================================================

XLSX_PATH = r"C:\Users\HP\Downloads\Data.xlsx"
OUTPUT_DIR = Path(r"C:\Users\HP\OneDrive\Desktop\Research\Data\geo_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_STATES = 16
EXPECTED_DATE_RANGE = ("2026-01-12", "2026-05-31")  # 140 days
EXPECTED_DAILY_ROWS = 16 * 140  # 2240

# ============================================================
# STEP 1: LOAD RAW DATA
# ============================================================

print("=" * 70)
print("STEP 1: LOADING RAW DATA")
print("=" * 70)

markets_raw = pd.read_excel(XLSX_PATH, sheet_name="Markets").dropna(how="all")
daily_raw = pd.read_excel(XLSX_PATH, sheet_name="Daily metrics").dropna(how="all")
overlap_raw = pd.read_excel(XLSX_PATH, sheet_name="Audience overlap").dropna(how="all")
calendar_raw = pd.read_excel(XLSX_PATH, sheet_name="Operations calendar").dropna(how="all")

print(f"  Markets:             {len(markets_raw)} rows x {len(markets_raw.columns)} cols")
print(f"  Daily metrics:       {len(daily_raw)} rows x {len(daily_raw.columns)} cols")
print(f"  Audience overlap:    {len(overlap_raw)} rows x {len(overlap_raw.columns)} cols")
print(f"  Operations calendar: {len(calendar_raw)} rows x {len(calendar_raw.columns)} cols")

# ============================================================
# STEP 2: CLEAN DATA
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: CLEANING DATA")
print("=" * 70)

# ----------------------------------------------------------
# 2A. Markets — minimal cleaning needed
# ----------------------------------------------------------

markets = markets_raw.copy()
markets["state"] = markets["state"].str.strip().str.upper()
markets["can_increase_spend"] = markets["can_increase_spend"].astype(int)

# Compute derived columns
markets["incr_spend_28d"] = (
    0.40 * markets["planned_bau_daily_spend_usd"] * 28
)
total_audited_rev = markets["audited_last_28d_revenue_usd"].sum()
markets["revenue_share_pct"] = (
    markets["audited_last_28d_revenue_usd"] / total_audited_rev * 100
)

print(f"\n--- Markets ---")
print(f"  States: {len(markets)} (expected {EXPECTED_STATES})")
print(f"  Total audited 28d revenue: ${total_audited_rev:,.2f}")
print(f"  can_increase_spend=0: {markets[markets.can_increase_spend == 0].state.tolist()}")
print(f"  Delivery groups: {markets.delivery_group.value_counts().to_dict()}")

# ----------------------------------------------------------
# 2B. Daily metrics — the messy one
# ----------------------------------------------------------

daily = daily_raw.copy()

# 2B.1 — Standardize state codes
raw_state_count = daily["state"].nunique()
daily["state"] = daily["state"].str.strip().str.upper()
clean_state_count = daily["state"].nunique()
print(f"\n--- Daily metrics ---")
print(f"  State codes: {raw_state_count} raw -> {clean_state_count} after strip+upper")

# 2B.2 — Ensure date is datetime
daily["date"] = pd.to_datetime(daily["date"])

# 2B.3 — Drop exact duplicates on date + state
n_before = len(daily)
# Keep first of each duplicate group (they have identical values)
daily = daily.drop_duplicates(subset=["date", "state"], keep="first")
n_dupes = n_before - len(daily)
print(f"  Dropped {n_dupes} exact duplicate rows")

# 2B.4 — Verify panel dimensions
n_dates = daily["date"].nunique()
n_states = daily["state"].nunique()
actual_rows = len(daily)
expected_rows = n_dates * n_states
print(f"  Dates: {n_dates} ({daily.date.min().date()} to {daily.date.max().date()})")
print(f"  States: {n_states}")
print(f"  Rows: {actual_rows} (expected {expected_rows}, "
      f"{'MATCH' if actual_rows == expected_rows else 'MISMATCH'})")

if actual_rows != expected_rows:
    # Find missing date-state combos
    full_idx = pd.MultiIndex.from_product(
        [daily["date"].unique(), daily["state"].unique()],
        names=["date", "state"],
    )
    existing_idx = pd.MultiIndex.from_frame(daily[["date", "state"]])
    missing = full_idx.difference(existing_idx)
    print(f"  Missing date-state pairs: {len(missing)}")
    if len(missing) > 0 and len(missing) <= 20:
        for d, s in missing:
            print(f"    {d.date()} - {s}")

# 2B.5 — Handle GA revenue feed incident (Apr 13-16)
# These 4 rows have net_revenue_usd = 0 and revenue_feed_complete = 0
# Set to NaN so they can be imputed later (not zero — zero is misleading)
ga_feed_mask = (
    (daily["state"] == "GA")
    & (daily["date"] >= "2026-04-13")
    & (daily["date"] <= "2026-04-16")
)
ga_feed_count = ga_feed_mask.sum()
daily.loc[ga_feed_mask, "net_revenue_usd"] = np.nan
daily.loc[ga_feed_mask, "orders"] = np.nan
print(f"\n  GA feed incident: set {ga_feed_count} rows (Apr 13-16) net_revenue/orders to NaN")

# 2B.6 — Handle GA NaN revenue (8 scattered NaN rows in Feb)
ga_nan_mask = (daily["state"] == "GA") & (daily["net_revenue_usd"].isna())
ga_nan_dates = daily.loc[ga_nan_mask, "date"].dt.date.tolist()
print(f"  GA NaN revenue: {len(ga_nan_dates)} additional rows "
      f"({ga_nan_dates[0]} to {ga_nan_dates[-1] if ga_nan_dates else 'N/A'})")

# Impute GA NaN rows using linear interpolation within GA's time series
ga_mask = daily["state"] == "GA"
for col in ["net_revenue_usd", "orders"]:
    daily.loc[ga_mask, col] = (
        daily.loc[ga_mask, col]
        .interpolate(method="linear", limit_direction="both")
    )
ga_remaining_nans = daily.loc[ga_mask, "net_revenue_usd"].isna().sum()
print(f"  GA after interpolation: {ga_remaining_nans} NaN remaining")

# 2B.7 — Handle NV negative values
nv_neg_spend = (daily["state"] == "NV") & (daily["paid_social_spend_usd"] < 0)
nv_neg_orders = (daily["state"] == "NV") & (daily["orders"] < 0)
n_neg_spend = nv_neg_spend.sum()
n_neg_orders = nv_neg_orders.sum()
daily.loc[nv_neg_spend, "paid_social_spend_usd"] = np.nan
daily.loc[nv_neg_orders, "orders"] = np.nan
print(f"  NV negatives: {n_neg_spend} spend + {n_neg_orders} orders -> set to NaN")

# Impute NV NaNs
nv_mask = daily["state"] == "NV"
for col in ["paid_social_spend_usd", "orders"]:
    daily.loc[nv_mask, col] = (
        daily.loc[nv_mask, col]
        .interpolate(method="linear", limit_direction="both")
    )

# 2B.8 — Merge markets data onto daily
daily = daily.merge(
    markets[["state", "can_increase_spend", "delivery_group",
             "planned_bau_daily_spend_usd", "audited_last_28d_revenue_usd"]],
    on="state",
    how="left",
)

# ----------------------------------------------------------
# 2C. Audience overlap — clean state codes
# ----------------------------------------------------------

overlap = overlap_raw.copy()
overlap["state_a"] = overlap["state_a"].str.strip().str.upper()
overlap["state_b"] = overlap["state_b"].str.strip().str.upper()
print(f"\n--- Audience overlap ---")
print(f"  Pairs: {len(overlap)}")
for _, row in overlap.iterrows():
    print(f"    {row.state_a} <-> {row.state_b}: "
          f"{row.cross_border_audience_share * 100:.1f}%")

# ----------------------------------------------------------
# 2D. Operations calendar — parse state_scope
# ----------------------------------------------------------

calendar = calendar_raw.copy()
calendar["start_date"] = pd.to_datetime(calendar["start_date"])
calendar["end_date"] = pd.to_datetime(calendar["end_date"])

# Parse state_scope: "NC;SC" -> ["NC", "SC"], "ALL" -> all 16 states
def parse_scope(scope_str, all_states):
    scope_str = str(scope_str).strip()
    if scope_str.upper() == "ALL":
        return list(all_states)
    return [s.strip().upper() for s in scope_str.replace(",", ";").split(";")]

all_state_codes = sorted(markets["state"].unique())
calendar["states_affected"] = calendar["state_scope"].apply(
    lambda x: parse_scope(x, all_state_codes)
)

print(f"\n--- Operations calendar ---")
print(f"  Events: {len(calendar)}")
for _, row in calendar.iterrows():
    scope = row.state_scope
    end = row.end_date.date() if pd.notna(row.end_date) else "ongoing"
    print(f"    {scope:<8} {row.start_date.date()} to {end:<12} {row.event}")

# ============================================================
# VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

# Final checks
assert len(markets) == EXPECTED_STATES, f"Markets has {len(markets)} states, expected {EXPECTED_STATES}"
assert daily["state"].nunique() == EXPECTED_STATES, "Daily doesn't have all 16 states"
assert daily["date"].min() == pd.Timestamp(EXPECTED_DATE_RANGE[0]), "Date range start mismatch"
assert daily["date"].max() == pd.Timestamp(EXPECTED_DATE_RANGE[1]), "Date range end mismatch"

total_nans = daily["net_revenue_usd"].isna().sum()
print(f"  Markets:       {len(markets)} states, ${total_audited_rev:,.0f} total rev")
print(f"  Daily:         {len(daily)} rows, {n_dates} dates x {n_states} states")
print(f"  Remaining NaN in net_revenue_usd: {total_nans}")
print(f"  Remaining NaN in orders: {daily.orders.isna().sum()}")
print(f"  Remaining NaN in paid_social_spend_usd: {daily.paid_social_spend_usd.isna().sum()}")
print(f"  Overlap pairs: {len(overlap)}")
print(f"  Calendar events: {len(calendar)}")

# ============================================================
# SAVE CLEAN DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING CLEAN DATA")
print("=" * 70)

markets.to_csv(OUTPUT_DIR / "markets_clean.csv", index=False)
daily.to_csv(OUTPUT_DIR / "daily_clean.csv", index=False)
overlap.to_csv(OUTPUT_DIR / "overlap_clean.csv", index=False)
calendar.to_csv(OUTPUT_DIR / "calendar_clean.csv", index=False)

print(f"  Saved to: {OUTPUT_DIR}")
for f in OUTPUT_DIR.glob("*.csv"):
    print(f"    {f.name}: {f.stat().st_size:,} bytes")

# ============================================================
# STATE-LEVEL SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("STATE-LEVEL SUMMARY")
print("=" * 70)

state_summary = (
    daily.groupby("state")
    .agg(
        mean_daily_rev=("net_revenue_usd", "mean"),
        std_daily_rev=("net_revenue_usd", "std"),
        mean_daily_spend=("paid_social_spend_usd", "mean"),
        mean_daily_orders=("orders", "mean"),
        n_days=("date", "count"),
        nan_rev_days=("net_revenue_usd", lambda x: x.isna().sum()),
    )
    .reset_index()
)

state_summary = state_summary.merge(
    markets[["state", "can_increase_spend", "delivery_group",
             "planned_bau_daily_spend_usd", "audited_last_28d_revenue_usd",
             "revenue_share_pct", "incr_spend_28d"]],
    on="state",
)

state_summary["cv_revenue"] = (
    state_summary["std_daily_rev"] / state_summary["mean_daily_rev"]
)

state_summary = state_summary.sort_values("audited_last_28d_revenue_usd", ascending=False)

print(f"\n{'State':<6} {'CanIncr':<8} {'DelGroup':<8} {'AuditedRev28d':>14} "
      f"{'RevShare%':>10} {'MeanDlyRev':>12} {'CV':>6} {'IncrSpend28d':>13} {'NaNDays':>8}")
print("-" * 100)
for _, r in state_summary.iterrows():
    print(f"{r.state:<6} {r.can_increase_spend:<8} {r.delivery_group:<8} "
          f"${r.audited_last_28d_revenue_usd:>12,.0f} "
          f"{r.revenue_share_pct:>9.2f}% "
          f"${r.mean_daily_rev:>10,.0f} "
          f"{r.cv_revenue:>5.3f} "
          f"${r.incr_spend_28d:>11,.0f} "
          f"{r.nan_rev_days:>7}")

print(f"\n  TOTAL: ${markets.audited_last_28d_revenue_usd.sum():,.2f}")

print("\n\nSTEP 1 & 2 COMPLETE.")
print(f"Clean data saved to: {OUTPUT_DIR}")
print("Next: Step 3 (EDA with Plotly) and Step 4 (Constraint filtering)")
