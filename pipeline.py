# =====================================================================
# GEO INCREMENTALITY TEST — COMPLETE PIPELINE (corrected)
# =====================================================================
# Every change from the original is marked with a "FIX n" comment.
# =====================================================================

# =====================================================================
# IMPORTS
# =====================================================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from pathlib import Path
import datetime
import json
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================
XLSX_PATH  = r"C:\Users\HP\Downloads\Data.xlsx"
OUTPUT_DIR = Path(r"C:\Users\HP\OneDrive\Desktop\Research\Data\geo_experiment")
FIG_DIR    = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PRE_START  = pd.Timestamp("2026-01-12")
PRE_END    = pd.Timestamp("2026-05-31")
TEST_START = pd.Timestamp("2026-06-01")
TEST_END   = pd.Timestamp("2026-06-28")

REV_SHARE_MIN = 0.12
REV_SHARE_MAX = 0.25
BUDGET_CAP    = 300000
MIN_TREAT     = 2
MAX_TREAT     = 4
MIN_CONTROLS  = 4
MDE_TARGET    = 0.025


# =====================================================================
# PART 1 — LOAD, CLEAN, VALIDATE
# =====================================================================
print("=" * 78)
print("PART 1 — LOAD, CLEAN, VALIDATE")
print("=" * 78)

if not Path(XLSX_PATH).exists():
    raise FileNotFoundError(f"Workbook not found: {XLSX_PATH}")

required_sheets = ["Markets", "Daily metrics", "Audience overlap", "Operations calendar"]
existing_sheets = pd.ExcelFile(XLSX_PATH).sheet_names
for s in required_sheets:
    if s not in existing_sheets:
        raise ValueError(f"Missing sheet: {s}")

markets_raw  = pd.read_excel(XLSX_PATH, sheet_name="Markets").dropna(how="all")
daily_raw    = pd.read_excel(XLSX_PATH, sheet_name="Daily metrics").dropna(how="all")
overlap_raw  = pd.read_excel(XLSX_PATH, sheet_name="Audience overlap").dropna(how="all")
calendar_raw = pd.read_excel(XLSX_PATH, sheet_name="Operations calendar").dropna(how="all")

daily    = daily_raw.copy()
markets  = markets_raw.copy()
overlap  = overlap_raw.copy()
calendar = calendar_raw.copy()


def clean_state(s):
    if pd.isna(s):
        return np.nan
    return str(s).strip().upper()


for df in (daily, markets, overlap, calendar):
    for col in df.columns:
        if "state" in col.lower():
            df[col] = df[col].apply(clean_state)

daily   = daily[daily["state"].notna()].copy()
markets = markets[markets["state"].notna()].copy()
overlap = overlap[overlap["state_a"].notna() & overlap["state_b"].notna()].copy()

# FIX 2: clean the Markets table itself (not only the "slim" copy), because
# Parts 2-5 use `markets`, not `markets_slim`. Before, `markets` still had the
# raw delivery_group / can_increase_spend values, so "== 1" checks and group
# matching could silently fail. Also, a blank delivery_group used to make
# groupby() drop that state, so it could never be picked as a treatment state.
def to_flag(v):
    return 1 if str(v).strip().lower() in ("1", "1.0", "true", "yes", "y") else 0

markets["can_increase_spend"] = markets["can_increase_spend"].apply(to_flag).astype(int)
markets["delivery_group"] = markets["delivery_group"].apply(clean_state)
markets["delivery_group"] = markets["delivery_group"].fillna(markets["state"])

daily["date"]          = pd.to_datetime(daily["date"], errors="coerce")
calendar["start_date"] = pd.to_datetime(calendar["start_date"], errors="coerce")
calendar["end_date"]   = pd.to_datetime(calendar["end_date"], errors="coerce")

if daily["date"].isna().any():
    raise ValueError(f"Unparseable dates: {daily['date'].isna().sum()} rows")

daily["revenue_feed_complete"] = pd.to_numeric(
    daily["revenue_feed_complete"], errors="coerce"
)

# Duplicate check
duplicate_mask = daily.duplicated(subset=["date", "state"], keep=False)
duplicates = daily[duplicate_mask].copy()
print(f"Duplicate rows found: {len(duplicates)}")

if len(duplicates) > 0:
    duplicate_pairs = duplicates.groupby(["date", "state"]).size()
    print(f"Duplicate state-date pairs: {len(duplicate_pairs)}")

    value_columns = [c for c in daily.columns if c not in ["date", "state"]]
    conflicting = (
        duplicates
        .groupby(["date", "state"])[value_columns]
        .nunique()
        .gt(1)
        .any(axis=1)
    )
    print(f"Conflicting duplicate pairs: {int(conflicting.sum())}")

    if conflicting.any():
        raise ValueError("Conflicting duplicate state-date records found.")

# FIX 1: nunique() ignores blanks, so a "non-conflicting" pair can be one full
# row + one row with blanks. drop_duplicates(keep="first") might keep the blank
# one. groupby().first() takes the first NON-blank value in each column instead.
daily = daily.groupby(["date", "state"], as_index=False).first()
daily = daily.sort_values(["state", "date"]).reset_index(drop=True)

# GA feed incident
ga_feed_mask = (
    (daily["state"] == "GA")
    & (daily["date"] >= pd.Timestamp("2026-04-13"))
    & (daily["date"] <= pd.Timestamp("2026-04-16"))
)
daily.loc[ga_feed_mask, "net_revenue_usd"] = np.nan
daily.loc[ga_feed_mask, "orders"] = np.nan

ga_missing_revenue = (
    (daily["state"] == "GA")
    & (daily["net_revenue_usd"].isna())
    & (~ga_feed_mask)
)
print(f"GA missing revenue outside April feed incident: {int(ga_missing_revenue.sum())}")

ga_feb_inconsistent = (
    (daily["state"] == "GA")
    & (daily["net_revenue_usd"].isna())
    & (daily["revenue_feed_complete"] == 1)
)
print(f"GA missing revenue with feed_complete=1: {int(ga_feb_inconsistent.sum())}")

# NV anomalies
nv_neg_spend  = (daily["state"] == "NV") & (daily["paid_social_spend_usd"] < 0)
nv_neg_orders = (daily["state"] == "NV") & (daily["orders"] < 0)
print(f"NV negative spend nulled: {int(nv_neg_spend.sum())}")
print(f"NV negative orders nulled: {int(nv_neg_orders.sum())}")
daily.loc[nv_neg_spend, "paid_social_spend_usd"] = np.nan
daily.loc[nv_neg_orders, "orders"] = np.nan

# Merge markets
markets_slim = markets[[
    "state", "can_increase_spend", "delivery_group",
    "planned_bau_daily_spend_usd", "audited_last_28d_revenue_usd"
]].copy()

if markets_slim["state"].duplicated().any():
    raise ValueError("Markets contains duplicate states.")

daily = daily.merge(markets_slim, on="state", how="left", indicator=True)

unmatched = daily[daily["_merge"] == "left_only"]
if len(unmatched) > 0:
    raise ValueError(f"Unmatched states: {unmatched['state'].unique().tolist()}")

daily = daily.drop(columns=["_merge"])

expected_states = sorted(markets_slim["state"].unique().tolist())
found_states = sorted(daily["state"].unique().tolist())
if set(expected_states) != set(found_states):
    raise ValueError("State mismatch.")

if daily["date"].min() != PRE_START:
    raise ValueError(f"Unexpected min date: {daily['date'].min()}")
if daily["date"].max() != PRE_END:
    raise ValueError(f"Unexpected max date: {daily['date'].max()}")

expected_days = pd.date_range(PRE_START, PRE_END, freq="D")
panel_check = daily.groupby("state")["date"].nunique()
if (panel_check != len(expected_days)).any():
    raise ValueError(f"Incomplete panels: {panel_check[panel_check != len(expected_days)].to_dict()}")


def parse_state_scope(scope, all_states):
    if pd.isna(scope):
        return []
    s = str(scope).strip().upper()
    if s == "ALL":
        return all_states
    return [x.strip() for x in s.split(";") if x.strip()]


calendar["affected_states"] = calendar["state_scope"].apply(
    lambda x: parse_state_scope(x, expected_states)
)

print(f"states:     {len(found_states)}")
print(f"date range: {daily['date'].min().date()} to {daily['date'].max().date()} ({len(expected_days)} days)")
print(f"daily rows: {len(daily)}")

daily.to_csv(OUTPUT_DIR / "daily_clean.csv", index=False)
markets_slim.to_csv(OUTPUT_DIR / "markets_clean.csv", index=False)
overlap.to_csv(OUTPUT_DIR / "overlap_clean.csv", index=False)
calendar.to_csv(OUTPUT_DIR / "calendar_clean.csv", index=False)
print()


# =====================================================================
# PART 2 — EXPLORATORY DATA ANALYSIS
# =====================================================================
print("=" * 78)
print("PART 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 78)

pre = daily[(daily["date"] >= PRE_START) & (daily["date"] <= PRE_END)].copy()

# FIX 3: build ONE wide revenue table (date x state) and fill the blanks.
# The original used pivot_table(aggfunc="sum"). Summing a blank day gives 0,
# so GA's incident days and February gaps became "zero revenue" days, which
# distorted the correlations, the synthetic-control fit, R2 and the MDE.
# pivot() keeps blanks as blanks; we then fill them by linear interpolation.
rev_wide_raw = pre.pivot(index="date", columns="state", values="net_revenue_usd").sort_index()
n_filled = int(rev_wide_raw.isna().sum().sum())
rev_wide = rev_wide_raw.interpolate(method="time", limit_direction="both")
print(f"blank state-days filled by interpolation for modelling: {n_filled}")

total_audited = markets["audited_last_28d_revenue_usd"].sum()
markets["revenue_share_pct"] = (
    markets["audited_last_28d_revenue_usd"] / total_audited * 100
).round(3)
markets["incr_spend_28d"] = (
    markets["planned_bau_daily_spend_usd"] * 0.40 * 28
).round(2)

print(f"total audited revenue: ${total_audited:,.2f}")
print(f"12% floor:             ${total_audited * 0.12:,.2f}")
print(f"25% ceiling:           ${total_audited * 0.25:,.2f}")
print()

pre_means = pre.groupby("state")["net_revenue_usd"].agg(
    ["mean", "std", "min", "max", "count"]
).reset_index()
pre_means["cv_pct"] = (pre_means["std"] / pre_means["mean"] * 100).round(2)

# Figures
fig = px.line(pre, x="date", y="net_revenue_usd", color="state",
              title="Daily net revenue by state (pre-period)")
fig.update_layout(height=550, hovermode="x unified")
fig.write_html(FIG_DIR / "01_daily_revenue_by_state.html")

fig = px.bar(markets.sort_values("revenue_share_pct", ascending=False),
             x="state", y="revenue_share_pct",
             title="Revenue share by state")
fig.add_hline(y=12, line_dash="dot", line_color="green")
fig.add_hline(y=25, line_dash="dot", line_color="red")
fig.update_layout(height=480)
fig.write_html(FIG_DIR / "02_revenue_share.html")

fig = px.bar(markets.sort_values("incr_spend_28d", ascending=False),
             x="state", y="incr_spend_28d",
             title="40% incremental spend over 28 days")
fig.add_hline(y=BUDGET_CAP, line_dash="dash", line_color="red")
fig.update_layout(height=480)
fig.write_html(FIG_DIR / "03_incremental_spend.html")

corr = rev_wide.corr().round(3)
fig = px.imshow(corr, text_auto=True, aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Pre-period correlation heatmap")
fig.update_layout(height=700, width=900)
fig.write_html(FIG_DIR / "04_correlation_heatmap.html")

print(f"figures saved to: {FIG_DIR}")
print()


# =====================================================================
# PART 3 — FEASIBLE TREATMENT DESIGN ENUMERATION
# =====================================================================
print("=" * 78)
print("PART 3 — FEASIBLE DESIGN ENUMERATION")
print("=" * 78)

eligible_states = sorted(
    markets.loc[markets["can_increase_spend"] == 1, "state"].tolist()
)
ineligible_states = sorted(
    markets.loc[markets["can_increase_spend"] != 1, "state"].tolist()
)

group_to_states = (
    markets.groupby("delivery_group")["state"]
    .apply(lambda x: tuple(sorted(x)))
    .to_dict()
)

print(f"treatment-eligible:   {eligible_states}")
print(f"treatment-ineligible: {ineligible_states}")
print()

calendar_test = calendar[
    (calendar["end_date"].fillna(pd.Timestamp("2099-12-31")) >= TEST_START)
    & (calendar["start_date"] <= TEST_END)
].copy()

# Only exclude states whose test-period events are direct promotions/discounts
excluded_from_comparison = set()
for _, row in calendar_test.iterrows():
    event = str(row["event"]).lower()
    if any(k in event for k in ["promotion", "discount"]):
        excluded_from_comparison.update(row["affected_states"])

print(f"calendar events in test window: {len(calendar_test)}")
# FIX 12: print the events so you can confirm the keyword match
# ("promotion"/"discount") really caught every promo event (e.g. "promo", "sale").
for _, r in calendar_test.iterrows():
    print(f"   - {r['state_scope']}: {r['event']}")
print(f"excluded from comparison (promotions): {sorted(excluded_from_comparison)}")
print()


def rev_share(states):
    return markets[markets["state"].isin(states)]["audited_last_28d_revenue_usd"].sum() / total_audited


def incr_spend(states):
    return markets[markets["state"].isin(states)]["planned_bau_daily_spend_usd"].sum() * 0.40 * 28


overlap_dict = {}
for _, row in overlap.iterrows():
    a, b = row["state_a"], row["state_b"]
    share = row["cross_border_audience_share"]
    overlap_dict[(a, b)] = share
    overlap_dict[(b, a)] = share


def internal_overlap(states):
    vals = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            v = overlap_dict.get((states[i], states[j]))
            if v is not None:
                vals.append(v)
    return max(vals) if vals else 0.0


def treatment_control_overlap(treatment, controls):
    vals = []
    for t in treatment:
        for c in controls:
            v = overlap_dict.get((t, c))
            if v is not None:
                vals.append(v)
    return max(vals) if vals else 0.0


def calendar_flags(states):
    flags = []
    for _, row in calendar_test.iterrows():
        if set(row["affected_states"]) & set(states):
            flags.append(f"{row['event']}")
    return flags


# FIX 4: treatment groups must also avoid the promotion-contaminated states.
# The risk table says "Exclude FL from treatment and comparison", but the
# original only removed them from the comparison pool, so FL could still be
# chosen as a treatment state.
valid_groups = [
    (g, s) for g, s in group_to_states.items()
    if all(st in eligible_states for st in s)
    and not (set(s) & excluded_from_comparison)
]

feasible = []
for n_groups in range(1, len(valid_groups) + 1):
    for group_combo in combinations(valid_groups, n_groups):
        states = sorted([st for _, grp in group_combo for st in grp])
        if not (MIN_TREAT <= len(states) <= MAX_TREAT):
            continue

        rs = rev_share(states)
        if rs < REV_SHARE_MIN or rs > REV_SHARE_MAX:
            continue

        isp = incr_spend(states)
        if isp > BUDGET_CAP:
            continue

        treated_groups = {g for g, _ in group_combo}
        controls = []
        for s in markets["state"].tolist():
            if s in states or s in excluded_from_comparison:
                continue
            sg = markets.loc[markets["state"] == s, "delivery_group"].iloc[0]
            if sg in treated_groups:
                continue
            controls.append(s)

        if len(controls) < MIN_CONTROLS:
            continue

        flags = calendar_flags(states)

        feasible.append({
            "treatment": tuple(states),
            "n_treatment": len(states),
            "groups": tuple(g for g, _ in group_combo),
            "revenue_share_pct": round(rs * 100, 4),
            "incremental_spend": round(isp, 2),
            "budget_remaining": round(BUDGET_CAP - isp, 2),
            "n_comparison": len(controls),
            "comparison_pool": tuple(sorted(controls)),
            "internal_overlap_pct": round(internal_overlap(states) * 100, 4),
            "trt_ctrl_overlap_pct": round(treatment_control_overlap(states, controls) * 100, 4),
            "calendar_flags": flags,
            "n_calendar_flags": len(flags),
        })

# FIX 7: stop with a clear message instead of a confusing KeyError below
if not feasible:
    raise SystemExit("No feasible designs passed the hard constraints. "
                     "Check the constraint settings / exclusions.")

feasible_df = pd.DataFrame([{
    "treatment": "+".join(f["treatment"]),
    "groups": "|".join(f["groups"]),
    "n_treatment": f["n_treatment"],
    "revenue_share_pct": f["revenue_share_pct"],
    "incremental_spend": f["incremental_spend"],
    "budget_remaining": f["budget_remaining"],
    "n_comparison": f["n_comparison"],
    "internal_overlap_pct": f["internal_overlap_pct"],
    "trt_ctrl_overlap_pct": f["trt_ctrl_overlap_pct"],
    "n_calendar_flags": f["n_calendar_flags"],
    "calendar_flags": "; ".join(f["calendar_flags"]) if f["calendar_flags"] else "",
    "comparison_pool": ";".join(f["comparison_pool"]),
} for f in feasible])

feasible_df["rev_share_dist_from_mid"] = (feasible_df["revenue_share_pct"] - 18.5).abs()

feasible_df = feasible_df.sort_values([
    "n_calendar_flags",
    "trt_ctrl_overlap_pct",
    "internal_overlap_pct",
    "rev_share_dist_from_mid",
]).reset_index(drop=True)

feasible_df["rank"] = feasible_df.index + 1
feasible_df.to_csv(OUTPUT_DIR / "candidate_designs.csv", index=False)

shortlist = feasible_df.head(10).copy()
shortlist.to_csv(OUTPUT_DIR / "geox_shortlist.csv", index=False)

print(f"feasible designs: {len(feasible_df)}")
print(f"shortlist size:   {len(shortlist)}")
print()
print(shortlist[["rank", "treatment", "revenue_share_pct",
                 "incremental_spend", "n_calendar_flags",
                 "trt_ctrl_overlap_pct"]].to_string(index=False))
print()

fig = px.scatter(feasible_df, x="revenue_share_pct", y="incremental_spend",
                 color="n_calendar_flags", size="n_treatment",
                 hover_name="treatment",
                 title="Feasible design space")
fig.add_vline(x=12, line_dash="dot", line_color="green")
fig.add_vline(x=25, line_dash="dot", line_color="red")
fig.add_hline(y=BUDGET_CAP, line_dash="dash", line_color="red")
fig.update_layout(height=600)
fig.write_html(FIG_DIR / "05_feasible_design_space.html")


# =====================================================================
# PART 3.5 — PRE-GEOX COUNTERFACTUAL SCREENING
# =====================================================================
print("=" * 78)
print("PART 3.5 — PRE-GEOX SCREENING")
print("=" * 78)

pre_pivot = rev_wide   # FIX 3: filled wide table, no fake zeros


def fit_synthetic_control(treatment_states, control_states, pre_matrix):
    y = pre_matrix[treatment_states].sum(axis=1).values
    X = pre_matrix[control_states].values

    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-9
    Xn = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std  = y.std() + 1e-9
    yn = (y - y_mean) / y_std

    w, *_ = np.linalg.lstsq(Xn, yn, rcond=None)
    yhat = (Xn @ w) * y_std + y_mean

    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / (ss_tot + 1e-9)
    resid = y - yhat

    # FIX 5b: day-to-day errors are usually correlated; needed for the MDE
    rho = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    rho = 0.0 if np.isnan(rho) else float(min(max(rho, 0.0), 0.9))

    return {
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "resid_std": float(np.std(resid, ddof=1)),
        "rho": rho,
        "weights": dict(zip(control_states, w.tolist())),
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_mean, "y_std": y_std,
    }


def predict_from_fit(fit, control_states, X_future):
    Xn = (X_future - fit["X_mean"]) / fit["X_std"]
    w = np.array([fit["weights"][c] for c in control_states])
    return (Xn @ w) * fit["y_std"] + fit["y_mean"]


def approx_mde(resid_std, rho, n_test_days, treatment_states, pre_matrix):
    z_alpha = 1.6449   # two-sided alpha = 0.10
    z_beta  = 0.8416   # power = 0.80

    # FIX 5a: resid_std is already the error of the COMBINED treatment revenue
    # (y is the sum over treatment states), so the original
    # "resid_std / sqrt(n_treat)" wrongly shrank the noise and made every
    # design look more powerful than it is. Removed.
    # FIX 5b: errors on neighbouring days are correlated, so summing 28 days
    # is noisier than sqrt(28) * sd. The AR(1) factor below corrects for that.
    inflation = np.sqrt((1 + rho) / (1 - rho))
    test_sd = resid_std * np.sqrt(n_test_days) * inflation

    mean_daily_rev = float(pre_matrix[treatment_states].sum(axis=1).mean())
    cumulative_rev = mean_daily_rev * n_test_days
    mde_abs = (z_alpha + z_beta) * test_sd
    mde_pct = mde_abs / (cumulative_rev + 1e-9)
    return {"mde_abs": mde_abs, "mde_pct": mde_pct}


def placebo_check(treatment_states, control_states, pre_matrix,
                  n_placebo=30, holdout_days=14, seed=42):
    rng = np.random.default_rng(seed)
    full = pre_matrix.copy()
    n = len(full)
    if n < 2 * holdout_days + 5:
        return {"placebo_mean": np.nan, "placebo_std": np.nan}
    diffs = []
    for _ in range(n_placebo):
        cut = int(rng.integers(holdout_days + 5, n - holdout_days))
        train = full.iloc[:cut]
        holdout = full.iloc[cut: cut + holdout_days]
        try:
            fit = fit_synthetic_control(treatment_states, control_states, train)
            yhat_h = predict_from_fit(fit, control_states, holdout[control_states].values)
            y_h = holdout[treatment_states].sum(axis=1).values
            diffs.append(float((y_h - yhat_h).mean()))
        except Exception:
            continue
    if not diffs:
        return {"placebo_mean": np.nan, "placebo_std": np.nan}
    diffs = np.array(diffs)
    return {"placebo_mean": float(diffs.mean()), "placebo_std": float(diffs.std(ddof=1))}


results = []
for _, row in shortlist.iterrows():
    treatment = [s.strip().upper() for s in str(row["treatment"]).split("+")]
    controls  = [s.strip().upper() for s in str(row["comparison_pool"]).split(";") if s.strip()]

    missing = [s for s in treatment + controls if s not in pre_pivot.columns]
    if missing:
        continue

    fit  = fit_synthetic_control(treatment, controls, pre_pivot)
    mde  = approx_mde(fit["resid_std"], fit["rho"],
                      (TEST_END - TEST_START).days + 1,
                      treatment, pre_pivot)
    plac = placebo_check(treatment, controls, pre_pivot)

    results.append({
        "rank_pre_geox": row["rank"],
        "treatment": "+".join(treatment),
        "revenue_share_pct": row["revenue_share_pct"],
        "incremental_spend": row["incremental_spend"],
        "n_calendar_flags": row["n_calendar_flags"],
        "trt_ctrl_overlap_pct": row["trt_ctrl_overlap_pct"],
        "internal_overlap_pct": row["internal_overlap_pct"],
        "pre_r2": round(fit["r2"], 4),
        "pre_rmse": round(fit["rmse"], 2),
        "approx_mde_pct": round(mde["mde_pct"], 4),
        "approx_mde_abs": round(mde["mde_abs"], 2),
        "mde_ok_approx": bool(mde["mde_pct"] <= MDE_TARGET),
        "placebo_mean": round(plac["placebo_mean"], 2) if not np.isnan(plac["placebo_mean"]) else np.nan,
        "placebo_std": round(plac["placebo_std"], 2) if not np.isnan(plac["placebo_std"]) else np.nan,
    })

# FIX 7: sort_values on an empty table crashes with KeyError
if not results:
    raise SystemExit("Screening produced no results (column names did not match pre-period data).")

screening_df = pd.DataFrame(results).sort_values(
    ["mde_ok_approx", "approx_mde_pct", "pre_r2"],
    ascending=[False, True, False],
).reset_index(drop=True)
screening_df["screen_rank"] = screening_df.index + 1
screening_df.to_csv(OUTPUT_DIR / "pre_geox_screening.csv", index=False)

print(screening_df[["screen_rank", "treatment", "pre_r2",
                    "approx_mde_pct", "mde_ok_approx"]].to_string(index=False))
print()


# =====================================================================
# PART 4 — GEOX DESIGN EVALUATION
# =====================================================================
print("=" * 78)
print("PART 4 — GEOX DESIGN EVALUATION")
print("=" * 78)

geox_metrics = None
geox_used = False

try:
    import meridian_geox as geox

    # FIX 10: the original dropped every row with a blank (GA incident days,
    # NV nulled spend), leaving an unbalanced panel that design tools usually
    # reject. Fill the blanks per state instead, so every state has every day.
    geox_panel = pre[["date", "state", "net_revenue_usd", "paid_social_spend_usd"]].copy()
    geox_panel = geox_panel.sort_values(["state", "date"])
    for c in ["net_revenue_usd", "paid_social_spend_usd"]:
        geox_panel[c] = geox_panel.groupby("state")[c].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
    geox_panel["net_revenue_usd"] = geox_panel["net_revenue_usd"].clip(lower=0)

    geox_data = geox_panel.rename(
        columns={
            "state": "location",
            "net_revenue_usd": "conversions",
            "paid_social_spend_usd": "spend",
        }
    ).dropna(subset=["date", "location", "conversions", "spend"])

    geox_data["location"] = geox_data["location"].astype(str)
    geox_data["date"]     = pd.to_datetime(geox_data["date"])

    print(f"pretest rows: {len(geox_data)}")
    print(f"locations:    {geox_data['location'].nunique()}")

    design_config = geox.DesignConfig(
        experiment_duration=datetime.timedelta(days=28),
        experiment_types=geox.ExperimentType.HEAVY_UP,
        methodology=geox.Methodology.TBR,
        geo_assignment_rule=geox.GeoAssignmentRule.STRATIFIED_SAMPLING,
        cell_count=1,
        alpha=0.10,
        power=0.80,
        test_type=geox.TestType.TWO_SIDED,
        design_output_count=10,
        n_candidates=10000,
        n_ranked_candidates=50,
        seed=42,
        min_r2=0.70,
    )

    constraints = geox.Constraints(
        excluded_geos=set(excluded_from_comparison),
        budget_constraint=geox.Budget(budget_pct=0.40),
        max_conversions_percent=0.25,
    )

    print("running geox.run_design()...")
    design_set = geox.run_design(geox_data, design_config, constraints)

    geox_metrics = design_set.design_metrics.copy()
    geox_metrics.to_csv(OUTPUT_DIR / "geox_design_metrics.csv", index=False)
    geox_used = True

    print("SUCCESS — GeoX designs generated.")
    print(geox_metrics.head(10).to_string(index=False))
    print()
    # FIX 10: these constraints only cover part of the rules (no 12% floor,
    # no $300k cap, no CA-as-control-only, no NC+SC pairing, no 2-4 states).
    print("WARNING: GeoX does not enforce all hard constraints. Check every GeoX "
          "design against candidate_designs.csv before choosing one.")
    print()

except Exception as e:
    print(f"GeoX ERROR: {type(e).__name__}: {e}")
    print("Falling back to pre-GeoX screening for the report.")
    print()


# =====================================================================
# PART 5 — INTERACTIVE REPORT + GITHUB PAGES
# =====================================================================
print("=" * 78)
print("PART 5 — INTERACTIVE REPORT")
print("=" * 78)

# State map
STATE_COORDS = {
    "OH": (40.4, -82.8), "PA": (41.2, -77.2), "MI": (44.3, -85.4),
    "WI": (43.8, -89.0), "MO": (38.4, -92.3), "IN": (39.8, -86.3),
    "TN": (35.8, -86.3), "KY": (37.5, -85.3), "CA": (37.2, -119.7),
    "TX": (31.5, -99.4), "FL": (27.8, -81.7), "GA": (32.6, -83.4),
    "NC": (35.6, -79.4), "SC": (33.9, -80.9), "NV": (39.3, -116.6),
    "CO": (39.0, -105.5),
}
map_rows = []
for _, r in markets.iterrows():
    if r["state"] in STATE_COORDS:
        lat, lon = STATE_COORDS[r["state"]]
        map_rows.append({
            "state": r["state"], "lat": lat, "lon": lon,
            "revenue_share_pct": r["revenue_share_pct"],
            "incr_spend": r["incr_spend_28d"],
            "eligible": "eligible" if r["can_increase_spend"] == 1 else "ineligible",
        })
map_df = pd.DataFrame(map_rows)

map_fig = px.scatter_geo(
    map_df, lat="lat", lon="lon",
    size="revenue_share_pct", color="eligible",
    hover_name="state",
    color_discrete_map={"eligible": "#2ecc71", "ineligible": "#e74c3c"},
    title="States by revenue share",
    scope="usa",
)
map_fig.update_layout(height=500)

state_share_fig = px.bar(
    markets.sort_values("revenue_share_pct", ascending=False),
    x="state", y="revenue_share_pct",
    color="can_increase_spend",
    color_continuous_scale=["#e74c3c", "#2ecc71"],
    title="Revenue share by state",
)
state_share_fig.add_hline(y=12, line_dash="dot", line_color="green")
state_share_fig.add_hline(y=25, line_dash="dot", line_color="red")
state_share_fig.update_layout(height=480)

daily_rev_fig = px.line(pre, x="date", y="net_revenue_usd", color="state",
                        title="Daily net revenue (pre-period)")
daily_rev_fig.update_layout(height=520, hovermode="x unified")

# FIX 11: colour by text labels. A True/False column with a True/False
# colour map can fail to match in Plotly Express and fall back to default colours.
screening_plot = screening_df.sort_values("approx_mde_pct").copy()
screening_plot["mde_status"] = screening_plot["mde_ok_approx"].map(
    {True: "meets target", False: "above target"}
)
screening_fig = px.bar(
    screening_plot,
    x="approx_mde_pct", y="treatment", orientation="h",
    color="mde_status",
    color_discrete_map={"meets target": "green", "above target": "crimson"},
    title="Pre-GeoX approximate MDE by design",
)
screening_fig.add_vline(x=MDE_TARGET, line_dash="dash", line_color="black")
screening_fig.update_layout(height=550)


def fig_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": True, "responsive": True})


constraint_html = markets[[
    "state", "can_increase_spend", "delivery_group",
    "audited_last_28d_revenue_usd", "revenue_share_pct",
    "planned_bau_daily_spend_usd", "incr_spend_28d",
]].sort_values("revenue_share_pct", ascending=False).to_html(
    index=False, border=0, classes="tbl",
    float_format=lambda x: f"{x:,.2f}"
)

cal_view = calendar[["state_scope", "start_date", "end_date", "event"]].copy()
cal_view["start_date"] = cal_view["start_date"].dt.date.astype(str)
cal_view["end_date"] = cal_view["end_date"].fillna(pd.Timestamp("2099-12-31")).dt.date.astype(str)
calendar_html = cal_view.to_html(index=False, border=0, classes="tbl")


# Recommendation decision
if geox_metrics is not None and len(geox_metrics) > 0:
    best = geox_metrics.iloc[0]
    try:
        mde = float(best["mde"])
        r2  = float(best["r2"])
        aa  = float(best.get("p_value (AA)", 1.0))
    except Exception:
        mde, r2, aa = 0.05, 0.0, 1.0

    if r2 >= 0.80 and aa > 0.10 and mde <= MDE_TARGET:
        decision, cls, source = "PROCEED", "green", "GeoX"
    elif r2 >= 0.70 and mde <= 0.035:
        decision, cls, source = "REVISE", "amber", "GeoX"
    else:
        decision, cls, source = "POSTPONE", "red", "GeoX"
    rec_block = f"""
    <p><span class="pill {cls}">{decision}</span></p>
    <p><b>Source:</b> {source}</p>
    <p><b>Pre-period R²:</b> {r2:.4f} &nbsp;|&nbsp;
       <b>MDE:</b> {mde:.4%} &nbsp;|&nbsp;
       <b>A/A p-value:</b> {aa:.4f}</p>
    """
else:
    best = screening_df.iloc[0]
    mde = float(best["approx_mde_pct"])
    r2  = float(best["pre_r2"])

    if r2 >= 0.80 and mde <= MDE_TARGET:
        decision, cls = "PROCEED", "green"
    elif r2 >= 0.70 and mde <= 0.035:
        decision, cls = "REVISE", "amber"
    else:
        decision, cls = "POSTPONE", "red"

    rec_block = f"""
    <p><span class="pill {cls}">{decision}</span></p>
    <p><b>Source:</b> pre-GeoX screening (GeoX unavailable)</p>
    <p><b>Selected design:</b> {best['treatment']}</p>
    <p><b>Pre-period R²:</b> {r2:.4f} &nbsp;|&nbsp;
       <b>Approximate MDE:</b> {mde:.4%}</p>
    """

# FIX 8: the original text always said "evaluated with GeoX", even when GeoX
# failed and the fallback was used. Now the text matches what actually ran.
if geox_used:
    design_method_text = "then evaluated with GeoX."
else:
    design_method_text = ("screened with the synthetic-control approximation only "
                          "(GeoX was not available in this run).")
tooling_text = "Python, GeoX, Plotly" if geox_used else "Python, Plotly (GeoX not used)"


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Geo Experiment Design Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f5f6fa; color: #1a1a1a; }}
  header {{ background: #1e272e; color: #fff; padding: 30px 40px; }}
  header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
  header p {{ margin: 0; opacity: 0.8; font-size: 14px; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 30px 20px 80px; }}
  section {{ background: #fff; padding: 24px 28px; margin-bottom: 22px;
            border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  section h2 {{ margin-top: 0; font-size: 20px;
               border-bottom: 2px solid #eaecef; padding-bottom: 8px; }}
  table.tbl {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  table.tbl th, table.tbl td {{ border: 1px solid #e0e3e7; padding: 7px 10px; text-align: left; }}
  table.tbl th {{ background: #f0f2f5; }}
  .metric {{ display: inline-block; margin: 6px 20px 6px 0; }}
  .metric .k {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
  .metric .v {{ font-size: 20px; font-weight: 600; }}
  .pill {{ display: inline-block; padding: 4px 12px; border-radius: 12px;
          font-size: 13px; font-weight: 600; }}
  .pill.green {{ background: #e6f7ec; color: #1e7d3f; }}
  .pill.red   {{ background: #fde8e8; color: #b32d2d; }}
  .pill.amber {{ background: #fff4e0; color: #a06300; }}
  .note {{ background: #fff8e1; border-left: 4px solid #f0b429;
          padding: 12px 16px; margin: 14px 0; border-radius: 4px; }}
</style>
</head>
<body>
<header>
  <h1>Geo Incrementality Test — Design Report</h1>
  <p>40% paid-social spend increase &nbsp;|&nbsp; Jun 1–28, 2026 &nbsp;|&nbsp; Planning MDE 2.5%</p>
</header>
<main>

<section>
<h2>Summary</h2>
<div class="metric"><div class="k">States</div><div class="v">{markets['state'].nunique()}</div></div>
<div class="metric"><div class="k">Pre-period days</div><div class="v">{pre['date'].nunique()}</div></div>
<div class="metric"><div class="k">Test days</div><div class="v">{(TEST_END - TEST_START).days + 1}</div></div>
<div class="metric"><div class="k">Total audited revenue</div><div class="v">${total_audited:,.0f}</div></div>
<div class="metric"><div class="k">Feasible designs</div><div class="v">{len(feasible_df)}</div></div>
</section>

<section>
<h2>1. Raw Data Analysis</h2>
<p>The panel covers {markets['state'].nunique()} states × {pre['date'].nunique()} days (Jan 12 – May 31, 2026). Data cleaning handled
duplicate state-date pairs, mixed-case state codes, GA revenue-feed incidents
(April 13–16), GA February missing revenue, and NV negative spend/order anomalies.
Blank days were filled by linear interpolation for modelling only.
The primary outcome is finance <code>net_revenue_usd</code>.</p>
<h3>Revenue share and eligibility</h3>
{constraint_html}
<h3>Operations calendar</h3>
{calendar_html}
<h3>State map</h3>
{fig_html(map_fig)}
<h3>Revenue share by state</h3>
{fig_html(state_share_fig)}
<h3>Daily revenue</h3>
{fig_html(daily_rev_fig)}
</section>

<section>
<h2>2. Hypothesis</h2>
<p><b>H<sub>0</sub>:</b> A 40% daily increase in paid-social spend produces no
incremental increase in combined finance net revenue in the treatment states
relative to BAU.</p>
<p><b>H<sub>1</sub>:</b> The 40% spend increase produces a positive incremental
increase in combined finance net revenue.</p>
<p><b>Planning effect:</b> 2.5% of combined treatment-state revenue (design target,
not a promised outcome).</p>
<p><b>Primary metric:</b> total <code>net_revenue_usd</code> over Jun 1–28, 2026.</p>
</section>

<section>
<h2>3. Experiment Design</h2>
<p>{len(feasible_df)} designs passed the hard constraints (2–4 treatment states,
12–25% revenue share, ≤ $300k incremental spend, treatment eligibility,
delivery-group coupling, ≥ 4 comparison states). Top 10 were {design_method_text}</p>
<h3>Pre-GeoX screening</h3>
{screening_fig.to_html(full_html=False, include_plotlyjs=False)}
</section>

<section>
<h2>4. Risks</h2>
<table class="tbl">
<thead><tr><th>Risk</th><th>Effect</th><th>Response</th></tr></thead>
<tbody>
<tr><td>FL June promotion</td><td>Contaminates FL revenue</td><td>Exclude FL from treatment and comparison</td></tr>
<tr><td>TX fulfilment expansion</td><td>Structural break</td><td>Flag TX; review if selected</td></tr>
<tr><td>GA revenue-feed incident</td><td>Missing pre-period revenue</td><td>Exclude affected observations</td></tr>
<tr><td>NC+SC shared buying group</td><td>Delivery-group coupling</td><td>Treat as one unit</td></tr>
<tr><td>CA media contract</td><td>Cannot increase CA spend</td><td>CA as control only</td></tr>
<tr><td>NV data anomalies</td><td>Negative spend/orders</td><td>Null out anomalies</td></tr>
<tr><td>CO attribution change</td><td>Platform metric shift</td><td>Finance revenue unaffected</td></tr>
<tr><td>Weak pre-period fit</td><td>Poor counterfactual</td><td>Reject design</td></tr>
<tr><td>Insufficient power</td><td>MDE above target</td><td>Revise or postpone</td></tr>
</tbody>
</table>
</section>

<section>
<h2>5. Recommendation</h2>
{rec_block}
<div class="note">
The final decision combines the hard assignment constraints, design diagnostics
(pre-period R², MDE{', A/A p-value' if geox_used else ''}), and operational considerations.
</div>
</section>

</main>
</body>
</html>
"""

report_path = REPORT_DIR / "index.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html)

# GitHub Pages helper
with open(REPORT_DIR / ".nojekyll", "w") as f:
    f.write("")

with open(OUTPUT_DIR / "README.md", "w") as f:
    f.write("# Geo Experiment Design Report\n\n"
            "Interactive report for the 40% paid-social spend increase geo test.\n\n"
            "- Test window: Jun 1 – Jun 28, 2026\n"
            "- Primary metric: finance net revenue (`net_revenue_usd`)\n"
            "- Planning effect (MDE target): 2.5%\n"
            f"- Tooling: {tooling_text}\n")

print(f"report HTML: {report_path}")
print(f"size:        {len(html) / 1024:.1f} KB")
print()
print("=" * 78)
print("PIPELINE COMPLETE")
print("=" * 78)
print()
print("To publish on GitHub Pages:")
print("  1. cd to the geo_experiment folder")
print("  2. git init && git add . && git commit -m \"geo experiment report\"")
print("  3. git remote add origin <your-repo-url>")
print("  4. git push -u origin main")
print("  5. GitHub → Settings → Pages → Source = main /report folder")
print()
print("The report goes live at https://<user>.github.io/<repo>/")
print()

# FIX 9: the original opened pipeline.py for writing (which empties it) BEFORE
# reading __file__. If you ran the script from that same location, it erased
# itself. Now we read first, and skip if source and destination are the same file.
try:
    src_path = Path(__file__).resolve()
    dst_path = (OUTPUT_DIR / "pipeline.py").resolve()
    if src_path != dst_path:
        dst_path.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
except NameError:
    pass   # running in a notebook: no __file__
except Exception as e:
    print(f"Could not save a copy of the script: {e}")
