"""
Geo Incrementality Test -- Step 4: Hard Constraints + Feasible Enumeration
===========================================================================
Applies all hard constraints, generates every feasible treatment combination
(respecting NC+SC atomicity), scores them, and outputs the ranked shortlist.

Constraints enforced:
  1. can_increase_spend = 1 for all treatment states
  2. FL excluded entirely (June promotion confound)
  3. NC + SC must share the same treatment assignment (delivery group NC_SC)
  4. 2-4 treatment states
  5. Revenue share 12-25% of total audited revenue
  6. Incremental spend <= $300,000
  7. Comparison pool >= 4 states
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(r"C:\Users\HP\OneDrive\Desktop\Research\Data\geo_experiment")

BUDGET_CAP = 300_000
REV_SHARE_MIN = 0.12
REV_SHARE_MAX = 0.25
TEST_DAYS = 28
SPEND_UPLIFT = 0.40
MIN_COMPARISON = 4

# ============================================================
# LOAD CLEAN DATA
# ============================================================

markets = pd.read_csv(DATA_DIR / "markets_clean.csv")
overlap = pd.read_csv(DATA_DIR / "overlap_clean.csv")
calendar = pd.read_csv(DATA_DIR / "calendar_clean.csv", parse_dates=["start_date", "end_date"])
daily = pd.read_csv(DATA_DIR / "daily_clean.csv", parse_dates=["date"])

TOTAL_REV = markets["audited_last_28d_revenue_usd"].sum()
ALL_STATES = set(markets["state"].tolist())

print("=" * 80)
print("STEP 4: HARD CONSTRAINTS + FEASIBLE ENUMERATION")
print("=" * 80)
print(f"\n  Total states: {len(ALL_STATES)}")
print(f"  Total audited 28d revenue: ${TOTAL_REV:,.2f}")
print(f"  Revenue share floor (12%): ${TOTAL_REV * 0.12:,.0f}")
print(f"  Revenue share ceiling (25%): ${TOTAL_REV * 0.25:,.0f}")

# ============================================================
# STEP 4A: DEFINE ELIGIBILITY
# ============================================================

print(f"\n--- 4A: Eligibility ---")

# Treatment ineligible: can_increase_spend = 0
treatment_ineligible = set(
    markets[markets["can_increase_spend"] == 0]["state"].tolist()
)
print(f"  can_increase_spend=0 (treatment ineligible): {treatment_ineligible}")

# Excluded from both treatment AND comparison pool
# FL: June 8-21 local promotion contaminates revenue during test window
excluded_entirely = {"FL"}
print(f"  Excluded entirely (confounded during test): {excluded_entirely}")

# Treatment-eligible states
treatment_eligible = ALL_STATES - treatment_ineligible - excluded_entirely
print(f"  Treatment eligible: {sorted(treatment_eligible)} ({len(treatment_eligible)} states)")

# States that can only be in the comparison pool
comparison_only = treatment_ineligible - excluded_entirely  # CA
print(f"  Comparison-only: {comparison_only}")

# ============================================================
# STEP 4B: OPERATIONAL FLAGS
# ============================================================

print(f"\n--- 4B: Operational Flags ---")

# Classify calendar events by risk to treatment states during test period
operational_flags = {}

for _, row in calendar.iterrows():
    scope = row["state_scope"]
    start = row["start_date"]
    end = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp("2099-12-31")
    event = row["event"]

    # Parse scope
    if str(scope).upper() == "ALL":
        continue  # National events affect everyone equally -- not a flag
    states_affected = [s.strip().upper() for s in str(scope).replace(",", ";").split(";")]

    # Flag if event overlaps test period (June 1-28) or is a structural break
    # that could affect pre-period fit
    test_start = pd.Timestamp("2026-06-01")
    test_end = pd.Timestamp("2026-06-28")

    overlaps_test = start <= test_end and end >= test_start
    is_structural = pd.isna(row["end_date"])  # No end date = permanent change

    for state in states_affected:
        if state in ALL_STATES:
            if state not in operational_flags:
                operational_flags[state] = []
            flag_type = "test_confound" if overlaps_test else (
                "structural_break" if is_structural else "pre_period_event"
            )
            operational_flags[state].append({
                "event": event,
                "type": flag_type,
                "start": start,
                "end": end if end.year < 2099 else None,
            })

for state, flags in sorted(operational_flags.items()):
    for f in flags:
        end_str = f["end"].date() if f["end"] else "ongoing"
        print(f"  {state}: [{f['type']}] {f['event']} ({f['start'].date()} to {end_str})")

# ============================================================
# STEP 4C: AUDIENCE OVERLAP MATRIX
# ============================================================

print(f"\n--- 4C: Audience Overlap ---")

overlap_dict = {}
for _, row in overlap.iterrows():
    a, b = row["state_a"], row["state_b"]
    share = row["cross_border_audience_share"]
    overlap_dict[frozenset([a, b])] = share
    print(f"  {a} <-> {b}: {share * 100:.1f}%")

print(f"  All other pairs: < 1% (omitted)")

# ============================================================
# STEP 4D: ENUMERATE FEASIBLE COMBINATIONS
# ============================================================

print(f"\n--- 4D: Enumerate Feasible Combinations ---")

# Build lookup for fast constraint checking
market_lookup = markets.set_index("state").to_dict("index")

def incremental_spend(state):
    return SPEND_UPLIFT * market_lookup[state]["planned_bau_daily_spend_usd"] * TEST_DAYS

def get_trt_ctrl_overlaps(trt_set, ctrl_set):
    """Get audience overlaps where one state is treatment, other is control."""
    overlaps = []
    for pair, share in overlap_dict.items():
        pair_list = list(pair)
        a, b = pair_list[0], pair_list[1]
        if (a in trt_set and b in ctrl_set) or (b in trt_set and a in ctrl_set):
            trt_state = a if a in trt_set else b
            ctrl_state = b if b in ctrl_set else a
            overlaps.append((trt_state, ctrl_state, share))
    return overlaps

def count_op_flags(trt_set):
    """Count operational flags for treatment states."""
    count = 0
    flag_details = []
    for s in trt_set:
        if s in operational_flags:
            for f in operational_flags[s]:
                # test_confound is severe, structural_break is moderate,
                # pre_period_event is minor
                if f["type"] == "test_confound":
                    count += 2  # double-weight test-period confounds
                elif f["type"] == "structural_break":
                    count += 1
                else:
                    count += 0.5  # half-weight pre-period events
                flag_details.append((s, f["type"], f["event"]))
    return count, flag_details

def evaluate_combination(trt_states):
    """Evaluate a treatment set against all hard constraints."""
    trt_set = frozenset(trt_states)
    n = len(trt_set)

    # Hard constraint: 2-4 treatment states
    if n < 2 or n > 4:
        return None

    # Hard constraint: all must be treatment-eligible
    if not trt_set.issubset(treatment_eligible):
        return None

    # Hard constraint: NC+SC atomicity
    nc_in = "NC" in trt_set
    sc_in = "SC" in trt_set
    if nc_in != sc_in:
        return None

    # Hard constraint: revenue share 12-25%
    trt_rev = sum(market_lookup[s]["audited_last_28d_revenue_usd"] for s in trt_set)
    rev_share = trt_rev / TOTAL_REV
    if rev_share < REV_SHARE_MIN or rev_share > REV_SHARE_MAX:
        return None

    # Hard constraint: budget
    spend = sum(incremental_spend(s) for s in trt_set)
    if spend > BUDGET_CAP:
        return None

    # Hard constraint: comparison pool >= 4
    ctrl_set = ALL_STATES - trt_set - excluded_entirely
    if len(ctrl_set) < MIN_COMPARISON:
        return None

    # --- Passed all hard constraints. Compute scoring metrics. ---

    abs_lift = 0.025 * trt_rev
    trt_ctrl_overlaps = get_trt_ctrl_overlaps(trt_set, ctrl_set)
    max_overlap = max((o[2] for o in trt_ctrl_overlaps), default=0.0)
    sum_overlap = sum(o[2] for o in trt_ctrl_overlaps)
    n_flags, flag_details = count_op_flags(trt_set)

    return {
        "treatment": sorted(trt_set),
        "treatment_label": "+".join(sorted(trt_set)),
        "n_treatment": n,
        "n_comparison": len(ctrl_set),
        "comparison": sorted(ctrl_set),
        "rev_share_pct": round(rev_share * 100, 2),
        "treatment_rev": round(trt_rev, 2),
        "incr_spend": round(spend, 2),
        "budget_remaining": round(BUDGET_CAP - spend, 2),
        "abs_lift_2_5pct": round(abs_lift, 2),
        "max_trt_ctrl_overlap": round(max_overlap * 100, 1),
        "sum_trt_ctrl_overlap": round(sum_overlap * 100, 1),
        "trt_ctrl_overlap_pairs": [(o[0], o[1], f"{o[2]*100:.1f}%") for o in trt_ctrl_overlaps],
        "n_op_flags": n_flags,
        "op_flag_details": flag_details,
        "has_nc_sc": nc_in and sc_in,
    }

# Generate all combinations
# Individual eligible states (excluding NC and SC, handled separately)
individual = sorted(treatment_eligible - {"NC", "SC"})

results = []

# Case 1: combinations without NC+SC
for k in range(2, 5):
    for combo in combinations(individual, k):
        r = evaluate_combination(combo)
        if r:
            results.append(r)

# Case 2: NC+SC pair + 0/1/2 additional states
for k_extra in range(0, 3):
    if k_extra == 0:
        r = evaluate_combination(["NC", "SC"])
        if r:
            results.append(r)
    else:
        for extra in combinations(individual, k_extra):
            combo = list(extra) + ["NC", "SC"]
            r = evaluate_combination(combo)
            if r:
                results.append(r)

print(f"\n  Total feasible combinations: {len(results)}")

# ============================================================
# STEP 4E: SCORE AND RANK
# ============================================================

print(f"\n--- 4E: Score and Rank ---")

# Pre-period fit proxy: compute how well each treatment aggregate
# correlates with the weighted comparison pool
# (True fit requires GeoX, but this is a reasonable proxy)

# Pivot daily revenue to wide format
rev_wide = daily.pivot(index="date", columns="state", values="net_revenue_usd")

def pre_period_fit_score(trt_states, ctrl_states):
    """
    Compute a proxy for synthetic control fit quality.
    Uses the R-squared of regressing the treatment aggregate
    on the top-5 correlated control states.
    """
    trt_agg = rev_wide[list(trt_states)].sum(axis=1)
    ctrl_df = rev_wide[list(ctrl_states)]

    # Get top-5 most correlated control states
    correlations = ctrl_df.corrwith(trt_agg).abs().sort_values(ascending=False)
    top5 = correlations.head(5).index.tolist()

    if len(top5) < 2:
        return 0.0

    # Simple OLS R-squared using numpy
    X = ctrl_df[top5].values
    y = trt_agg.values

    # Add intercept
    X_aug = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        y_hat = X_aug @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return max(0.0, r2)
    except Exception:
        return 0.0

print(f"  Computing pre-period fit scores for {len(results)} designs...")

for r in results:
    ctrl_for_fit = set(r["comparison"]) - excluded_entirely
    r["pre_period_r2"] = pre_period_fit_score(r["treatment"], ctrl_for_fit)

# Composite score
def compute_score(r):
    # Power: revenue share (higher = more signal)
    rev_score = (r["rev_share_pct"] - 12) / (25 - 12)
    rev_score = max(0, min(1, rev_score))

    # Overlap: lower = cleaner
    overlap_score = 1.0 - (r["max_trt_ctrl_overlap"] / 23.0)
    overlap_score = max(0, min(1, overlap_score))

    # Operational flags: fewer = better
    flag_score = max(0, 1.0 - (r["n_op_flags"] / 3.0))

    # Budget headroom: more = safer
    budget_score = r["budget_remaining"] / BUDGET_CAP

    # Donor pool: more = better synthetic control
    pool_score = (r["n_comparison"] - MIN_COMPARISON) / 10.0
    pool_score = max(0, min(1, pool_score))

    # Pre-period fit: higher R2 = better counterfactual
    fit_score = r["pre_period_r2"]

    # Weighted composite
    score = (
        0.25 * rev_score +
        0.20 * overlap_score +
        0.15 * flag_score +
        0.10 * budget_score +
        0.10 * pool_score +
        0.20 * fit_score
    )
    return round(score, 4)

for r in results:
    r["composite_score"] = compute_score(r)

results.sort(key=lambda r: r["composite_score"], reverse=True)

# ============================================================
# OUTPUT: TOP 30 + CATEGORY BESTS
# ============================================================

print(f"\n{'=' * 120}")
print(f"{'Rank':<5} {'Treatment':<24} {'#T':<4} {'RevShare%':<10} {'Spend':>10} "
      f"{'BudgetRem':>10} {'MaxOvlp%':>9} {'Flags':>6} {'#C':>4} {'FitR2':>6} {'Score':>7}")
print(f"{'=' * 120}")

for i, r in enumerate(results[:30]):
    print(f"{i+1:<5} {r['treatment_label']:<24} {r['n_treatment']:<4} "
          f"{r['rev_share_pct']:<10.2f} ${r['incr_spend']:>9,.0f} "
          f"${r['budget_remaining']:>9,.0f} {r['max_trt_ctrl_overlap']:>8.1f}% "
          f"{r['n_op_flags']:>5.1f} {r['n_comparison']:>4} "
          f"{r['pre_period_r2']:>5.3f} {r['composite_score']:>7.4f}")

# Category analysis
print(f"\n{'=' * 80}")
print(f"CATEGORY ANALYSIS")
print(f"{'=' * 80}")

categories = {
    "2-state": [r for r in results if r["n_treatment"] == 2],
    "3-state": [r for r in results if r["n_treatment"] == 3],
    "4-state": [r for r in results if r["n_treatment"] == 4],
    "Zero trt-ctrl overlap": [r for r in results if r["max_trt_ctrl_overlap"] == 0],
    "Zero op flags": [r for r in results if r["n_op_flags"] == 0],
    "Fully clean (0 overlap + 0 flags)": [
        r for r in results
        if r["max_trt_ctrl_overlap"] == 0 and r["n_op_flags"] == 0
    ],
}

for name, cat in categories.items():
    print(f"\n  {name}: {len(cat)} designs")
    if cat:
        best = cat[0]
        print(f"    Best: {best['treatment_label']} "
              f"(Score={best['composite_score']:.4f}, "
              f"RevShare={best['rev_share_pct']}%, "
              f"Overlap={best['max_trt_ctrl_overlap']}%, "
              f"Flags={best['n_op_flags']}, "
              f"FitR2={best['pre_period_r2']:.3f})")

# ============================================================
# DETAILED TOP 10
# ============================================================

print(f"\n{'=' * 80}")
print(f"DETAILED TOP 10")
print(f"{'=' * 80}")

for i, r in enumerate(results[:10]):
    print(f"\n--- Rank {i+1}: {r['treatment_label']} (Score: {r['composite_score']:.4f}) ---")
    print(f"  Treatment:        {r['treatment']}")
    print(f"  # Treatment:      {r['n_treatment']}")
    print(f"  Revenue share:    {r['rev_share_pct']}%")
    print(f"  Treatment rev:    ${r['treatment_rev']:,.2f}")
    print(f"  Incremental spend: ${r['incr_spend']:,.2f}")
    print(f"  Budget remaining: ${r['budget_remaining']:,.2f}")
    print(f"  Abs lift @2.5%:   ${r['abs_lift_2_5pct']:,.2f}")
    print(f"  # Comparison:     {r['n_comparison']}")
    print(f"  Comparison pool:  {r['comparison']}")
    print(f"  Max trt-ctrl overlap: {r['max_trt_ctrl_overlap']}%")
    if r['trt_ctrl_overlap_pairs']:
        print(f"  Overlap pairs:    {r['trt_ctrl_overlap_pairs']}")
    else:
        print(f"  Overlap pairs:    None")
    print(f"  Op flags:         {r['n_op_flags']}")
    if r['op_flag_details']:
        for s, ftype, event in r['op_flag_details']:
            print(f"    {s}: [{ftype}] {event}")
    print(f"  Pre-period R2:    {r['pre_period_r2']:.4f}")
    print(f"  NC+SC pair:       {'Yes' if r['has_nc_sc'] else 'No'}")

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame([{
    "rank": i + 1,
    "treatment": r["treatment_label"],
    "n_treatment": r["n_treatment"],
    "n_comparison": r["n_comparison"],
    "rev_share_pct": r["rev_share_pct"],
    "incr_spend": r["incr_spend"],
    "budget_remaining": r["budget_remaining"],
    "abs_lift_2_5pct": r["abs_lift_2_5pct"],
    "max_trt_ctrl_overlap": r["max_trt_ctrl_overlap"],
    "sum_trt_ctrl_overlap": r["sum_trt_ctrl_overlap"],
    "n_op_flags": r["n_op_flags"],
    "pre_period_r2": r["pre_period_r2"],
    "composite_score": r["composite_score"],
    "comparison": "+".join(r["comparison"]),
} for i, r in enumerate(results)])

results_df.to_csv(DATA_DIR / "feasible_designs_ranked.csv", index=False)

# Also save top 10 as a summary
results_df.head(10).to_csv(DATA_DIR / "top10_designs.csv", index=False)

print(f"\n{'=' * 80}")
print(f"STEP 4 COMPLETE")
print(f"{'=' * 80}")
print(f"  Total feasible: {len(results)}")
print(f"  Saved: feasible_designs_ranked.csv ({len(results)} rows)")
print(f"  Saved: top10_designs.csv")
print(f"\n  NEXT: Run GeoX/CausalImpact on the top 10 designs to get")
print(f"        MDE, power at 2.5%, and pre-period fit statistics.")
