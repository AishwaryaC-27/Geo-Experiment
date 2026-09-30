# Geo Experiment Design Report

## Overview

This repository contains the pre-test analysis and experiment design for a geo-based incrementality experiment for a fictional online retailer operating across **16 U.S. states**.

The objective is to determine whether a **40% increase in daily paid-social spend** can generate incremental **finance net revenue** compared with what would have occurred under business-as-usual (BAU).

The experiment is planned for **June 1–28, 2026 (28 days)**.

## Experiment Setup

- **Geographic units:** 16 U.S. states
- **Treatment:** 40% increase in paid-social spend
- **Test window:** Jun 1 – Jun 28, 2026
- **Treatment states:** 2–4 states
- **Treatment revenue share:** 12–25% of total audited revenue, **combined across treatment states**
- **Comparison group:** At least 4 suitable untreated states
- **Incremental budget:** Maximum $300,000
- **Primary metric:** Finance net revenue (`net_revenue_usd`)
- **Planning effect:** 2.5%

Existing advertising continues under BAU, while only the selected treatment states receive the additional 40% spend.

## Experiment Objective

The key question is:

> **Does the additional paid-social spend generate revenue that would not have occurred under BAU?**

This requires estimating a **BAU counterfactual** — the revenue that the selected treatment states would have generated during the test period if the additional advertising had not been applied.

The experiment therefore compares the observed treatment outcome with the estimated BAU counterfactual rather than relying only on a simple before-and-after comparison.

## Data and Analysis

Historical pre-test data from **January 12, 2026 to May 31, 2026** is used for experiment planning and design.

The analysis includes:

- Data cleaning and validation
- State-level revenue analysis
- Revenue-share assessment
- Incremental spend calculations
- Treatment eligibility checks
- Delivery-group constraints
- Audience-overlap analysis
- Revenue-volatility assessment
- Pre-period fit and counterfactual diagnostics
- Approximate minimum detectable effect (MDE) analysis
- Feasible treatment-design screening

## Design Constraints

Candidate treatment configurations must satisfy the required business constraints:

1. 2–4 treatment states
2. 12–25% **combined** treatment revenue share
3. Incremental spend ≤ $300,000
4. Treatment-state spending eligibility
5. Delivery-group consistency
6. At least 4 suitable comparison states

## Statistical Hypotheses

**H₀:** The 40% paid-social increase produces no incremental increase in finance net revenue relative to the BAU counterfactual.

**H₁:** The 40% paid-social increase produces a positive incremental increase in finance net revenue.

The **2.5% effect** is used as a planning/detection target and is not a guaranteed observed uplift.

## Tools

- Python
- Pandas
- NumPy
- Plotly
- GeoX
- Geo-experiment / counterfactual methodology

## Report

The repository includes an interactive HTML report with the main figures and supporting analysis used to assess experiment feasibility, statistical sensitivity, and potential risks.


