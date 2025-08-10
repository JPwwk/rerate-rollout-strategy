# streamlit_app.py
# Manual Rerate Rollout Strategy Comparison Tool

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Manual Rerate Rollout Strategy Comparison Tool")

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_excel(
        "Random number generation.xlsx",
        sheet_name="Core_Data",
        engine="openpyxl",
    )

df = load_data().copy()

# Derived columns from the workbook
df["Avg Refund"] = df["Sum Refunds (Inverted)"] / df["Count of Valid Refunds"]
df["Manual_Rerate_Risk"] = df["Member Count"] * df["Avg Refund"]

ALL_STATES: list[str] = sorted(df["State"].dropna().unique().tolist())

# Projection window and quarter list
PROJ_YEARS = list(range(2026, 2043))  # 2026–2042 inclusive
QUARTERS = [f"Q{q} {y}" for y in range(2026, 2033) for q in range(1, 5)]  # 2026–2032

# -----------------------------------------------------------------------------
# Top-level explainer (exec friendly)
# -----------------------------------------------------------------------------
with st.expander("What this simulator does (read me first)", expanded=True):
    st.markdown("""
**Goal.** Compare two state roll‑out plans (Original vs Proposed) and estimate the **manual rerate** impact over time.

**How it works.**
- A state contributes **zero** manual rerates until the quarter its system conversion completes.
- After conversion, that state’s manual rerates are spread across the **next 8 years** using the **Lookback Weights** (defaults reflect our historical analysis of prior migrations).
- You can optionally scale yearly activity with **Growth / Reduction Modifiers**. These apply to **both manual dollars and manual counts** in that calendar year.
- View a **deterministic** (no randomness) or **Monte Carlo** (uncertainty bands) run in **Simulation Mode**.

**What “manual rerate” means here.**
- **Manual $**: estimated dollars rerated manually (not total rerates, not claim spend).
- **Manual Count**: estimated number of **manual rerate occurrences**.

**Disclaimer**
- Texas UMB incident has been excluded from this analysis as an outlier.
- Ops costs/savings are **not** modeled.

**Weights clarity.**
- If **Year 1 weight = 40%**, that means **40% of the rerates are captured in the first year after conversion** and **60% remain** for later years. The **cumulative** list shows percent captured **to‑date** by the end of each year.

**How to read results.**
- **Key Results** show totals for each plan and **Δ (Original − Proposed)**. The banner directly below states who is worse and by how much **in dollars and counts**.
- The chart shows yearly profiles with confidence bands.
- The table shows per‑year details and bounds (dollars always; counts if enabled).

**Defaults / data.**
- Lookback Weights default to **42%, 30%, 14%, 6%, 4%, 2%, 1%, 1%** (≈100%) — from **historical analysis** of when manual rerates surface post‑conversion.
- Modifiers default to **1.00** (no change).
""")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def quarter_to_date(qstr: str) -> datetime:
    """Convert 'Q3 2029' -> datetime(2029, 7, 1)."""
    q, year = qstr.split()
    month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[q]
    return datetime(int(year), month, 1)

def normalize_unique_assignments(assignments: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Keep each state at most once; the earliest quarter wins (for simulation math).
    We DO NOT mutate the visible widgets after render; we normalize only for computation.
    """
    seen: set[str] = set()
    cleaned: dict[str, list[str]] = {}
    for q in QUARTERS:
        current = [s for s in assignments.get(q, []) if s in ALL_STATES]
        out = []
        for s in current:
            if s not in seen:
                out.append(s)
                seen.add(s)
        cleaned[q] = sorted(out)
    return cleaned

def assignments_complete(assignments: dict[str, list[str]]) -> tuple[bool, set[str], set[str]]:
    flat = [s for v in assignments.values() for s in v]
    missing = set(ALL_STATES) - set(flat)
    dups = {s for s in flat if flat.count(s) > 1}
    return len(missing) == 0 and len(dups) == 0, missing, dups

def assignments_to_dates(assignments: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for q, states in assignments.items():
        for s in states:
            rows.append({"State": s, "Rollout_Group": q, "Rollout_Date": quarter_to_date(q)})
    return pd.DataFrame(rows)

CL_TO_PCT = {"90% Confidence": (5.0, 95.0), "95% Confidence": (2.5, 97.5), "99% Confidence": (0.5, 99.5)}

def human_money(x: float) -> str:
    """$12.5M style for hover."""
    n = float(x)
    sgn = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e9:  return f"{sgn}${n/1e9:.1f}B"
    if n >= 1e6:  return f"{sgn}${n/1e6:.1f}M"
    if n >= 1e3:  return f"{sgn}${n/1e3:.1f}K"
    return f"{sgn}${n:.0f}"

# -----------------------------------------------------------------------------
# Simulation (deterministic or Monte Carlo)
# -----------------------------------------------------------------------------
def simulate_rollout(
    base_df: pd.DataFrame,
    assignments: dict[str, list[str]],
    weights: dict[int, float],
    modifiers: dict[int, float],
    z_score: float,
    deterministic: bool,
    n_sims: int = 1000,
    seed: int = 42,
    start_year: int = 2026,
    end_year: int = 2042,
    ignore_before_year: int | None = None,
    return_sims: bool = False,
    sample_counts: bool = False,  # if True, we also randomize counts
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute manual rerate dollars and counts by year.

    - deterministic=True: expected values (no randomness).
    - deterministic=False: Monte Carlo using bounded normal for refund amounts.
    - ignore_before_year: if set, zero-out any year < this.
    - return_sims: when Monte Carlo, also return per-simulation matrices (dollars, counts).
    - sample_counts: when Monte Carlo, also vary counts (to produce count bands).
    """
    # Use first-occurrence plan for math
    plan_for_math = normalize_unique_assignments(assignments)

    a_df = assignments_to_dates(plan_for_math)
    df_local = base_df.merge(a_df, on="State", how="left")

    rollout_year_series = pd.to_datetime(df_local["Rollout_Date"]).dt.year
    valid_idx = rollout_year_series.dropna().index.to_numpy()
    rollout_years = rollout_year_series.fillna(-1).to_numpy(dtype=int)

    years = list(range(start_year, end_year + 1))
    sims = 1 if deterministic else int(max(1, n_sims))
    rng = np.random.default_rng(seed if not deterministic else 12345)

    totals_dollars = np.zeros((sims, len(years)), dtype=float)
    totals_counts  = np.zeros((sims, len(years)), dtype=float)

    members = df_local["Member Count"].to_numpy(dtype=float)
    avg_refund = df_local["Avg Refund"].to_numpy(dtype=float)
    std_refund = df_local["Stdev refunds"].to_numpy(dtype=float)
    min_refund = df_local["Min Refund"].to_numpy(dtype=float)
    max_refund = df_local["Max Refund"].to_numpy(dtype=float)
    baseline_annual_counts = (df_local["Count of Valid Refunds"].to_numpy(dtype=float) / 5.0)

    for si in range(sims):
        for yi, year in enumerate(years):
            if ignore_before_year is not None and year < int(ignore_before_year):
                continue

            total_dol = 0.0
            total_cnt = 0.0

            for i in valid_idx:
                ryear = rollout_years[i]
                if ryear < 0:
                    continue

                years_out = (year - int(ryear)) + 1  # 1..8
                if years_out < 1 or years_out > 8:
                    continue

                w = float(weights.get(years_out, 0.0))
                m = float(modifiers.get(year, 1.0))

                if deterministic:
                    refund = float(np.clip(avg_refund[i], min_refund[i], max_refund[i]))
                    cnt = baseline_annual_counts[i]
                else:
                    refund = float(
                        np.clip(rng.normal(loc=avg_refund[i], scale=std_refund[i]),
                                min_refund[i], max_refund[i])
                    )
                    refund = max(refund, 0.0)

                    if sample_counts:
                        # Poisson around the baseline annual count (>=0), then scale by weights & modifiers
                        lam = max(baseline_annual_counts[i], 0.0)
                        cnt = float(rng.poisson(lam=lam))
                    else:
                        cnt = baseline_annual_counts[i]

                total_dol += members[i] * refund * w * m
                total_cnt += cnt * w * m

            totals_dollars[si, yi] = total_dol
            totals_counts[si, yi]  = total_cnt

    # Per-year means for table/chart defaults
    mean_dol = totals_dollars.mean(axis=0)
    mean_cnt = totals_counts.mean(axis=0)
    std_dol  = totals_dollars.std(axis=0, ddof=1) if sims > 1 else np.zeros_like(mean_dol)

    lower_dol = np.maximum(0.0, mean_dol - z_score * std_dol)
    upper_dol = mean_dol + z_score * std_dol

    result_df = pd.DataFrame(
        {
            "Year": years,
            "Manual_Rerate": mean_dol,
            "Manual_Count":  mean_cnt,
            "Lower_Bound":   lower_dol,
            "Upper_Bound":   upper_dol,
            # place-holders for counts; caller will fill when sample_counts=True
            "Count_Lower":   mean_cnt,
            "Count_Upper":   mean_cnt,
        }
    )

    if return_sims and (not deterministic):
        return result_df, totals_dollars, totals_counts
    else:
        return result_df

# -----------------------------------------------------------------------------
# Defaults used to seed both plans
# -----------------------------------------------------------------------------
DEFAULT_QUARTERS: dict[str, list[str]] = {
    "Q3 2026": ["WI"],
    "Q3 2028": ["OH", "TX"],
    "Q2 2029": ["AZ", "IA", "IL", "IN", "MN", "MS", "TN", "UT"],
    "Q3 2029": ["AL", "AR", "ID", "KS", "ME", "MO", "ND", "OK", "WV"],
    "Q4 2029": ["AK", "CT", "DC", "NM", "NV", "OR", "RI", "SC", "WY"],
    "Q1 2030": ["DE", "KY", "NE", "SD", "VT"],
    "Q2 2030": ["CA", "MT", "NH", "NJ"],
    "Q3 2030": ["FL", "GA", "LA", "MD", "PA", "VA", "WA"],
    "Q4 2030": ["CO", "HI"],
    "Q1 2031": ["NY", "MI"],
    "Q2 2031": ["NC"],
    "Q3 2031": ["MA"],
}
for q in QUARTERS:
    DEFAULT_QUARTERS.setdefault(q, [])

# -----------------------------------------------------------------------------
# Plan editors (duplicate call-outs; no post-render mutation)
# -----------------------------------------------------------------------------
def render_plan_editor(state_key: str, title: str, defaults: dict[str, list[str]]) -> dict[str, list[str]]:
    if state_key not in st.session_state:
        st.session_state[state_key] = {q: list(defaults.get(q, [])) for q in QUARTERS}

    plan = st.session_state[state_key]

    with st.expander(title, expanded=False):
        st.markdown("""
**Assign each state to exactly one conversion quarter.**
- A state contributes **only after** its quarter.
- If a state is selected more than once, the **earliest** quarter will be used in the math. We’ll call out duplicates below.
""")

        cols = st.columns(4)
        for i, q in enumerate(QUARTERS):
            with cols[i % 4]:
                plan[q] = st.multiselect(
                    q,
                    options=ALL_STATES,
                    default=[s for s in plan.get(q, []) if s in ALL_STATES],
                    key=f"{state_key}_{q}",
                )

        # Validation summaries (without mutating the widgets)
        ok, missing, dups = assignments_complete(plan)
        if not ok:
            if missing:
                st.warning(f"Unassigned ({title.split()[0]}): " + ", ".join(sorted(missing)))
            if dups:
                st.error("Duplicate selections: " + ", ".join(sorted(dups)) + " — using the **earliest** quarter for each in calculations.")
        else:
            st.success(f"{title.split()[0]} plan: all states assigned exactly once (no duplicates).")

    return plan

# Build plans BEFORE modifiers/weights/sim
original_plan = render_plan_editor("original_quarters", "Original State Launch Plan (Quarterly)", DEFAULT_QUARTERS)
proposed_plan = render_plan_editor("proposed_quarters", "Proposed State Launch Plan (Quarterly)", DEFAULT_QUARTERS)

# -----------------------------------------------------------------------------
# Modifiers and weights
# -----------------------------------------------------------------------------
with st.expander("📈 Growth / Reduction Modifiers by Year", expanded=False):
    st.markdown("""
**What this does.** Multiplies **manual rerate dollars and counts** in that calendar year **for all active states**.

- **1.00** = no change (baseline)
- **1.10** = +10% manual activity in that year
- **0.90** = −10% manual activity in that year

> These are activity assumptions (not membership growth or claim inflation).
""")
    mods_cols = st.columns(4)
    modifiers: dict[int, float] = {}
    for i, year in enumerate(PROJ_YEARS):
        with mods_cols[i % 4]:
            modifiers[year] = st.slider(
                str(year), 0.50, 1.50, 1.00, 0.01,
                help="Scalar applied to manual rerate **dollars and counts** in this calendar year."
            )

with st.expander("🧮 Lookback Capture Weights (8‑year roll‑off)", expanded=False):
    st.markdown("""
**What this does.** Defines how a state’s manual rerates are spread over the **8 years after conversion**.

- **Interpretation:** If **Year 1 weight = 42%**, that means **42% of the lifetime rerates are captured in the first year after conversion** and **60% remain** to be captured later.  
- The **cumulative list below** shows the percent captured **to‑date** by the end of each post‑conversion year.
- Weights should sum to about **1.00** across 8 years (≈100% of a state’s lifetime manual opportunity).

> Defaults (from historical analysis): 42%, 30%, 14%, 6%, 4%, 2%, 1%, 1%.
""")
    default_weights = [0.42, 0.30, 0.14, 0.06, 0.04, 0.02, 0.01, 0.01]
    weights_cols = st.columns(4)
    weights: dict[int, float] = {}
    for i in range(1, 9):
        with weights_cols[(i - 1) % 4]:
            weights[i] = st.number_input(
                f"Year {i} weight", 0.0, 1.0, default_weights[i - 1], 0.01,
                help=f"Share of manual rerates captured in **year {i} after conversion**."
            )
    st.markdown("**Cumulative capture:**")
    cum = 0.0
    for i in range(1, 9):
        cum += weights[i]
        st.markdown(f"- Year {i}: **{cum:.0%}**")
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_w:.2f}. Consider adjusting to 1.00.")

# -----------------------------------------------------------------------------
# Simulation controls
# -----------------------------------------------------------------------------
st.markdown("### Simulation Mode")
st.caption("""
- **Deterministic** = expected values only (no randomness) to validate math.
- **Monte Carlo** = adds uncertainty by sampling refund amounts within historical min/max; bands show your chosen confidence interval.
- **Random seed** = keeps Monte Carlo runs reproducible.
- **Ignore contributions before year** = reporting cutoff (zeroes out earlier years).
""")

ctrl = st.columns(5)
with ctrl[0]:
    deterministic = st.checkbox(
        "Use deterministic expected values (no randomness) to validate math",
        value=True,
        help="Turn off randomness. Useful for math checks and predictable totals."
    )
with ctrl[1]:
    n_sims = st.slider(
        "Monte Carlo runs", 100, 5000, 1000, 100,
        disabled=deterministic,
        help="More runs = smoother bands but slower."
    )
with ctrl[2]:
    seed = st.number_input(
        "Random seed", 0, 10_000_000, 42, 1,
        disabled=deterministic,
        help="Fixes the random number generator so results are reproducible."
    )
with ctrl[3]:
    ignore_before_year = st.selectbox(
        "Ignore contributions before year",
        options=[None] + list(range(2026, 2043)),
        index=0,
        format_func=lambda x: "—" if x is None else str(x),
        help="Zero out **all** dollars and counts before this year (viewing/decision cutoff)."
    )
with ctrl[4]:
    sample_counts = st.checkbox(
        "Sample counts in Monte Carlo (adds bounds for counts)",
        value=False,
        help="If checked, counts vary stochastically; table and hover show count bands."
    )

confidence_choice = st.selectbox(
    "Confidence Interval (affects chart, KPIs, and table):",
    options=["90% Confidence", "95% Confidence", "99% Confidence"],
    index=1,
)

center_choice = st.selectbox(
    "Center statistic (Monte Carlo only):",
    options=["Median (recommended)", "Mean", "Lower bound center", "Upper bound center"],
    index=0,
    help="Which 'center' to use for KPIs and the line on the chart when Monte Carlo is on."
)

# map selected center to function over a vector
lo_p, hi_p = CL_TO_PCT[confidence_choice]
def center_from_vec(v: np.ndarray) -> float:
    if center_choice.startswith("Median"):
        return float(np.percentile(v, 50.0))
    if center_choice == "Mean":
        return float(np.mean(v))
    if center_choice == "Lower bound center":
        return float(np.percentile(v, lo_p))
    if center_choice == "Upper bound center":
        return float(np.percentile(v, hi_p))
    return float(np.mean(v))

# -----------------------------------------------------------------------------
# Run simulations (2026–2042) – return per-sim matrices to compute centers/CL
# -----------------------------------------------------------------------------
orig_out = simulate_rollout(
    df, original_plan, weights, modifiers,
    z_score=1.96,  # per-year shading uses z-score internally; totals use percentiles below
    deterministic=deterministic,
    n_sims=n_sims, seed=seed,
    start_year=2026, end_year=2042,
    ignore_before_year=ignore_before_year,
    return_sims=True,
    sample_counts=sample_counts,
)
prop_out = simulate_rollout(
    df, proposed_plan, weights, modifiers,
    z_score=1.96,
    deterministic=deterministic,
    n_sims=n_sims, seed=seed,
    start_year=2026, end_year=2042,
    ignore_before_year=ignore_before_year,
    return_sims=True,
    sample_counts=sample_counts,
)

if deterministic:
    original_df, orig_sims_dol, orig_sims_cnt = orig_out, None, None
    proposed_df, prop_sims_dol, prop_sims_cnt = prop_out, None, None
else:
    original_df, orig_sims_dol, orig_sims_cnt = orig_out
    proposed_df, prop_sims_dol, prop_sims_cnt = prop_out

# For per‑year **center** series (lines on chart), compute from the per‑sim matrices
if not deterministic:
    # per‑year vectors across sims
    years = original_df["Year"].to_numpy()
    o_center_series = np.array([center_from_vec(orig_sims_dol[:, i]) for i in range(orig_sims_dol.shape[1])])
    p_center_series = np.array([center_from_vec(prop_sims_dol[:, i]) for i in range(prop_sims_dol.shape[1])])
    original_df["Manual_Rerate_Center"] = o_center_series
    proposed_df["Manual_Rerate_Center"] = p_center_series
else:
    original_df["Manual_Rerate_Center"] = original_df["Manual_Rerate"]
    proposed_df["Manual_Rerate_Center"] = proposed_df["Manual_Rerate"]

# Per‑year count bounds if enabled
if (not deterministic) and sample_counts:
    original_df["Count_Lower"] = np.percentile(orig_sims_cnt, lo_p, axis=0)
    original_df["Count_Upper"] = np.percentile(orig_sims_cnt, hi_p, axis=0)
    proposed_df["Count_Lower"] = np.percentile(prop_sims_cnt, lo_p, axis=0)
    proposed_df["Count_Upper"] = np.percentile(prop_sims_cnt, hi_p, axis=0)

# -----------------------------------------------------------------------------
# KPI totals (center & CI) and Comparison build
# -----------------------------------------------------------------------------
def totals_from_df(df_: pd.DataFrame, use_center: bool) -> float:
    col = "Manual_Rerate_Center" if use_center else "Manual_Rerate"
    return float(df_[col].sum())

if deterministic:
    tot_orig_dollars = totals_from_df(original_df, use_center=False)
    tot_prop_dollars = totals_from_df(proposed_df, use_center=False)
    tot_orig_counts = float(original_df["Manual_Count"].sum())
    tot_prop_counts = float(proposed_df["Manual_Count"].sum())
    tot_orig_lower = tot_orig_upper = tot_orig_dollars
    tot_prop_lower = tot_prop_upper = tot_prop_dollars
else:
    # totals per sim (dollars)
    o_tot_d = orig_sims_dol.sum(axis=1)
    p_tot_d = prop_sims_dol.sum(axis=1)

    # KPI centers based on selected center statistic
    tot_orig_dollars = center_from_vec(o_tot_d)
    tot_prop_dollars = center_from_vec(p_tot_d)

    # KPI CL ranges from percentiles at the **total** level
    def ci(v): return (float(np.percentile(v, lo_p)), float(np.percentile(v, hi_p)))
    tot_orig_lower, tot_orig_upper = ci(o_tot_d)
    tot_prop_lower, tot_prop_upper = ci(p_tot_d)

    # counts: if sampling, center = chosen center; otherwise deterministic mean
    if sample_counts:
        o_tot_c = orig_sims_cnt.sum(axis=1)
        p_tot_c = prop_sims_cnt.sum(axis=1)
        tot_orig_counts = center_from_vec(o_tot_c)
        tot_prop_counts = center_from_vec(p_tot_c)
    else:
        tot_orig_counts = float(original_df["Manual_Count"].sum())
        tot_prop_counts = float(proposed_df["Manual_Count"].sum())

# Build comparison table base (align years)
comparison = (
    original_df.rename(columns=lambda c: f"{c}_Original" if c != "Year" else c)
    .merge(proposed_df.rename(columns=lambda c: f"{c}_Proposed" if c != "Year" else c), on="Year")
)

# Use center series for chart/summary per year
comparison["Manual_Rerate_Original"] = comparison["Manual_Rerate_Center_Original"]
comparison["Manual_Rerate_Proposed"] = comparison["Manual_Rerate_Center_Proposed"]

# deltas
comparison["Delta ($)"] = comparison["Manual_Rerate_Original"] - comparison["Manual_Rerate_Proposed"]
comparison["Delta Count"] = comparison["Manual_Count_Original"] - comparison["Manual_Count_Proposed"]

# Optionally hide years where both plans are zero
mask_nonzero = ~(
    (comparison["Manual_Rerate_Original"].round(0) == 0)
    & (comparison["Manual_Rerate_Proposed"].round(0) == 0)
    & (comparison["Manual_Count_Original"].round(0) == 0)
    & (comparison["Manual_Count_Proposed"].round(0) == 0)
)
comparison = comparison.loc[mask_nonzero].reset_index(drop=True)

# Count bounds aligned to shown years
year_idx = comparison["Year"]
orig_cnt_low_map  = dict(zip(original_df["Year"], original_df["Count_Lower"]))
orig_cnt_up_map   = dict(zip(original_df["Year"], original_df["Count_Upper"]))
prop_cnt_low_map  = dict(zip(proposed_df["Year"], proposed_df["Count_Lower"]))
prop_cnt_up_map   = dict(zip(proposed_df["Year"], proposed_df["Count_Upper"]))
comparison["Count_Lower_Original"] = year_idx.map(orig_cnt_low_map)
comparison["Count_Upper_Original"] = year_idx.map(orig_cnt_up_map)
comparison["Count_Lower_Proposed"] = year_idx.map(prop_cnt_low_map)
comparison["Count_Upper_Proposed"] = year_idx.map(prop_cnt_up_map)

# -----------------------------------------------------------------------------
# Key results + banner
# -----------------------------------------------------------------------------
delta_dollars = tot_orig_dollars - tot_prop_dollars
delta_counts = tot_orig_counts - tot_prop_counts
delta_pct = (delta_dollars / tot_orig_dollars * 100.0) if tot_orig_dollars else 0.0

st.markdown("### Key Results (Totals)")
k1, k2, k3, k4, k5, k6 = st.columns(6)

center_label = "median" if center_choice.startswith("Median") else (
    "mean" if center_choice == "Mean" else ("lower‑bound" if "Lower" in center_choice else "upper‑bound")
)

with k1:
    st.metric(
        f"Original – Manual $ ({center_label})",
        f"${tot_orig_dollars:,.0f}",
        delta=f"{confidence_choice}: ${tot_orig_lower:,.0f} – ${tot_orig_upper:,.0f}",
    )
with k2:
    st.metric(
        f"Proposed – Manual $ ({center_label})",
        f"${tot_prop_dollars:,.0f}",
        delta=f"{confidence_choice}: ${tot_prop_lower:,.0f} – ${tot_prop_upper:,.0f}",
    )
with k3:
    st.metric(
        "Δ Manual $ (Orig − Prop)",
        f"${delta_dollars:,.0f}",
        delta=f"{delta_pct:.1f}%",
        delta_color="inverse" if delta_dollars > 0 else "normal",
    )
with k4:
    st.metric(f"Original – Manual Count ({center_label})", f"{tot_orig_counts:,.0f}")
with k5:
    st.metric(f"Proposed – Manual Count ({center_label})", f"{tot_prop_counts:,.0f}")
with k6:
    st.metric("Δ Count (Orig − Prop)", f"{delta_counts:,.0f}",
              help="Positive = Original has more manual rerate count (worse).")

banner_color = "#ffe5e5" if delta_dollars > 0 else ("#e8f6e8" if delta_dollars < 0 else "#f6f6f6")
banner_text = (
    f"<b>Original plan is worse</b> — ${abs(delta_dollars):,.0f} more manual dollars "
    f"(<b>{abs(delta_pct):.1f}% higher</b>) and {abs(delta_counts):,.0f} more manual counts."
    if delta_dollars > 0 else
    f"<b>Proposed plan is worse</b> — ${abs(delta_dollars):,.0f} more manual dollars "
    f"(<b>{abs(delta_pct):.1f}% higher</b>) and {abs(delta_counts):,.0f} more manual counts."
    if delta_dollars < 0 else
    "Both plans have identical total manual dollars; compare per‑year patterns."
)
st.markdown(
    f"""
    <div style="background:{banner_color};border:1px solid #ddd;border-radius:10px;
                padding:10px 14px;margin:6px 0;">
        {banner_text}
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Chart (clean drilldown: one $ and one #; hide band hovers)
# -----------------------------------------------------------------------------
st.subheader("Projection: Manual Rerate Totals (Original vs Proposed)")
st.caption("Moving a state earlier shifts dollars forward in time but the lifetime total only changes if you use Modifiers or a year cutoff.")

# Precompute customdata (counts) so hover shows clean whole numbers
orig_counts = comparison["Manual_Count_Original"].round(0).astype(int).to_numpy()
prop_counts = comparison["Manual_Count_Proposed"].round(0).astype(int).to_numpy()

fig = go.Figure()

# --- Original bands (no hover) ---
fig.add_trace(go.Scatter(
    x=comparison["Year"], y=comparison["Upper_Bound_Original"],
    name=f"Original Upper Bound ({confidence_choice})",
    line=dict(width=0), showlegend=False, hoverinfo="skip"
))
fig.add_trace(go.Scatter(
    x=comparison["Year"], y=comparison["Lower_Bound_Original"],
    name=f"Original Lower Bound ({confidence_choice})",
    line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 0, 255, 0.10)",
    hoverinfo="skip"
))

# --- Proposed bands (no hover) ---
fig.add_trace(go.Scatter(
    x=comparison["Year"], y=comparison["Upper_Bound_Proposed"],
    name=f"Proposed Upper Bound ({confidence_choice})",
    line=dict(width=0), showlegend=False, hoverinfo="skip"
))
fig.add_trace(go.Scatter(
    x=comparison["Year"], y=comparison["Lower_Bound_Proposed"],
    name=f"Proposed Lower Bound ({confidence_choice})",
    line=dict(width=0), fill="tonexty", fillcolor="rgba(255, 165, 0, 0.10)",
    hoverinfo="skip"
))

# --- Original line (concise hover) ---
fig.add_trace(go.Scatter(
    x=comparison["Year"],
    y=comparison["Manual_Rerate_Original"],
    mode="lines+markers",
    name="Original Plan",
    line=dict(color="blue"),
    customdata=np.stack([orig_counts], axis=-1),
    # $ with SI units (e.g., 12.5M) + count with commas
    hovertemplate="<b>%{x}</b><br>Original: $%{y:.3s}<br>Count: %{customdata[0]:,}<extra></extra>"
))

# --- Proposed line (concise hover) ---
fig.add_trace(go.Scatter(
    x=comparison["Year"],
    y=comparison["Manual_Rerate_Proposed"],
    mode="lines+markers",
    name="Proposed Plan",
    line=dict(color="orange"),
    customdata=np.stack([prop_counts], axis=-1),
    hovertemplate="<b>%{x}</b><br>Proposed: $%{y:.3s}<br>Count: %{customdata[0]:,}<extra></extra>"
))

fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Manual Rerate ($)",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# Table (explicit headers + legend + totals)
# -----------------------------------------------------------------------------
st.subheader("Delta Summary: Key Risk Indicator (KRI)")
st.caption(
    f"Legend (Bounds shown at {confidence_choice}). In Monte Carlo, centers are {center_label}."
    "\n\n- **Manual $ / Manual Count** = estimated manual rerate dollars / occurrences."
    "\n- **Lower / Upper** = confidence bounds from Monte Carlo."
    "\n- **Δ Manual $ / Δ Count** = Original − Proposed (positive = Original worse)."
    "\n- **Δ % (vs Orig)** = Δ Manual $ as a % of Original in that year."
)

table_df = comparison.copy().rename(columns={
    "Manual_Rerate_Original":  "Manual $ (Orig)",
    "Manual_Count_Original":   "Manual Count (Orig)",
    "Lower_Bound_Original":    "Lower $ (Orig)",
    "Upper_Bound_Original":    "Upper $ (Orig)",
    "Manual_Rerate_Proposed":  "Manual $ (Prop)",
    "Manual_Count_Proposed":   "Manual Count (Prop)",
    "Lower_Bound_Proposed":    "Lower $ (Prop)",
    "Upper_Bound_Proposed":    "Upper $ (Prop)",
    "Delta ($)":               "Δ Manual $",
    "Delta Count":             "Δ Count",
})

# Count bounds via map (aligned)
table_df["Lower Cnt (Orig)"] = table_df["Count_Lower_Original"]
table_df["Upper Cnt (Orig)"] = table_df["Count_Upper_Original"]
table_df["Lower Cnt (Prop)"] = table_df["Count_Lower_Proposed"]
table_df["Upper Cnt (Prop)"] = table_df["Count_Upper_Proposed"]

# Round counts to whole numbers
for c in ["Manual Count (Orig)", "Manual Count (Prop)", "Δ Count",
          "Lower Cnt (Orig)", "Upper Cnt (Orig)", "Lower Cnt (Prop)", "Upper Cnt (Prop)"]:
    table_df[c] = table_df[c].round(0).astype(int)

# Δ % vs Orig (avoid div/0)
with np.errstate(divide="ignore", invalid="ignore"):
    pct = (comparison["Delta ($)"] / np.where(comparison["Manual_Rerate_Original"] != 0,
                                              comparison["Manual_Rerate_Original"], np.nan) * 100.0)
table_df["Δ % (vs Orig)"] = pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Totals row
totals_row = {
    "Year": "Total",
    "Manual $ (Orig)": comparison["Manual_Rerate_Original"].sum(),
    "Manual $ (Prop)": comparison["Manual_Rerate_Proposed"].sum(),
    "Manual Count (Orig)": int(comparison["Manual_Count_Original"].sum()),
    "Manual Count (Prop)": int(comparison["Manual_Count_Proposed"].sum()),
    "Lower $ (Orig)": comparison["Lower_Bound_Original"].sum(),
    "Upper $ (Orig)": comparison["Upper_Bound_Original"].sum(),
    "Lower $ (Prop)": comparison["Lower_Bound_Proposed"].sum(),
    "Upper $ (Prop)": comparison["Upper_Bound_Proposed"].sum(),
    "Lower Cnt (Orig)": int(np.nansum(table_df["Lower Cnt (Orig)"])),
    "Upper Cnt (Orig)": int(np.nansum(table_df["Upper Cnt (Orig)"])),
    "Lower Cnt (Prop)": int(np.nansum(table_df["Lower Cnt (Prop)"])),
    "Upper Cnt (Prop)": int(np.nansum(table_df["Upper Cnt (Prop)"])),
    "Δ Manual $": (comparison["Manual_Rerate_Original"] - comparison["Manual_Rerate_Proposed"]).sum(),
    "Δ Count": int((comparison["Manual_Count_Original"] - comparison["Manual_Count_Proposed"]).sum()),
    "Δ % (vs Orig)": (
        (comparison["Manual_Rerate_Original"] - comparison["Manual_Rerate_Proposed"]).sum()
        / comparison["Manual_Rerate_Original"].sum() * 100.0
        if comparison["Manual_Rerate_Original"].sum() else 0.0
    ),
}
table_df.loc[len(table_df)] = totals_row

ordered_cols = [
    "Year",
    "Manual $ (Orig)", "Manual Count (Orig)", "Lower $ (Orig)", "Upper $ (Orig)",
    "Lower Cnt (Orig)", "Upper Cnt (Orig)",
    "Manual $ (Prop)", "Manual Count (Prop)", "Lower $ (Prop)", "Upper $ (Prop)",
    "Lower Cnt (Prop)", "Upper Cnt (Prop)",
    "Δ Manual $", "Δ Count", "Δ % (vs Orig)",
]
table_df = table_df[ordered_cols]

orig_cols = ["Manual $ (Orig)", "Manual Count (Orig)", "Lower $ (Orig)", "Upper $ (Orig)", "Lower Cnt (Orig)", "Upper Cnt (Orig)"]
prop_cols = ["Manual $ (Prop)", "Manual Count (Prop)", "Lower $ (Prop)", "Upper $ (Prop)", "Lower Cnt (Prop)", "Upper Cnt (Prop)"]

sty = (
    table_df.style
    .set_properties(subset=orig_cols, **{"background-color": "#ecf1ff"})
    .set_properties(subset=prop_cols, **{"background-color": "#eaf6ea"})
    .format({
        "Manual $ (Orig)": "${:,.0f}",
        "Manual $ (Prop)": "${:,.0f}",
        "Lower $ (Orig)": "${:,.0f}",
        "Upper $ (Orig)": "${:,.0f}",
        "Lower $ (Prop)": "${:,.0f}",
        "Upper $ (Prop)": "${:,.0f}",
        "Δ Manual $": "${:,.0f}",
        "Manual Count (Orig)": "{:,}",
        "Manual Count (Prop)": "{:,}",
        "Lower Cnt (Orig)": "{:,}",
        "Upper Cnt (Orig)": "{:,}",
        "Lower Cnt (Prop)": "{:,}",
        "Upper Cnt (Prop)": "{:,}",
        "Δ Count": "{:,}",
        "Δ % (vs Orig)": "{:.1f}%",
    })
)

table_height = int((len(table_df) + 2) * 34)
st.dataframe(sty, use_container_width=True, height=table_height)

# -----------------------------------------------------------------------------
# Download (raw numeric values)
# -----------------------------------------------------------------------------
st.subheader("📁 Download Plan Results")
st.download_button(
    label="Download KRI Summary as CSV",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name="kri_comparison_summary.csv",
    mime="text/csv",
)
