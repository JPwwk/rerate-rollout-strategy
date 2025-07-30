import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

st.set_page_config(layout='wide')
st.title("Manual Rerate Rollout Strategy Comparison Tool")

# --- INSTRUCTIONS AND PURPOSE ---
with st.expander("ℹ️ About this Tool", expanded=True):
    st.markdown("""
    This tool helps you compare the impact of different state-level rerate rollout strategies.

    ### 🎯 Purpose:
    Estimate how much of the total rerate activity is considered **manual rerate** under different rollout plan assignments and timing scenarios.

    ### 🛠️ How to Use:
    - Assign U.S. states into rollout groups (R1–R10) for both original and proposed plans
    - Choose rollout dates by quarter
    - Adjust the **growth/reduction modifiers** to reflect plan assumptions
    - Set **Lookback Capture Weights** to determine how fast rerates are realized manually
    - View changes in the output rerate projections and risk indicator (KRI)

    ### 📊 Assumptions:
    - Rerates are distributed over 8 years post-rollout (weight-controlled)
    - Each state gets a **single rollout date**
    - For app performance improvements, app utilizes standard deviation to inject variance in simulation
    - Confidence intervals are based on user-selected bands
    - Simulation isolates only the **manual** portion of rerate projections
    - **Simulation does not take into consideration any costs associated with conducting the manual rerate**

    ---
    """)
    
# --- LOAD DATA ---

@st.cache_data
def load_data():
    return pd.read_excel(
        "Random number generation.xlsx",
        sheet_name="Core_Data",
        engine="openpyxl"
    )

df = load_data()

# --- ROLLOUT GROUP ASSIGNMENT ---

st.subheader("Rollout Group Assignment")

# Get all states
all_states = sorted(df['State'].unique().tolist())
group_names = [f'R{i}' for i in range(1, 11)]  # R1 through R10

# --- DEFAULT ASSIGNMENTS TO PRELOAD ---

default_original_assignments = {
    'R1': ['WI'],
    'R2': ['AZ', 'FL', 'IL', 'LA', 'MO', 'MN', 'OH', 'TN', 'TX'],
    'R3': ['CT', 'IN', 'KS', 'KY', 'MD', 'MI', 'NY', 'OR', 'SC', 'VA', 'UT'],
    'R4': ['AL', 'AR', 'DE', 'GA', 'ID', 'NC', 'OK', 'RI', 'CO', 'NJ'],
    'R5': ['DC', 'IA', 'MS', 'NM', 'NV', 'PA', 'WV', 'CA', 'HI', 'WA', 'MA', 'VI'],
    'R6': ['AK', 'ME', 'MT', 'ND', 'NE', 'SD', 'VT', 'WY', 'NH'],
    'R7': [],
    'R8': [],
    'R9': [],
    'R10': []
}

default_proposed_assignments = {
    'R1': ['WI'],
    'R2': ['NC', 'MD', 'CO', 'AZ', 'VA', 'CT', 'IN', 'SD'],
    'R3': ['TX', 'KS', 'IL', 'GA', 'LA', 'AK', 'NH', 'NE', 'ND'],
    'R4': ['MA', 'SC', 'WA', 'AR', 'WY', 'NM', 'ID', 'RI'],
    'R5': ['NY', 'NV', 'MS', 'TN', 'OK', 'DC', 'ME'],
    'R6': ['MI', 'PA', 'OH', 'CA', 'AL', 'DE', 'FL', 'HI', 'IA', 'KY', 'MN', 'MO', 'MT', 'NJ', 'OR', 'UT', 'VT', 'VI', 'WV'],
    'R7': [],
    'R8': [],
    'R9': [],
    'R10': []
}
# Initialize Streamlit session state with default assignments
if "assignments" not in st.session_state:
    st.session_state["assignments"] = default_original_assignments.copy()

if "proposed_assignments" not in st.session_state:
    st.session_state["proposed_assignments"] = default_proposed_assignments.copy()
    
# SAFELY initialize assignments dictionary
    
# This works around notebook/runtime issues
session_assignments = st.session_state["assignments"]

assigned_states = set()
user_assignments = {}
cols = st.columns(5)

for i, group in enumerate(group_names):
    with cols[i % 5]:
        remaining = sorted(set(all_states).difference(assigned_states).union(session_assignments[group]))
        selected = st.multiselect(f"{group} States", options=remaining, default=session_assignments[group], key=group)
        user_assignments[group] = selected
        session_assignments[group] = selected
        assigned_states.update(selected)

# Show missing states
unassigned = set(all_states) - assigned_states
if unassigned:
    st.warning(f"Unassigned States: {', '.join(sorted(unassigned))}")
else:
    st.success("All states assigned.")
    
# --- SECOND PLAN: PROPOSED ROLLOUT GROUP ASSIGNMENT ---

st.subheader("Proposed Rollout Plan")

# Safely store second plan assignments (session-safe)
if "proposed_assignments" not in st.__dict__:
    st.proposed_assignments = {g: [] for g in group_names}

proposed_session = st.session_state["proposed_assignments"]

proposed_assigned = set()
proposed_assignments = {}
cols2 = st.columns(5)

for i, group in enumerate(group_names):
    with cols2[i % 5]:
        # Include current selection to prevent deselection conflicts
        remaining = sorted(set(all_states).difference(proposed_assigned).union(proposed_session[group]))
        selected = st.multiselect(
            f"{group} (Proposed)",
            options=remaining,
            default=proposed_session[group],
            key=f"proposed_{group}"
        )
        proposed_assignments[group] = selected
        proposed_session[group] = selected
        proposed_assigned.update(selected)

# Warn if any are unassigned in proposed
unassigned2 = set(all_states) - proposed_assigned
if unassigned2:
    st.warning(f" Proposed Plan Missing States: {', '.join(sorted(unassigned2))}")
else:
    st.success(" Proposed plan fully assigned.")
    
  # --- ROLLOUT DATES ASSIGNMENT ---
  
st.markdown("""
### 📅 Rollout Group Dates

Each rollout group (R1–R10) is assigned a launch quarter, which sets the start date for rerate capture in each state. From that date forward:

- Manual rerate exposure is calculated over an 8-year capture window.
- Year 1 begins the quarter following rollout.
- Timing has a direct effect on the size and duration of manual exposure.
- If not using a group, leave it at N/A

**Use these inputs to test timing strategies for risk reduction**.
""")

st.subheader("📅 Rollout Group Dates")

# Define quarter options from Q1 2026 to Q4 2030, plus 'N/A'
quarters = [f"Q{q} {year}" for year in range(2026, 2031) for q in range(1, 5)]
quarters.insert(0, "N/A")

# Initialize session state if needed
if "selected_rollout_dates" not in st.session_state:
    st.session_state.selected_rollout_dates = {f"R{i}": "N/A" for i in range(1, 11)}

# Layout into 5 columns (2 rows x 5)
cols = st.columns(5)
for i, col in enumerate(cols * 2):  # Repeat cols to make 10
    group = f"R{i+1}"
    st.session_state.selected_rollout_dates[group] = col.selectbox(
        f"R{i+1}",
        options=quarters,
        index=quarters.index(st.session_state.selected_rollout_dates.get(group, "N/A")),
        key=f"rollout_date_{group}"
    )
        
# --- CLEANUP ---

df['Manual_Rerate_Risk'] = df['Member Count'] * df['Average refund']

st.markdown("""
### 🧮 Manual Rerate Inputs: Growth / Reduction Modifiers

These annual modifiers allow you to adjust refund projections based on expected business trends, operational changes, or policy assumptions.

- A value of **1.00** means no change from baseline.
- A value above 1.00 (e.g., 1.05) reflects **growth** in refund exposure that year.
- A value below 1.00 (e.g., 0.95) reflects a **reduction** in exposure, potentially due to improved controls or automation.

Changes here scale the refund total each year, allowing for more realistic forecasting of future risk under your rollout plan.
""")

# --- Modifier Sliders ---

st.markdown("**Yearly Modifier (Growth/Reduction Factor):**")
modifiers = {}
mod_cols = st.columns(4)
for i, year in enumerate(range(2030, 2038)):
    with mod_cols[i % 4]:
        modifiers[year] = st.slider(
            f"{year}",
            min_value=0.5,
            max_value=1.5,
            step=0.01,
            value=1.0,
        )
        
# --- Rerate Weights ---

st.subheader("📈 Lookback Capture Weights")

st.markdown("**Adjust how much of a the total rerate is considered manual**")
st.markdown("""
    - Using historical analysis we see that for each year of data you increase the percentage of captured rerates, reducing the risk of manual rerates. 
    - These weights model the percentage of captured rerates (cumulative) by year and are default set at the historical average. 
    - When adjusting this note the values must sum to ~1.0. 
    
    -*For example, if you expect most of the rerates to be captured in the first few years, use heavier weights early on.*
""")

rerate_weights = {}
weight_cols = st.columns(4)
for i in range(1, 9):  # Year 1 to Year 8
    with weight_cols[(i - 1) % 4]:
        rerate_weights[i] = st.number_input(
            f"Weight for Year {i}", min_value=0.0, max_value=1.0, step=0.01,
            value=[0.4, 0.3, 0.14, 0.08, 0.04, 0.02, 0.01, 0.01][i - 1]
        )

# Show cumulative capture summary
st.markdown("### 📈 Cumulative Capture Summary")
cum_sum = 0
for i in range(1, 9):
    cum_sum += rerate_weights.get(i, 0)
    st.markdown(f"- Year {i}: **{cum_sum:.0%}**")
            
# Normalize (optional safety)
total_weight = sum(rerate_weights.values())
if abs(total_weight - 1.0) > 0.01:
    st.warning(f" Rerate weights sum to {total_weight:.2f}. Consider adjusting.")
    
# --- SIMULATION ---
def simulate_rollout(assignments, df, rollout_dates, rerate_weights, modifiers, z_score=1.96):
    df = df.copy()
    df['Rollout_Group'] = 'Unassigned'
    df['Rollout_Date'] = pd.NaT

    # Map rollout dates to each state
    for group, states in assignments.items():
        for state in states:
            df.loc[df['State'] == state, 'Rollout_Group'] = group
            df.loc[df['State'] == state, 'Rollout_Date'] = rollout_dates[group]

    projection_years = list(range(2030, 2043))  # Extend through 2042
    results = []

    for year in projection_years:
        total = 0
        variance = 0

        for _, row in df.iterrows():
            rollout = row['Rollout_Date']
            if pd.isnull(rollout):
                continue

            years_out = (year - rollout.year) + 1
            if years_out < 1 or years_out > 8:
                continue  # Outside lookback window

            weight = rerate_weights.get(years_out, 0)
            modifier = modifiers.get(year, 1.0)

            members = row['Member Count']
            avg = row['Average refund']
            std = row['Stdev refunds']
            rmin = row['Min Refund']
            rmax = row['Max Refund']

            # Sample from normal with bounds
            sampled_refund = np.clip(np.random.normal(loc=avg, scale=std), rmin, rmax)
            sampled_refund = max(sampled_refund, 0)

            manual = members * sampled_refund * weight * modifier
            total += manual

            # Use adjusted std dev for confidence interval
            variance += (members * sampled_refund * weight * modifier) ** 2

        std_total = np.sqrt(variance)
        results.append({
            'Year': year,
            'Manual_Rerate': total,
            'Lower_Bound': max(0, total - z_score * std_total),
            'Upper_Bound': total + z_score * std_total
        })

    return pd.DataFrame(results)
        
# --- STATIC INPUTS (PLACEHOLDERS FOR NOW) ---

rollout_dates = {
    'R1': datetime(2026, 1, 30),
    'R2': datetime(2027, 1, 30),
    'R3': datetime(2027, 4, 30),
    'R4': datetime(2027, 10, 30),
    'R5': datetime(2028, 1, 30),
    'R6': datetime(2028, 7, 30),
}

rerate_weights = {i+1: v for i, v in enumerate([0.4, 0.3, 0.14, 0.08, 0.04, 0.02, 0.01, 0.01])}
modifiers = {year: 1.0 for year in range(2030, 2038)}

# --- TEMP STATIC GROUPINGS (SAME FOR BOTH) ---

states = df['State'].tolist()
equal_groups = [states[i::6] for i in range(6)]

# --- OPTIONAL: Upload a saved rollout plan from file ---

st.subheader("📤 Upload Rollout Plan (Optional)")

uploaded_file = st.file_uploader("Upload saved plan (.csv with columns: State, Rollout_Group)", type="csv")

if uploaded_file:
    plan_df = pd.read_csv(uploaded_file)
    st.write("📋 Uploaded Plan Preview:")
    st.dataframe(plan_df)

    # Overwrite user_assignments if valid structure
    if {'State', 'Rollout_Group'}.issubset(plan_df.columns):
        uploaded_assignments = {g: [] for g in group_names}
        for _, row in plan_df.iterrows():
            uploaded_assignments[row['Rollout_Group']].append(row['State'])

        user_assignments = uploaded_assignments
        st.success(" Rollout plan loaded successfully and applied.")
        
original_assignments = user_assignments
proposed_assignments = proposed_assignments

# --- CONFIDENCE INTERVAL SELECTION (UI near chart only) ---

confidence_choice = st.selectbox(
    "Confidence Interval (affects chart bands):",
    options=["90% Confidence", "95% Confidence", "99% Confidence"],
    index=1,
)

# Match the selected label to a z-score and short label
confidence_map = {
    "90% Confidence": (1.64, "90% CI"),
    "95% Confidence": (1.96, "95% CI"),
    "99% Confidence": (2.58, "99% CI")
}
z_score, conf_label = confidence_map[confidence_choice]

# --- RUN SIMULATION ---

original_df = simulate_rollout(original_assignments, df, rollout_dates, rerate_weights, modifiers, z_score)
proposed_df = simulate_rollout(proposed_assignments, df, rollout_dates, rerate_weights, modifiers, z_score)

    # --- Build Comparison ---
    
comparison_df = original_df[['Year', 'Manual_Rerate']].merge(
proposed_df[['Year', 'Manual_Rerate']],
on='Year',
suffixes=('_Original', '_Proposed')
)
comparison_df['Delta ($)'] = comparison_df['Manual_Rerate_Original'] - comparison_df['Manual_Rerate_Proposed']
comparison_df['Delta (%)'] = (comparison_df['Delta ($)'] / comparison_df['Manual_Rerate_Original']) * 100
comparison_df['Delta (%)'] = comparison_df['Delta (%)'].map('{:.1f}%'.format)

    # --- Chart ---
    
st.subheader("Projection: Manual Rerate Totals (Original vs Proposed)")

import plotly.graph_objects as go

st.subheader("Manual Rerate Projection with Confidence Bands")

# Used for chart band size and dynamic labeling in the plot

fig = go.Figure()

#  Add Original Plan: Confidence Band and Mean Estimate

# 1. Upper Bound (first, invisible trace for top line)
fig.add_trace(go.Scatter(
    x=original_df['Year'],
    y=original_df['Upper_Bound'],
    name=f"Original Upper Bound ({conf_label})",
    line=dict(width=0),
    showlegend=False,
    hovertemplate="Year: %{x}<br>Original Upper Bound: $%{y:,.0f}<extra></extra>"
))

# 2. Lower Bound (second, filled to upper)
fig.add_trace(go.Scatter(
    x=original_df['Year'],
    y=original_df['Lower_Bound'],
    name=f"Original Lower Bound ({conf_label})",
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0, 0, 255, 0.1)',
    showlegend=True,
    hovertemplate="Year: %{x}<br>Original Lower Bound: $%{y:,.0f}<extra></extra>"
))

# 3. Mean Estimate (shown on top)
fig.add_trace(go.Scatter(
    x=original_df['Year'],
    y=original_df['Manual_Rerate'],
    mode="lines+markers",
    name="Original Plan",
    line=dict(color="blue"),
    hovertemplate="Year: %{x}<br>Original Mean Estimate: $%{y:,.0f}<extra></extra>"
))

# Proposed Plan: Confidence Band and Mean Estimate

# 1. Proposed Upper Bound (invisible trace for fill base)
fig.add_trace(go.Scatter(
    x=proposed_df['Year'],
    y=proposed_df['Upper_Bound'],
    name=f"Proposed Upper Bound ({conf_label})",
    line=dict(width=0),
    showlegend=False,
    hovertemplate="Year: %{x}<br>Proposed Upper Bound: $%{y:,.0f}<extra></extra>"
))

# 2. Proposed Lower Bound (fills to upper)
fig.add_trace(go.Scatter(
    x=proposed_df['Year'],
    y=proposed_df['Lower_Bound'],
    name=f"Proposed Lower Bound ({conf_label})",
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(255, 165, 0, 0.1)',  # Light orange
    showlegend=True,
    hovertemplate="Year: %{x}<br>Proposed Lower Bound: $%{y:,.0f}<extra></extra>"
))

# 3. Proposed Mean Estimate (on top of fill)
fig.add_trace(go.Scatter(
    x=proposed_df['Year'],
    y=proposed_df['Manual_Rerate'],
    mode="lines+markers",
    name="Proposed Plan",
    line=dict(color="orange"),
    hovertemplate="Year: %{x}<br>Proposed Mean Estimate: $%{y:,.0f}<extra></extra>"
))

fig.update_layout(
    yaxis_title="Manual Rerate ($)",
    xaxis_title="Year",
    hovermode="x unified",
    title=f"Manual Rerate Projection with {conf_label}",
)

st.plotly_chart(fig, use_container_width=True)

# --- RESULTS TABLE ---

st.subheader("Delta Summary: Key Risk Indicator (KRI)")
formatted_df = comparison_df.copy()

# Format Year column as plain strings (after summary row is added)
comparison_df["Year"] = comparison_df["Year"].astype(str)

# Format dollar columns with $ and commas
for col in ["Manual_Rerate_Original", "Manual_Rerate_Proposed", "Delta ($)"]:
    formatted_df[col] = pd.to_numeric(comparison_df[col], errors="coerce").apply(
        lambda x: f"${x:,.0f}" if pd.notnull(x) else ""
    )

# Format Delta (%) as percentage with 1 decimal place
formatted_df["Delta (%)"] = (
    pd.to_numeric(comparison_df["Delta ($)"], errors="coerce") /
    pd.to_numeric(comparison_df["Manual_Rerate_Original"], errors="coerce") * 100
).apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")

# Calculate totals and append as a summary row
totals = {
    "Year": "Total",
    "Manual_Rerate_Original": "${:,.0f}".format(
        pd.to_numeric(comparison_df["Manual_Rerate_Original"], errors="coerce").sum()
    ),
    "Manual_Rerate_Proposed": "${:,.0f}".format(
        pd.to_numeric(comparison_df["Manual_Rerate_Proposed"], errors="coerce").sum()
    ),
    "Delta ($)": "${:,.0f}".format(
        pd.to_numeric(comparison_df["Delta ($)"], errors="coerce").sum()
    ),
}

# Compute overall percent delta
original_total = pd.to_numeric(comparison_df["Manual_Rerate_Original"], errors="coerce").sum()
delta_total = pd.to_numeric(comparison_df["Delta ($)"], errors="coerce").sum()

if original_total != 0:
    totals["Delta (%)"] = "{:.1f}%".format((delta_total / original_total) * 100)
else:
    totals["Delta (%)"] = "N/A"

# Append to formatted_df
formatted_df.loc[len(formatted_df)] = totals

# Display formatted DataFrame
st.dataframe(formatted_df, use_container_width=True)

st.subheader("📁 Download Plan Results")

csv = comparison_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download KRI Summary as CSV",
    data=csv,
    file_name='kri_comparison_summary.csv',
    mime='text/csv',
)

# --- FOOTER ---

st.caption("Next: Add user interactivity for plan assignments, weight adjustments, and export.")
