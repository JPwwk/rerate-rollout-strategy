import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

st.set_page_config(layout='wide')
st.title("Manual Rerate Rollout Strategy Comparison Tool")

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

st.subheader("🧩 Rollout Group Assignment")

# Get all states
all_states = sorted(df['State'].unique().tolist())
group_names = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']

# --- DEFAULT ASSIGNMENTS TO PRELOAD ---

default_original_assignments = {
    'R1': ['WI'],
    'R2': ['AZ', 'FL', 'IL', 'LA', 'MO', 'MN', 'OH', 'TN', 'TX'],
    'R3': ['CT', 'IN', 'KS', 'KY', 'MI', 'NY', 'OR', 'SC', 'VA', 'UT'],
    'R4': ['AL', 'AR', 'DE', 'GA', 'ID', 'MD', 'NC', 'OK', 'RI', 'VI', 'CO', 'NJ'],
    'R5': ['DC', 'IA', 'MS', 'NM', 'NV', 'PA', 'WV', 'CA', 'HI', 'WA', 'MA'],
    'R6': ['AK', 'ME', 'MT', 'ND', 'NE', 'SD', 'VT', 'WY', 'NH']
}

default_proposed_assignments = {
    'R1': ['WI'],
    'R2': ['NC', 'MD', 'CO', 'AZ', 'VA', 'CT', 'IN', 'SD'],
    'R3': ['TX', 'KS', 'IL', 'GA', 'LA', 'AK', 'NH', 'NE', 'ND'],
    'R4': ['MA', 'SC', 'WA', 'AR', 'WY', 'NM', 'ID', 'RI'],
    'R5': ['NY', 'NV', 'MS', 'TN', 'OK', 'DC', 'ME'],
    'R6': ['MI', 'PA', 'OH', 'CA', 'AL', 'DE', 'FL', 'HI', 'IA', 'KY', 'MN', 'MO', 'MT', 'NJ', 'OR', 'UT', 'VT', 'VI', 'WV', 'WY']
}

# Initialize Streamlit session state with default assignments
if "assignments" not in st.session_state:
    st.session_state["assignments"] = default_original_assignments.copy()

if "proposed_assignments" not in st.session_state:
    st.session_state["proposed_assignments"] = default_proposed_assignments.copy()
    
# ✅ SAFELY initialize assignments dictionary
    
# This works around notebook/runtime issues
session_assignments = st.session_state["assignments"]

assigned_states = set()
user_assignments = {}
cols = st.columns(3)

for i, group in enumerate(group_names):
    with cols[i % 3]:
        remaining = sorted(set(all_states).difference(assigned_states).union(session_assignments[group]))
        selected = st.multiselect(f"{group} States", options=remaining, default=session_assignments[group], key=group)
        user_assignments[group] = selected
        session_assignments[group] = selected
        assigned_states.update(selected)

# Show missing states
unassigned = set(all_states) - assigned_states
if unassigned:
    st.warning(f"⚠️ Unassigned States: {', '.join(sorted(unassigned))}")
else:
    st.success("✅ All states assigned.")

# --- SECOND PLAN: PROPOSED ROLLOUT GROUP ASSIGNMENT ---

st.subheader("🟠 Proposed Rollout Plan")

# Safely store second plan assignments (session-safe)
if "proposed_assignments" not in st.__dict__:
    st.proposed_assignments = {g: [] for g in group_names}

proposed_session = st.session_state["proposed_assignments"]

proposed_assigned = set()
proposed_assignments = {}
cols2 = st.columns(3)

for i, group in enumerate(group_names):
    with cols2[i % 3]:
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
    st.warning(f"🟠 Proposed Plan Missing States: {', '.join(sorted(unassigned2))}")
else:
    st.success("🟠 Proposed plan fully assigned.")
    
# --- CLEANUP ---

df['Manual_Rerate_Risk'] = df['Member Count'] * df['Average refund']

st.subheader("📉 Manual Rerate Inputs")

# --- Modifier Sliders ---

# --- Confidence Interval Selection ---
st.markdown("**Confidence Interval (for Bands):**")
z_choice = st.selectbox("Z-score (affects width of confidence bands)", options={
    "90% (±1.64)": 1.64,
    "95% (±1.96)": 1.96,
    "99% (±2.58)": 2.58
})
z_score = float(z_choice.split("±")[1][:-1])  # extracts the Z number

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

st.markdown("**Rerate Weighting by Year Since Rollout (must sum to ~1.0):**")
rerate_weights = {}
weight_cols = st.columns(4)
for i in range(1, 9):
    with weight_cols[i % 4]:
        rerate_weights[i] = st.number_input(
            f"Year {i}", min_value=0.0, max_value=1.0, step=0.01, value=[0.4, 0.3, 0.14, 0.08, 0.04, 0.02, 0.01, 0.01][i - 1]
        )

# Normalize (optional safety)
total_weight = sum(rerate_weights.values())
if abs(total_weight - 1.0) > 0.01:
    st.warning(f"⚠️ Rerate weights sum to {total_weight:.2f}. Consider adjusting.")

# --- SIMULATION PLACEHOLDER ---

def simulate_rollout(assignments, df, rollout_dates, rerate_weights, modifiers, z_score=1.96):
    df = df.copy(deep=True)
    df['Rollout_Group'] = 'Unassigned'
    df['Rollout_Date'] = pd.NaT

    for group, states in assignments.items():
        for state in states:
            df.loc[df['State'] == state, 'Rollout_Group'] = group
            df.loc[df['State'] == state, 'Rollout_Date'] = rollout_dates[group]

    projection_years = list(range(2030, 2038))
    results = []

    for year in projection_years:
        total_manual_rerate = 0
        variance_sum = 0

        for _, row in df.iterrows():
            if pd.isnull(row['Rollout_Date']):
                continue
            rollout = row['Rollout_Date']
            capture_end_year = rollout.year + 8
            if year > capture_end_year - 1:
                continue

            years_out = (year - rollout.year) + 1
            if years_out < 1 or years_out > 8:
                continue

            weight = rerate_weights.get(years_out, 0)
            modifier = modifiers.get(year, 1.0)

            base_rerate = row['Member Count'] * row['Average refund']
            manual_rerate = base_rerate * weight * modifier
            std_dev = (row['Member Count'] * row['Stdev refunds'] * weight * modifier) ** 2

            total_manual_rerate += manual_rerate
            variance_sum += std_dev

        std_total = np.sqrt(variance_sum)
        results.append({
            'Year': year,
            'Manual_Rerate': total_manual_rerate,
            'Lower_Bound': max(0, total_manual_rerate - z_score * std_total),
            'Upper_Bound': total_manual_rerate + z_score * std_total
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
        st.success("✅ Rollout plan loaded successfully and applied.")
        
original_assignments = user_assignments
proposed_assignments = proposed_assignments

# --- RUN SIMULATION ---

original_df = simulate_rollout(original_assignments, df, rollout_dates, rerate_weights, modifiers, z_score)
proposed_df = simulate_rollout(proposed_assignments, df, rollout_dates, rerate_weights, modifiers, z_score)

# --- BUILD COMPARISON ---

comparison_df = original_df[['Year', 'Manual_Rerate']].merge(
    proposed_df[['Year', 'Manual_Rerate']],
    on='Year',
    suffixes=('_Original', '_Proposed')
)
comparison_df['Delta ($)'] = comparison_df['Manual_Rerate_Original'] - comparison_df['Manual_Rerate_Proposed']
comparison_df['Delta (%)'] = (comparison_df['Delta ($)'] / comparison_df['Manual_Rerate_Original']) * 100
comparison_df['Delta (%)'] = comparison_df['Delta (%)'].map('{:.1f}%'.format)

# --- CHART ---

st.subheader("Projection: Manual Rerate Totals (Original vs Proposed)")

import plotly.graph_objects as go

st.subheader("📊 Manual Rerate Projection with Confidence Bands")

Z_choice = st.selectbox(
    "Z-score (affects width of confidence bands)",
    options=[
        "90% (±1.64)",
        "95% (±1.96)",
        "99% (±2.58)"
    ],
    index=1
)
z_score = float(Z_choice.split("±")[1][:-1])  # pulls out 1.64, 1.96, or 2.58

# Extract CI label for legend
conf_label = f"{int(float(Z_choice.split('%')[0]))}% CI"

fig = go.Figure()

# Add Original Plan
fig.add_trace(go.Scatter(
    x=original_df['Year'],
    y=original_df['Manual_Rerate'],
    mode='lines+markers',
    name='Original Plan',
    line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=original_df['Year'],
    y=original_df['Upper_Bound'],
    name=f'Original {conf_label}',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=original_df['Year'],
    y=original_df['Lower_Bound'],
    name=f'Original {conf_label}',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(0, 0, 255, 0.1)',
    showlegend=True
))

# Add Proposed Plan
fig.add_trace(go.Scatter(
    x=proposed_df['Year'],
    y=proposed_df['Manual_Rerate'],
    mode='lines+markers',
    name='Proposed Plan',
    line=dict(color='orange')
))
fig.add_trace(go.Scatter(
    x=proposed_df['Year'],
    y=proposed_df['Upper_Bound'],
    name=f'Proposed {conf_label}',
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=proposed_df['Year'],
    y=proposed_df['Lower_Bound'],
    name=f'Proposed {conf_label}',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(255, 165, 0, 0.1)',
    showlegend=True
))

fig.update_layout(
    yaxis_title="Manual Rerate ($)",
    xaxis_title="Year",
    hovermode="x unified",
    title=f"Manual Rerate Projection with {conf_label}",
)

st.plotly_chart(fig, use_container_width=True)
# --- TABLE ---

st.subheader("Delta Summary: Key Risk Indicator (KRI)")
st.dataframe(comparison_df, use_container_width=True)

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