import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Setup
df_old = pd.read_csv("local_files/dataset_report_stats_6.csv")
df_new = pd.read_csv("local_files/dataset_report_stats_7.csv")

# 2. Merge data on common classes
merged = pd.merge(
    df_old, df_new,
    on="morphos_name",
    suffixes=("_old", "_new")
)

# 3. Calculate Change in Recall
merged["Recall_Delta"] = merged["Recall_new"] - merged["Recall_old"]

# Categorize the change type for visual grouping
def categorize_change(x, threshold=0.02):
    if x > threshold: return "Improved (▲)"
    elif x < -threshold: return "Degraded (▼)"
    return "Stable (●)"

merged["Status"] = merged["Recall_Delta"].apply(categorize_change)

# 4. Bin by Total Samples (Using New Run's counts)
bins = [0, 30, 70, 130, 200]

# You must provide exactly 1 less label than the number of bins
labels = [
    '10-30 (Scarce)',
    '31-70 (Low-Mid)',
    '71-130 (Established)',
    '131-200 (Abundant)'
]
merged['Sample_Bin'] = pd.cut(merged['dataset_report_total_samples_new'], bins=bins, labels=labels, include_lowest=True)

# ==========================================
# VISUALIZATION 1: Interactive Delta Plot (Plotly)
# ==========================================
# This plots every single class.
# X-axis is log-scaled because sample sizes vary wildly.
fig = px.scatter(
    merged,
    x="dataset_report_total_samples_new",
    y="Recall_Delta",
    color="Status",
    hover_name="morphos_name",
    hover_data=["Recall_old", "Recall_new", "dataset_report_total_samples_new"],
    color_discrete_map={"Improved (▲)": "green", "Degraded (▼)": "red", "Stable (●)": "gray"},
    log_x=True,
    title="Recall Metric Shifts Across 1,200+ Classes",
    labels={"dataset_report_total_samples_new": "Class Sample Size (Log Scale)", "Recall_Delta": "Change in Recall (New - Old)"}
)
# Add a baseline horizontal line at zero change
fig.add_hline(y=0, line_dash="dash", line_color="black")
fig.show()

# ==========================================
# VISUALIZATION 2: Binned Distribution (Seaborn)
# ==========================================
# This groups classes into bins and shows where the bulk of changes happened.
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=merged,
    x="Sample_Bin",
    y="Recall_Delta",
    palette="vlag",
    hue="Sample_Bin",
    legend=False
)
plt.axhline(0, color='red', linestyle='--', alpha=0.7)
plt.title("Distribution of Recall Changes Grouped by Class Representation Size")
plt.ylabel("Recall Delta (New - Old)")
plt.xlabel("Images Per Class Bin")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
