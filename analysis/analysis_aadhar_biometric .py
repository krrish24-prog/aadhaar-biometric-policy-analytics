import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# STEP 0: LOAD MERGED DATASET
# ==============================

file_path = r"C:\Users\k manoj\Downloads\aadhaar_biometric_FULL.csv"

print("📥 Loading dataset...")
df = pd.read_csv(file_path)

print("✅ Loaded successfully")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ==============================
# STEP 1: RENAME COLUMNS
# ==============================

df = df.rename(columns={
    "bio_age_5_17": "age_5_to_17_updates",
    "bio_age_17_": "age_17_plus_updates"
})

# ==============================
# STEP 2: FIX DATE COLUMN
# ==============================

df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

# ==============================
# STEP 3: TOTAL BIOMETRIC UPDATES
# ==============================

df["total_biometric_updates"] = (
    df["age_5_to_17_updates"] + df["age_17_plus_updates"]
)

# ==============================
# STEP 4: PERCENTAGES
# ==============================

df["pct_age_5_to_17"] = (
    df["age_5_to_17_updates"] / df["total_biometric_updates"] * 100
)

df["pct_age_17_plus"] = (
    df["age_17_plus_updates"] / df["total_biometric_updates"] * 100
)

# ==============================
# STEP 5: CLEAN DATA
# ==============================

df = df[df["total_biometric_updates"] > 0]

print("\n🧹 Cleaned dataset")
print("Final shape:", df.shape)

# ==============================
# STEP 6: STATE SUMMARY (PRINTED)
# ==============================

state_summary = (
    df.groupby("state", as_index=False)
    .agg({
        "age_5_to_17_updates": "sum",
        "age_17_plus_updates": "sum",
        "total_biometric_updates": "sum"
    })
    .sort_values("total_biometric_updates", ascending=False)
)

print("\n📊 Top 10 States by Biometric Updates:")
print(state_summary.head(10))

# ==============================
# STEP 7: DISTRICT HOTSPOTS (PRINTED)
# ==============================

district_hotspots = (
    df.groupby(["state", "district"], as_index=False)
    .agg({"total_biometric_updates": "sum"})
    .sort_values("total_biometric_updates", ascending=False)
)

print("\n🔥 Top 10 District Hotspots:")
print(district_hotspots.head(10))

# ==============================
# CHART 1: STATE-WISE TOTAL UPDATES
# ==============================

plt.figure(figsize=(10, 7))
plt.barh(
    state_summary["state"].head(15),
    state_summary["total_biometric_updates"].head(15)
)
plt.xlabel("Total Biometric Updates")
plt.title("State-wise Aadhaar Biometric Updates (Top 15)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("chart_1_state_updates.png")
plt.show()

# ==============================
# CHART 2: TOP 10 DISTRICTS
# ==============================

top_10 = district_hotspots.head(10)

labels = (
    top_10["district"].astype(str)
    + " ("
    + top_10["state"].astype(str)
    + ")"
)

plt.figure(figsize=(10, 7))
plt.barh(labels, top_10["total_biometric_updates"])
plt.xlabel("Total Biometric Updates")
plt.title("Top 10 District Hotspots")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("chart_2_top_districts.png")
plt.show()

# ==============================
# CHART 3: AGE GROUP DISTRIBUTION
# ==============================

age_totals = df[
    ["age_5_to_17_updates", "age_17_plus_updates"]
].sum()

print("\n👶🧑 Age-wise Totals:")
print(age_totals)

plt.figure(figsize=(7, 7))
plt.pie(
    age_totals,
    labels=["Age 5–17", "Age 17+"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Age-wise Distribution of Biometric Updates")
plt.tight_layout()
plt.savefig("chart_3_age_distribution.png")
plt.show()

# ==============================
# CHART 4: TIME TREND
# ==============================

date_trend = (
    df.groupby("date", as_index=False)
    .agg({"total_biometric_updates": "sum"})
)

print("\n📈 Time Trend Preview:")
print(date_trend.head())

plt.figure(figsize=(10, 6))
plt.plot(
    date_trend["date"],
    date_trend["total_biometric_updates"],
    linewidth=2
)
plt.xlabel("Date")
plt.ylabel("Total Updates")
plt.title("Time Trend of Aadhaar Biometric Updates")
plt.tight_layout()
plt.savefig("chart_4_time_trend.png")
plt.show()

# ==============================
# CHART 5: TOP VS BOTTOM STATES
# ==============================

comparison = pd.concat([
    state_summary.head(5),
    state_summary.tail(5)
])

print("\n⚖️ Top vs Bottom States:")
print(comparison)

plt.figure(figsize=(10, 6))
plt.bar(
    comparison["state"],
    comparison["total_biometric_updates"]
)
plt.xticks(rotation=45)
plt.ylabel("Total Updates")
plt.title("Top vs Bottom States – Biometric Update Inequality")
plt.tight_layout()
plt.savefig("chart_5_top_vs_bottom.png")
plt.show()

# ==============================
# CHART 6: STATE-WISE HEATMAP (FIXED)
# ==============================

pivot_heatmap = (
    state_summary
    .set_index("state")[["total_biometric_updates"]]
)

plt.figure(figsize=(8, 12))

im = plt.imshow(
    pivot_heatmap.values,
    aspect="auto",
    cmap="YlOrRd"
)

plt.colorbar(im, label="Total Biometric Updates")

plt.yticks(
    range(len(pivot_heatmap.index)),
    pivot_heatmap.index.tolist()
)

plt.xticks([0], ["Total Updates"])

plt.title("State-wise Heatmap of Aadhaar Biometric Updates")
plt.tight_layout()
plt.savefig("chart_6_state_heatmap.png")
plt.show()

print("\n✅ Analysis + charts generation completed successfully")
