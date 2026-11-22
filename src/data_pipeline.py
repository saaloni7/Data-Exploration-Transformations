import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def run_pipeline():
    raw_path = os.path.join("week4", "data", "raw_insurance.csv")
    cleaned_path = os.path.join("week4", "data", "cleaned_insurance.csv")
    vis_dir = os.path.join("week4", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print("="*70)
    print("WEEK 4: DATA EXPLORATION & TRANSFORMATIONS PIPELINE")
    print("Dataset: Medical Cost Personal Datasets (Kaggle)")
    print("Source: https://www.kaggle.com/datasets/mirichoi0218/insurance")
    print("="*70)

    # 1. Load Raw Dataset
    df_raw = pd.read_csv(raw_path)
    print(f"\n[Step 1] Ingested raw dataset with shape: {df_raw.shape}")
    print("First 5 records:")
    print(df_raw.head())

    # 2. Data Inspection
    print("\n[Step 2] Data Inspection:")
    print(f" -> Missing values across columns:\n{df_raw.isnull().sum()}")
    num_duplicates = df_raw.duplicated().sum()
    print(f" -> Number of duplicate rows detected: {num_duplicates}")
    if num_duplicates > 0:
        dup_rows = df_raw[df_raw.duplicated(keep=False)]
        print(f" -> Duplicate row details:\n{dup_rows}")

    # 3. Data Cleaning
    df_cleaned = df_raw.drop_duplicates().copy()
    print(f"\n[Step 3] Removed duplicates. New shape: {df_cleaned.shape}")

    # Correct Data Types
    categorical_cols = ['sex', 'smoker', 'region']
    for col in categorical_cols:
        df_cleaned[col] = df_cleaned[col].astype('category')
    df_cleaned['age'] = df_cleaned['age'].astype(int)
    df_cleaned['children'] = df_cleaned['children'].astype(int)

    # Outlier Analysis using IQR
    print("\n[Step 4] Outlier Analysis (Interquartile Range Method):")
    for col in ['bmi', 'charges']:
        q1 = df_cleaned[col].quantile(0.25)
        q3 = df_cleaned[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        print(f" -> Feature '{col}': Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Bounds=[{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"    Identified {len(outliers)} outliers ({len(outliers)/len(df_cleaned)*100:.2f}%)")

    # 4. Data Transformations & Feature Engineering
    print("\n[Step 5] Applying Transformations & Feature Engineering:")
    # Log Transformation of Charges to mitigate severe positive skewness
    raw_skew = df_cleaned['charges'].skew()
    df_cleaned['log_charges'] = np.log(df_cleaned['charges'])
    log_skew = df_cleaned['log_charges'].skew()
    print(f" -> Skewness of 'charges' before transformation: {raw_skew:.4f}")
    print(f" -> Skewness of 'log_charges' after ln(x) transformation: {log_skew:.4f}")

    # BMI Categorization according to World Health Organization (WHO)
    bmi_bins = [0, 18.5, 24.9, 29.9, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df_cleaned['bmi_category'] = pd.cut(df_cleaned['bmi'], bins=bmi_bins, labels=bmi_labels, right=True)

    # Smoker Binary
    df_cleaned['smoker_code'] = df_cleaned['smoker'].map({'no': 0, 'yes': 1}).astype(int)

    # Age Groupings
    age_bins = [17, 29, 44, 54, 65]
    age_labels = ['18-29', '30-44', '45-54', '55-64']
    df_cleaned['age_group'] = pd.cut(df_cleaned['age'], bins=age_bins, labels=age_labels)

    # Save Cleaned Dataset
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"\n[Step 6] Saved cleaned and transformed dataset to: {cleaned_path}")

    # 5. Visualizations
    print("\n[Step 7] Generating Professional Visualizations (300 DPI):")

    # Figure 1: Distribution of Annual Medical Charges (Raw vs. Log-Transformed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Raw Charges Distribution
    mean_charges = df_cleaned['charges'].mean()
    median_charges = df_cleaned['charges'].median()
    sns.histplot(df_cleaned['charges'], kde=True, bins=35, color="#1f77b4", edgecolor="black", alpha=0.6, ax=axes[0])
    axes[0].axvline(mean_charges, color="#d62728", linestyle="--", linewidth=2, label=f"Mean: ${mean_charges:,.2f}")
    axes[0].axvline(median_charges, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Median: ${median_charges:,.2f}")
    axes[0].set_title(f"Raw Medical Charges Distribution\n(Positive Skewness = {raw_skew:.2f})", fontweight="bold")
    axes[0].set_xlabel("Individual Medical Charges (USD $)")
    axes[0].set_ylabel("Frequency Count")
    axes[0].legend(loc="upper right", frameon=True)

    # Log-Transformed Charges Distribution
    mean_log = df_cleaned['log_charges'].mean()
    median_log = df_cleaned['log_charges'].median()
    sns.histplot(df_cleaned['log_charges'], kde=True, bins=35, color="#ff7f0e", edgecolor="black", alpha=0.6, ax=axes[1])
    axes[1].axvline(mean_log, color="#d62728", linestyle="--", linewidth=2, label=f"Mean: {mean_log:.2f}")
    axes[1].axvline(median_log, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Median: {median_log:.2f}")
    axes[1].set_title(f"Log-Transformed Charges (ln(Charges))\n(Adjusted Skewness = {log_skew:.2f})", fontweight="bold")
    axes[1].set_xlabel("Logarithm of Medical Charges ln(USD $)")
    axes[1].set_ylabel("Frequency Count")
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig1_path = os.path.join(vis_dir, "figure1_medical_charges_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" -> Saved Figure 1 to: {fig1_path}")

    # Figure 2: Scatter Plot of BMI vs. Medical Charges Stratified by Smoking Status
    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = {"yes": "#d62728", "no": "#1f77b4"}
    
    # Scatter plot with regression fit lines
    sns.scatterplot(
        data=df_cleaned,
        x="bmi",
        y="charges",
        hue="smoker",
        style="smoker",
        palette=palette,
        alpha=0.75,
        s=60,
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    # Regression lines for each smoker group
    sns.regplot(
        data=df_cleaned[df_cleaned['smoker'] == 'no'],
        x="bmi",
        y="charges",
        scatter=False,
        ax=ax,
        color="#1f77b4",
        line_kws={"linewidth": 2, "linestyle": "--", "label": "Non-Smoker Trend"}
    )
    sns.regplot(
        data=df_cleaned[df_cleaned['smoker'] == 'yes'],
        x="bmi",
        y="charges",
        scatter=False,
        ax=ax,
        color="#d62728",
        line_kws={"linewidth": 2, "linestyle": "-", "label": "Smoker Trend"}
    )

    # Reference clinical thresholds
    ax.axvline(30.0, color="gray", linestyle=":", linewidth=1.5, label="Obesity Threshold (BMI ≥ 30)")
    ax.set_title("Relationship Between BMI and Medical Charges by Smoking Status", fontweight="bold", pad=12)
    ax.set_xlabel("Body Mass Index (BMI, kg/m²)")
    ax.set_ylabel("Annual Medical Charges (USD $)")
    
    # Formatting y-axis with dollar currency
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc="upper left", frameon=True, title="Smoking Category")

    plt.tight_layout()
    fig2_path = os.path.join(vis_dir, "figure2_bmi_vs_charges_by_smoking.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f" -> Saved Figure 2 to: {fig2_path}")

    # Summary Statistics Export
    summary_stats = df_cleaned.describe()
    summary_path = os.path.join("week4", "data", "summary_statistics.csv")
    summary_stats.to_csv(summary_path)
    print(f" -> Exported descriptive summary stats to: {summary_path}")

    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    run_pipeline()
