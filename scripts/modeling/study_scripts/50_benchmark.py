"""
===============================================================================
Project : BCRABL-AI

File:
    50_benchmark.py

Purpose:
    Automated AI Model Benchmarking Pipeline

    - Collect model performance results
    - Generate comparison tables
    - Create publication-quality figures

===============================================================================
"""


# =============================================================================
# Imports
# =============================================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# =============================================================================
# Paths
# =============================================================================

RESULTS_DIR = Path(
    "07_Results"
)


OUTPUT_DIR = Path(
    "08_Benchmark"
)


TABLE_DIR = OUTPUT_DIR / "Tables"


FIGURE_DIR = OUTPUT_DIR / "Figures"


# Create folders

TABLE_DIR.mkdir(

    parents=True,

    exist_ok=True

)


FIGURE_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# =============================================================================
# Header
# =============================================================================

print()

print("=" * 80)
print("BCRABL-AI MODEL BENCHMARK")
print("=" * 80)

print()

print("Results Directory:")
print(RESULTS_DIR)

print()

print("Output Directory:")
print(OUTPUT_DIR)


# =============================================================================
# Detect Models Automatically
# =============================================================================


print()

print("=" * 80)
print("SEARCHING FOR TRAINED MODELS")
print("=" * 80)


models = []


for folder in RESULTS_DIR.iterdir():

    if not folder.is_dir():

        continue


    performance_file = folder / "performance_summary.csv"


    if performance_file.exists():

        models.append(

            {

                "name": folder.name,

                "path": folder

            }

        )


print()


if len(models) == 0:

    raise FileNotFoundError(

        "No performance_summary.csv files found inside 07_Results"

    )


print("Detected Models")
print("-" * 80)


for model in models:

    print(

        "✓",

        model["name"]

    )


print()

print(

    f"Total Models Found : {len(models)}"

)


print()

print("=" * 80)
print("PART 1 COMPLETED")
print("=" * 80)
# =============================================================================
# Load Performance Results
# =============================================================================


print()

print("=" * 80)
print("LOADING MODEL PERFORMANCE")
print("=" * 80)


performance_data = []


for model in models:

    file = model["path"] / "performance_summary.csv"


    print()

    print(
        "Loading:",
        model["name"]
    )


    df = pd.read_csv(file)


    # Add model name

    df.insert(

        0,

        "Model",

        model["name"]

    )


    performance_data.append(df)



# Combine all models

performance_all = pd.concat(

    performance_data,

    ignore_index=True

)


print()

print("=" * 80)
print("COMBINED PERFORMANCE DATA")
print("=" * 80)

print()

print(performance_all)


print()

print("Columns")

print("-" * 80)

print(

    list(performance_all.columns)

)


print()

print("=" * 80)
print("PART 2 COMPLETED")
print("=" * 80)
# =============================================================================
# Create Benchmark Summary Table
# =============================================================================


print()

print("=" * 80)
print("CREATING BENCHMARK SUMMARY")
print("=" * 80)


summary_rows = []


for model_name in performance_all["Model"].unique():

    model_df = performance_all[

        performance_all["Model"] == model_name

    ]


    train = model_df[

        model_df["Dataset"].str.lower() == "train"

    ]


    validation = model_df[

        model_df["Dataset"].str.lower() == "validation"

    ]


    test = model_df[

        model_df["Dataset"].str.lower() == "test"

    ]


    row = {

        "Model": model_name,


        "Train R2":

            train["R2"].values[0]
            if len(train) > 0 else np.nan,


        "Validation R2":

            validation["R2"].values[0]
            if len(validation) > 0 else np.nan,


        "Test R2":

            test["R2"].values[0]
            if len(test) > 0 else np.nan,


        "Test RMSE":

            test["RMSE"].values[0]
            if len(test) > 0 else np.nan,


        "Test MAE":

            test["MAE"].values[0]
            if len(test) > 0 else np.nan

    }


    summary_rows.append(row)



benchmark_table = pd.DataFrame(

    summary_rows

)


# =============================================================================
# Ranking by Test R2
# =============================================================================


benchmark_table = benchmark_table.sort_values(

    by="Test R2",

    ascending=False

).reset_index(

    drop=True

)


benchmark_table.insert(

    0,

    "Rank",

    range(

        1,

        len(benchmark_table) + 1

    )

)


# Round values

for col in benchmark_table.columns[2:]:

    benchmark_table[col] = benchmark_table[col].round(4)



print()

print("=" * 80)
print("FINAL BENCHMARK TABLE")
print("=" * 80)

print()

print(benchmark_table)



# =============================================================================
# Save Tables
# =============================================================================


benchmark_table.to_csv(

    TABLE_DIR / "benchmark_summary.csv",

    index=False

)


benchmark_table.to_excel(

    TABLE_DIR / "benchmark_summary.xlsx",

    index=False

)


print()

print("Saved:")

print(

    TABLE_DIR / "benchmark_summary.csv"

)

print(

    TABLE_DIR / "benchmark_summary.xlsx"

)


print()

print("=" * 80)
print("PART 3 COMPLETED")
print("=" * 80)

# =============================================================================
# Figure 1 - Test R2 Comparison
# =============================================================================


print()

print("=" * 80)
print("GENERATING TEST R2 FIGURE")
print("=" * 80)


# Sort models by performance

plot_data = benchmark_table.sort_values(

    by="Test R2",

    ascending=True

)


plt.figure(

    figsize=(8, 5)

)


bars = plt.barh(

    plot_data["Model"],

    plot_data["Test R2"]

)


# Add values on bars

for bar in bars:

    value = bar.get_width()

    plt.text(

        value + 0.01,

        bar.get_y() + bar.get_height()/2,

        f"{value:.3f}",

        va="center",

        fontsize=10

    )


plt.xlabel(

    "Test R²"

)


plt.ylabel(

    "Model"

)


plt.title(

    "Comparison of AI Models Based on Test R²"

)


plt.xlim(

    0,

    1

)


plt.tight_layout()



# Save high resolution

plt.savefig(

    FIGURE_DIR / "Fig1_Test_R2_Comparison.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



print()

print("Saved:")

print(

    FIGURE_DIR / "Fig1_Test_R2_Comparison.png"

)


print()

print("=" * 80)
print("PART 4 COMPLETED")
print("=" * 80)
# =============================================================================
# Figure 2 - Test RMSE Comparison
# =============================================================================


print()

print("=" * 80)
print("GENERATING RMSE FIGURE")
print("=" * 80)


plot_data = benchmark_table.sort_values(

    by="Test RMSE",

    ascending=False

)


plt.figure(

    figsize=(8, 5)

)


bars = plt.barh(

    plot_data["Model"],

    plot_data["Test RMSE"]

)


for bar in bars:

    value = bar.get_width()

    plt.text(

        value + 0.02,

        bar.get_y() + bar.get_height()/2,

        f"{value:.3f}",

        va="center",

        fontsize=10

    )


plt.xlabel(

    "Test RMSE"

)


plt.ylabel(

    "Model"

)


plt.title(

    "Comparison of AI Models Based on Test RMSE"

)


plt.tight_layout()


plt.savefig(

    FIGURE_DIR / "Fig2_Test_RMSE_Comparison.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



# =============================================================================
# Figure 3 - Test MAE Comparison
# =============================================================================


print()

print("=" * 80)
print("GENERATING MAE FIGURE")
print("=" * 80)


plot_data = benchmark_table.sort_values(

    by="Test MAE",

    ascending=False

)


plt.figure(

    figsize=(8, 5)

)


bars = plt.barh(

    plot_data["Model"],

    plot_data["Test MAE"]

)


for bar in bars:

    value = bar.get_width()

    plt.text(

        value + 0.02,

        bar.get_y() + bar.get_height()/2,

        f"{value:.3f}",

        va="center",

        fontsize=10

    )


plt.xlabel(

    "Test MAE"

)


plt.ylabel(

    "Model"

)


plt.title(

    "Comparison of AI Models Based on Test MAE"

)


plt.tight_layout()


plt.savefig(

    FIGURE_DIR / "Fig3_Test_MAE_Comparison.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



print()

print("Figures Saved:")

print(

    FIGURE_DIR / "Fig2_Test_RMSE_Comparison.png"

)

print(

    FIGURE_DIR / "Fig3_Test_MAE_Comparison.png"

)


print()

print("=" * 80)
print("PART 5 COMPLETED")
print("=" * 80)

# =============================================================================
# Figure 4 - Performance Heatmap
# =============================================================================


print()

print("=" * 80)
print("GENERATING PERFORMANCE HEATMAP")
print("=" * 80)


# Prepare heatmap data

heatmap_data = benchmark_table.copy()


heatmap_data = heatmap_data[

    [

        "Model",

        "Test R2",

        "Test RMSE",

        "Test MAE"

    ]

]


heatmap_data = heatmap_data.set_index(

    "Model"

)


plt.figure(

    figsize=(8, 5)

)


sns.heatmap(

    heatmap_data,

    annot=True,

    fmt=".3f",

    linewidths=0.5,

    cmap="viridis"

)


plt.title(

    "AI Model Performance Heatmap"

)


plt.tight_layout()



plt.savefig(

    FIGURE_DIR / "Fig4_Performance_Heatmap.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



print()

print("Saved:")

print(

    FIGURE_DIR / "Fig4_Performance_Heatmap.png"

)


print()

print("=" * 80)
print("PART 6 COMPLETED")
print("=" * 80)
# =============================================================================
# Figure 5 - Radar Chart
# =============================================================================


print()

print("=" * 80)
print("GENERATING RADAR CHART")
print("=" * 80)


radar_data = benchmark_table.copy()


# Normalize metrics

radar_data["R2_score"] = (

    radar_data["Test R2"]

    /

    radar_data["Test R2"].max()

)


radar_data["RMSE_score"] = (

    1 -

    (

        radar_data["Test RMSE"]

        /

        radar_data["Test RMSE"].max()

    )

)


radar_data["MAE_score"] = (

    1 -

    (

        radar_data["Test MAE"]

        /

        radar_data["Test MAE"].max()

    )

)


# Composite score

radar_data["Composite Score"] = (

    radar_data["R2_score"] * 0.5

    +

    radar_data["RMSE_score"] * 0.25

    +

    radar_data["MAE_score"] * 0.25

)


radar_data["Composite Score"] = (

    radar_data["Composite Score"] * 100

).round(2)


# Save ranking

radar_data.sort_values(

    by="Composite Score",

    ascending=False

).to_csv(

    TABLE_DIR / "model_ranking_score.csv",

    index=False

)



# Radar plot

categories = [

    "R2_score",

    "RMSE_score",

    "MAE_score"

]


N = len(categories)


angles = [

    n / float(N) * 2 * pi

    for n in range(N)

]


angles += angles[:1]


plt.figure(

    figsize=(7,7)

)


ax = plt.subplot(

    111,

    polar=True

)


for _, row in radar_data.iterrows():

    values = [

        row[c]

        for c in categories

    ]

    values += values[:1]


    ax.plot(

        angles,

        values,

        linewidth=2,

        label=row["Model"]

    )


    ax.fill(

        angles,

        values,

        alpha=0.1

    )


plt.xticks(

    angles[:-1],

    [

        "R²",

        "RMSE",

        "MAE"

    ]

)


plt.title(

    "AI Model Performance Radar Chart"

)


plt.legend(

    loc="upper right",

    bbox_to_anchor=(1.3,1.1)

)


plt.tight_layout()


plt.savefig(

    FIGURE_DIR / "Fig5_Radar_Chart.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



print()

print("Saved:")

print(

    FIGURE_DIR / "Fig5_Radar_Chart.png"

)

print()

print("Composite Ranking")

print(

    radar_data[

        [

            "Model",

            "Composite Score"

        ]

    ].sort_values(

        by="Composite Score",

        ascending=False

    )

)


print()

print("=" * 80)
print("PART 7 COMPLETED")
print("=" * 80)
# =============================================================================
# Part 8 - Prediction Analysis
# =============================================================================

print()

print("=" * 80)
print("GENERATING PREDICTION ANALYSIS")
print("=" * 80)


prediction_data = []


for model in models:

    prediction_file = (

        model["path"]

        /

        "test_predictions.csv"

    )


    if prediction_file.exists():

        df = pd.read_csv(

            prediction_file

        )


        df["Model"] = model["name"]


        prediction_data.append(df)


    else:

        print(

            "Prediction file missing:",

            model["name"]

        )



# Continue only if predictions exist

if len(prediction_data) > 0:


    predictions = pd.concat(

        prediction_data,

        ignore_index=True

    )


    print()

    print("Prediction files loaded")



    # ------------------------------------------------------------
    # Detect columns
    # ------------------------------------------------------------

    print()

    print("Prediction columns:")

    print(

        predictions.columns.tolist()

    )



    # ------------------------------------------------------------
    # Observed vs Predicted
    # ------------------------------------------------------------


    plt.figure(

        figsize=(10,8)

    )


    for model_name in predictions["Model"].unique():


        subset = predictions[

            predictions["Model"] == model_name

        ]


        plt.scatter(

            subset["Observed"],

            subset["Predicted"],

            alpha=0.5,

            label=model_name

        )


    minimum = min(

        predictions["Observed"].min(),

        predictions["Predicted"].min()

    )


    maximum = max(

        predictions["Observed"].max(),

        predictions["Predicted"].max()

    )


    plt.plot(

        [minimum, maximum],

        [minimum, maximum],

        linestyle="--"

    )


    plt.xlabel(

        "Observed pIC50"

    )


    plt.ylabel(

        "Predicted pIC50"

    )


    plt.title(

        "Observed vs Predicted Performance"

    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR / "Fig6_Observed_vs_Predicted.png",

        dpi=600,

        bbox_inches="tight"

    )


    plt.close()



    # ------------------------------------------------------------
    # Residual Analysis
    # ------------------------------------------------------------


    predictions["Residual"]


    plt.figure(

        figsize=(10,6)

    )


    for model_name in predictions["Model"].unique():


        subset = predictions[

            predictions["Model"] == model_name

        ]


        plt.hist(

            subset["Residual"],

            bins=30,

            alpha=0.4,

            label=model_name

        )


    plt.xlabel(

        "Residual Error"

    )


    plt.ylabel(

        "Frequency"

    )


    plt.title(

        "Residual Distribution"

    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(

        FIGURE_DIR / "Fig7_Residual_Distribution.png",

        dpi=600,

        bbox_inches="tight"

    )


    plt.close()



    print()

    print("Prediction Figures Saved")



# =============================================================================
# Generate Summary Report
# =============================================================================


report_file = OUTPUT_DIR / "Benchmark_Report.txt"


with open(

    report_file,

    "w",

    encoding="utf-8"

) as f:


    f.write(

        "BCRABL-AI Model Benchmark Report\n"

    )


    f.write(

        "=" * 50

    )


    f.write("\n\n")


    best_model = benchmark_table.iloc[0]


    f.write(

        f"Best Model: {best_model['Model']}\n"

    )


    f.write(

        f"Test R2: {best_model['Test R2']}\n"

    )


    f.write(

        f"Test RMSE: {best_model['Test RMSE']}\n"

    )


    f.write(

        f"Test MAE: {best_model['Test MAE']}\n"

    )



print()

print("=" * 80)
print("BENCHMARK COMPLETED")
print("=" * 80)


print()

print("Output:")

print(

    OUTPUT_DIR

)