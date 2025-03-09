import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from pathlib import Path


# Function: Ensure that a directory exists
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' has been created.")
    else:
        print(f"Directory '{directory}' already exists.")


# Function: Plot combined unique employee IDs by year
def plot_combined_unique_employee_ids_by_year(df, column_name, year_column, plots_dir, plot_name):
    if "Employee_ID" in df.columns:
        # Ensure that the year column exists
        if year_column not in df.columns:
            raise KeyError(f"Column '{year_column}' not found in the DataFrame.")

        # Only unique employee IDs
        df_unique = df.drop_duplicates(subset="Employee_ID")

        # Ensure that the desired column exists
        if column_name in df.columns:
            # Filter data by year
            years = sorted(df_unique[year_column].dropna().unique())

            # Define the number of subplots (one per year)
            fig, axes = plt.subplots(len(years), 1, figsize=(8, 6 * len(years)), sharex=True)
            fig.suptitle(f"Categorical Distribution: {column_name} by Year", fontsize=16)

            for i, year in enumerate(years):
                df_year = df_unique[df_unique[year_column] == year]
                ax = axes[i] if len(years) > 1 else axes
                sns.countplot(x=column_name, data=df_year, ax=ax)
                ax.set_title(f"Year: {year}")
                ax.set_xlabel(column_name)
                ax.set_ylabel("Count of Unique IDs")

            # Layout and save
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for the main title
            plt.savefig(plots_dir / plot_name, bbox_inches="tight")
            plt.show()
        else:
            raise KeyError(f"Column '{column_name}' not found in the DataFrame.")
    else:
        raise KeyError("Column 'Employee_ID' not found in the DataFrame.")


# Function: Compute and visualize skewness and kurtosis
def plot_skewness_kurtosis(df, numerical_columns):
    print("\nSkewness and Kurtosis for numerical columns:")
    for column in numerical_columns:
        skew = df[column].skew()
        kurt = df[column].kurtosis()
        print(f"{column}: Skewness = {skew:.2f}, Kurtosis = {kurt:.2f}")


# Function: Create histograms for numerical columns
def plot_histograms(df, numerical_columns, plots_dir, plot_name):
    print("\nCreating histograms for numerical columns...")
    num_cols = 3
    num_rows = -(-len(numerical_columns) // num_cols)  # Round up
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):
        sns.histplot(df[column].dropna(), bins=30, kde=True, ax=axes[i], edgecolor="black")
        axes[i].set_title(f"Histogram: {column}")

    for i in range(len(numerical_columns), len(axes)):  # Remove empty plots
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(plots_dir / plot_name, bbox_inches="tight")
    plt.show()


# Function: Generate scatterplots between pairs of columns
def plot_scatterplots(df, relevant_columns, plots_dir, plot_name):
    if len(relevant_columns) < 2:
        print("Not enough numerical columns for scatterplots.")
        return

    print("\nCreating a compact view of scatterplots...")
    num_cols = 3
    column_pairs = [(relevant_columns[i], relevant_columns[j])
                    for i in range(len(relevant_columns))
                    for j in range(i + 1, len(relevant_columns))]
    num_rows = -(-len(column_pairs) // num_cols)  # Round up
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
    axes = axes.flatten()

    for i, (column_x, column_y) in enumerate(column_pairs):
        sns.scatterplot(data=df, x=column_x, y=column_y, alpha=0.7, ax=axes[i])
        axes[i].set_title(f"{column_x} vs {column_y}")
        axes[i].set_xlabel(column_x)
        axes[i].set_ylabel(column_y)

    for i in range(len(column_pairs), len(axes)):  # Remove unused spaces
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(plots_dir / plot_name, bbox_inches="tight")
    plt.show()

# Funktion: Erstelle Heatmap der Korrelationsmatrix
def plot_correlation_heatmap(df, numerische_spalten, plots_dir, plot_name):
    if len(numerische_spalten) > 1:
        print("\nErstelle Heatmap der Korrelationsmatrix...")
        corr_matrix = df[numerische_spalten].corr()
        mask = np.eye(corr_matrix.shape[0], dtype=bool)

        # Pandas-Anzeigeoptionen setzen, um die Matrix vollständig zu drucken
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(corr_matrix)

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, mask=mask)
        plt.title("Korrelationsmatrix mit ausgeschlossener Hauptdiagonale")
        plt.savefig(plots_dir / plot_name, bbox_inches="tight")
        plt.show()


# Function: Density Plot (KDE) Example
def plot_kde(df, column, plots_dir, plot_name):
    if column in df.columns:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(df[column], fill=True, color="green")
        plt.title(f"Density Distribution of {column}")
        plt.savefig(plots_dir / plot_name, bbox_inches="tight")
        plt.show()




# Main Function
def main():
    # Start Timer
    start_time = time.time()

    # Define Base Directory for Plots
    PLOTS_DIR = Path("plots")
    ensure_dir_exists(PLOTS_DIR)

    # Load Data
    data_file_path = Path("../data/HR_cleaned.csv", low_memory=False)
    if not data_file_path.exists():
        print(f"ERROR: File '{data_file_path}' does not exist.")
        exit(1)

    df = pd.read_csv(data_file_path)
    print("Cleaned dataset successfully loaded!")

    # Analyze Columns
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns
    print("Data Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Visualization: Unique IDs per Year
    plot_combined_unique_employee_ids_by_year(df, "Gender", "Year", PLOTS_DIR, "02_gender_by_year.png")
    plot_combined_unique_employee_ids_by_year(df, "Job Level", "Year", PLOTS_DIR, "03_job_level_by_year.png")

    plot_skewness_kurtosis(df, numerical_columns)

    plot_histograms(df, numerical_columns, PLOTS_DIR, "03_histograms.png")

    relevant_columns = [col for col in numerical_columns if col not in [
        "Employee_ID", "Satisfaction", "Training Costs",
        "Team Size", "Internal Trainings", "Vacation Days Taken",
        "Annual Performance Review", "Year", "Number of Subordinates", "Time to Retirement", "Turnover Willingness", "Month"]]
    plot_scatterplots(df, relevant_columns, PLOTS_DIR, "04_scatterplots.png")

    plot_correlation_heatmap(df, numerical_columns, PLOTS_DIR, "05_correlation_heatmap.png")

    plot_kde(df, "Turnover Willingness", PLOTS_DIR, "06_kde_turnover_willingness.png")

    # Stop Timer
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()