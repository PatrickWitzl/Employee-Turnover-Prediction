import pandas as pd
import time

# Start timer
start_time = time.time()


def clean_dataset(df):
    """
    Clean the dataset by handling missing values, processing object-based columns, and removing duplicates.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    if df is None:
        print("No dataset provided.")
        return None

    # Check for missing values
    missing_values = df.isnull().sum()
    print(missing_values)

    # Fill missing values in 'Exit Date' for active employees
    df["Exit Date"] = df["Exit Date"].fillna("No Exit")  # Or leave as None

    # Handle missing values
    df = df.fillna({
        "Internal Training": 0,
        "Training Costs": 0,
        "Illness Days": df["Illness Days"].median(),
        "Overtime": df["Overtime"].median(),
    })

    # Clean 'Hiring Date' without conversion
    df["Hiring Date"] = df["Hiring Date"].astype(str)  # Ensure all values are strings
    df["Hiring Date"] = df["Hiring Date"].str.strip()  # Remove leading/trailing whitespaces
    df["Hiring Date"] = df["Hiring Date"].replace(["", "null", "N/A"], pd.NA)

    # Further clean and convert columns
    df["Switching Readiness"] = df["Switching Readiness"].astype(float)
    df["Illness Days"] = df["Illness Days"].astype(int)
    df["Overtime"] = df["Overtime"].astype(int)

    # Remove duplicate entries
    df.drop_duplicates(inplace=True)

    print("\nDataset cleaned successfully!")
    print(df.info())
    print(df.head())

    return df

# Calculate duration
end_time = time.time()
print(f"Analysis completed in {end_time - start_time:.2f} seconds.")

# Save the cleaned dataset
if __name__ == "__main__":
    file_path = "../data/HR_dataset_filtered.csv"
    df = pd.read_csv(file_path)  # Load the dataset
    cleaned_df = clean_dataset(df)

    # Save the cleaned dataset
    cleaned_file_path = "../data/HR_cleaned.csv"
    cleaned_df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_file_path}")