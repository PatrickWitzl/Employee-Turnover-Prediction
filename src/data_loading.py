import pandas as pd
import os



def load_dataset(file_path, save_filtered_path):
    """
    Load dataset from a CSV file, filter data based on a year range (2015–2025),
    and save the filtered data to a new CSV file.

    Args:
        file_path (str): Path to the CSV file to load.
        save_filtered_path (str): Path to save the filtered data.

    Returns:
        pd.DataFrame or None: Loaded and filtered DataFrame, else None if an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Check if a "Year" column exists or generate it if not present
        if "Year" not in df.columns:
            if "Hiring Date" in df.columns:
                # Extract year from the "Hiring Date" column if it exists
                df["Hiring Date"] = pd.to_datetime(df["Hiring Date"])  # Parse dates
                df["Year"] = df["Hiring Date"].dt.year
            else:
                print("Error: Missing 'Year' or 'Hiring Date' column.")
                return None

        # Filter data: Year between 2015 and 2025
        start_year = 2015
        end_year = 2025
        df_filtered = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]

        # Display filtered data
        print("Filtered Dataset (2015-2025):")
        print(df_filtered.info())
        print("\nFirst 5 rows preview of filtered data:")
        print(df_filtered.head())

        # Save filtered dataset
        df_filtered.to_csv(save_filtered_path, index=False, date_format="%Y-%m-%d")
        print(f"\nFiltered data successfully saved to '{save_filtered_path}'.")

        return df_filtered
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Check the path.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: Problem parsing the CSV file.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


if __name__ == "__main__":
    # Example: Input and output paths
    file_path = "../data/HR_dataset.csv"  # Input file
    save_filtered_path = "../data/HR_dataset_filtered.csv"  # Output file for filtered data
    df = load_dataset(file_path, save_filtered_path)