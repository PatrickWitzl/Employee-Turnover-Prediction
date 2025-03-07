import os
import joblib
import pandas as pd
import xgboost as xgb
from data_loading import load_dataset
from data_cleaning import clean_dataset
from ML1_Fluctuation_best_model_6_ohne_pca import preprocess_data
from ML1_Fluctuation_best_model_6_ohne_pca import get_critical_employees


def process_and_identify_critical_employees(
        file_path,  # Input file
        save_filtered_path=None,  # Path to save filtered data
        models_dir="Models",  # Directory where the model is stored
        model_file_name="xgboost_model.pkl",  # Name of the model file
        feature_names_file="Models/lightgbm_feature_names.pkl",  # File containing feature names
        threshold=0.0  # Threshold for identifying critical employees
):
    """
    This function handles data preprocessing, model loading, and identification
    of critical employees based on a predictive model.

    Args:
        file_path (str): Path to the input file (CSV).
        save_filtered_path (str): Path to save the filtered input file.
        models_dir (str): Directory where the trained model is stored.
        model_file_name (str): Name of the file in which the model is stored.
        feature_names_file (str): File containing feature names for interpretation.
        threshold (float): Threshold for identifying critical employees.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame with critical employees and the top 15 employees by risk.
    """
    try:
        # Load and clean data
        df = load_dataset(file_path, save_filtered_path)
        df_cleaned = clean_dataset(df)

        # Preprocess data
        df_processed, X_transformed, X_resampled, y_resampled, preprocessor = preprocess_data(df_cleaned)

        # Create model path and load the model
        model_file = os.path.join(models_dir, model_file_name)
        try:
            model = joblib.load(model_file)  # Load the model
            print("The model was successfully loaded.")
        except FileNotFoundError:
            print(f"Error: The model file '{model_file}' was not found.")
            return None, None
        except Exception as e:
            print(f"Error while loading the model: {e}")
            return None, None

        # Identify critical employees
        critical_employees, top_15_employees = get_critical_employees(
            model, X_transformed, df_processed, feature_names_file=feature_names_file, threshold=threshold
        )

        # Output results
        print(
            f"Top 15 Employees:\n{top_15_employees[['Employee_ID', 'Name', 'Turnover_Probability']]}"
            # Update column names
        )

        return critical_employees, top_15_employees
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None