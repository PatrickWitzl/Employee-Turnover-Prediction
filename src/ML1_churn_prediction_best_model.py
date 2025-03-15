import os
import sys
import time
import select
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    log_loss,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
def load_data(file_path):
    """
    Loads data from a CSV file and processes it according to requirements.
    Creates a new column 'Turnover' which contains 1 for 'Terminated' and 0 for other values,
    and removes the 'Status' column.

    Args:
        file_path (str): File path to the CSV file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    # Load data
    df = pd.read_csv(file_path, low_memory=False)

    return df


def preprocess_data(df):
    """
    Data preprocessing steps:
    - Derive the target variable "Turnover"
    - Select features based on analysis
    - Apply one-hot encoding for categorical features
    - Handle class imbalance using SMOTE
    - Adjust "Absence Days" based on "Absence Reason"

    Args:
        df (pd.DataFrame): The original DataFrame.

    Returns:
        pd.DataFrame: Original transformed features (X_transformed).
        pd.DataFrame: Resampled features (X_resampled).
        np.ndarray: Target variable (y_resampled) after SMOTE.
        ColumnTransformer: Preprocessor for future transformations.
    """
    # Remove employees with "Retirement" status
    df = df.copy()  # Avoid modifying the original
    df = df[df['Status'] != 'Retired']
    print(f"Shape after removing 'Retired': {df.shape}")

    # Clean "Absence Days" if "Absence Reason" is not "Sick"
    df['Absence Days'] = df.apply(
        lambda row: row['Absence Days'] if row['Absence Reason'] == "Illness" else 0,
        axis=1
    )

    # Derive the target variable "Turnover"
    df['Turnover'] = df['Status'].apply(lambda x: 1 if x == "Left" else 0)

    # Combine features (manual selection after analysis)
    selected_features = [
        'Year', 'Month', 'Age', 'Overtime', 'Absence Days',
        'Salary', 'Satisfaction', 'Switching Readiness', 'Training Costs',
        'Position', 'Gender', 'Location',
        'Work Model', 'Married',
        'Children', 'Job Role Progression', 'Job Level', 'Tenure'
    ]
    X = df[selected_features]
    y = df['Turnover']

    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include='object').columns
    numerical_columns = X.select_dtypes(exclude='object').columns

    # Define preprocessor with scaling for numerical columns and one-hot encoding for categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
        ]
    )

    # Transform features
    X_transformed = preprocessor.fit_transform(X)

    # New column names after transformation
    transformed_columns = list(preprocessor.named_transformers_['num'].feature_names_in_) + \
                          list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns))

    # Write back to DataFrame
    X_transformed = pd.DataFrame(X_transformed, columns=transformed_columns, index=X.index)

    # Ensure synchronization of df with X_transformed
    if not X_transformed.index.equals(df.index):
        print("Index mismatch! Syncing is being done.")
        df = df.loc[X_transformed.index]  # Synchronize based on the index of X_transformed

    # SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_transformed, y)

    # Return as DataFrame for readable columns
    X_resampled = pd.DataFrame(X_resampled, columns=X_transformed.columns)

    return df, X_transformed, X_resampled, y_resampled, preprocessor


def split(X_resampled, y_resampled, test_size=0.2):
    """
    Splits the data into training and test sets and scales it.

    Args:
        X_resampled (np.ndarray): Features after resampling.
        y_resampled (np.ndarray): Target variable after resampling.
        test_size (float): Proportion of test data.

    Returns:
        pd.DataFrame: Scaled training data (X_train_scaled).
        pd.DataFrame: Scaled test data (X_test_scaled).
        np.ndarray: Training target variables.
        np.ndarray: Test target variables.
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=42, stratify=y_resampled
    )

    return X_train, X_test, y_train, y_test
def model_selection():
    """
    Allows the user to select models for analysis. The user can either select specific models
    or choose all models. If the user does not make a selection within 10 seconds,
    all models will be automatically selected.

    Returns:
        include_models (list): List of selected models.
    """
    # Mapping for models
    model_mapping = {
        1: "Logistic Regression",
        2: "Random Forest",
        3: "XGBoost",
        4: "LightGBM",
        5: "Neural Network"
    }

    print("\nPlease select models for analysis:")
    print("1: Logistic Regression")
    print("2: Random Forest")
    print("3: XGBoost")
    print("4: LightGBM")
    print("5: Neural Network")
    print("6: All models")

    def input_with_timeout(prompt, timeout=0.10):
        """
        Processes user input with a timeout. If the user does not respond within
        the specified time, None is returned.
        """
        print(prompt, end='', flush=True)  # Show input prompt
        inputs, _, _ = select.select([sys.stdin], [], [], timeout)
        if inputs:
            return sys.stdin.readline().strip()  # Read input and return it
        else:
            print("\nTimeout expired. No input detected.")
            return None

    # Wait for user input with a timeout
    selected_input = input_with_timeout(
        "Enter the numbers of the desired models separated by commas "
        "(e.g., 1,3,5 or 6 for all models):\n",
        timeout=0.10  # Timeout in seconds
    )

    # If no input is provided, select all models by default
    if selected_input is None:
        print("\nNo selection made. All models will be selected by default.")
        return list(model_mapping.values())  # Return all model names

    # Process user input
    try:
        selected_numbers = [int(n) for n in selected_input.split(",")]  # Convert user input into a list
    except ValueError:
        raise ValueError("Invalid input. Please enter valid model numbers (e.g., 1,2 or 6).")

    if 6 in selected_numbers:  # 'All models' has been selected
        include_models = list(model_mapping.values())  # Return all models
    else:
        # Compile only the selected models
        include_models = [model_mapping[n] for n in selected_numbers if n in model_mapping]

    if not include_models:  # Ensure at least one model is selected
        raise ValueError("No valid models selected. Please select at least one model.")

    return include_models


def get_user_choice(models_dir, timeout=0.10):
    """
    Asks the user if stored models should be used or new models should be trained.

    Args:
        models_dir (str): Directory where models are stored.
        timeout (int): Time in seconds for user input.

    Returns:
        bool: True if stored models should be used and they exist, otherwise False.
    """

    def input_with_timeout(prompt, timeout=0.10):
        print(prompt, end='', flush=True)
        inputs, _, _ = select.select([sys.stdin], [], [], timeout)
        if inputs:
            return sys.stdin.readline().strip().lower()  # Read input
        else:
            print("\nTimeout expired. Default: 'y' (use stored models).")
            return "y"  # Default option is now "y"

    # Let the user select an option
    user_choice = input_with_timeout(
        "Use stored models? [y/n] (Default: y): ", timeout
    )

    # Check if models exist
    def models_exist(models_dir):
        # Look for existing model files in `model_dir`
        for file in os.listdir(models_dir):
            if file.endswith(".pkl") or file.endswith(".json") or file.endswith(".keras"):
                return True
        return False

    # If the user selected 'y', check if models exist
    if user_choice == "y":
        if models_exist(models_dir):
            print("Stored models will be used.")
            return True
        else:
            print("No stored models found. New training will be initiated.")
            return False
    elif user_choice == "n":
        print("New models will be trained.")
        return False
    else:
        print("Invalid input. Default: 'y' (use stored models).")
        if models_exist(models_dir):
            print("Stored models will be used.")
            return True
        else:
            print("No stored models found. New training will be initiated.")
            return False

def train_models(X_train, X_test, y_train, y_test, include_models, use_saved_models, models_dir):
    """
    Dynamically trains models based on the selection in `include_models`.

    Args:
        X_train, X_test, y_train, y_test: Training and test data.
        include_models (list): List of models to be trained.

    Returns:
        list: A list of results (scores and models for each model).
    """
    trained_models_results = []

    if "Logistic Regression" in include_models:
        try:
            result = train_logistic_regression(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:  # Ensure that the result is valid
                trained_models_results.append(result)
            else:
                print(f"Warning: Logistic Regression did not return a valid result.")
        except Exception as e:
            print(f"Error while training Logistic Regression: {e}")

    if "Random Forest" in include_models:
        try:
            result = train_random_forest(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:
                trained_models_results.append(result)
            else:
                print(f"Warning: Random Forest did not return a valid result.")
        except Exception as e:
            print(f"Error while training Random Forest: {e}")

    if "XGBoost" in include_models:
        try:
            result = train_xgboost(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:
                trained_models_results.append(result)
            else:
                print(f"Warning: XGBoost did not return a valid result.")
        except Exception as e:
            print(f"Error while training XGBoost: {e}")

    if "LightGBM" in include_models:
        try:
            result = train_lightgbm(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:
                trained_models_results.append(result)
            else:
                print(f"Warning: LightGBM did not return a valid result.")
        except Exception as e:
            print(f"Error while training LightGBM: {e}")

    if "Neural Network" in include_models:
        try:
            result = train_neural_network(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:
                trained_models_results.append(result)
            else:
                print(f"Warning: Neural Network did not return a valid result.")
        except Exception as e:
            print(f"Error while training Neural Network: {e}")

    return trained_models_results


def train_logistic_regression(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trains or loads Logistic Regression.
    """
    model_file = os.path.join(models_dir, 'logistic_regression.pkl')

    try:
        # Create directory for models if it does not exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Check if a saved model should be loaded
        model = None
        if use_saved_models and os.path.exists(model_file):
            print("[INFO] Loading saved Logistic Regression model...")
            model = joblib.load(model_file)

            # Check if the training data match the feature set of the model
            if X_train.shape[1] != model.n_features_in_:
                print(
                    f"[WARNING] The number of features in the model ({model.n_features_in_}) "
                    f"does not match the input data ({X_train.shape[1]}).")
                print("[INFO] Attempting automatic feature adjustment...")

                # Add missing columns
                missing_columns = set(model.feature_names_in_) - set(X_train.columns)
                for col in missing_columns:
                    X_train[col] = 0
                    X_test[col] = 0

                # Remove extra columns
                X_train = X_train[model.feature_names_in_]
                X_test = X_test[model.feature_names_in_]

        # If no model was loaded, start training
        if model is None:
            print("[INFO] No saved model found. Starting new training...")
            param_grid = {
                'solver': ['lbfgs'],
                'C': [0.1, 1.0, 10],
                'class_weight': ['balanced']
            }
            logreg_pipeline = GridSearchCV(
                LogisticRegression(max_iter=500, random_state=42),
                param_grid=param_grid,
                scoring='roc_auc',
                cv=3,
                n_jobs=-1
            )

            # Train the model
            print("[INFO] Starting GridSearchCV for Logistic Regression...")
            logreg_pipeline.fit(X_train, y_train)
            model = logreg_pipeline.best_estimator_
            print("[INFO] Training completed successfully.")

            # Save column names to avoid dimension mismatch in the future
            if isinstance(X_train, pd.DataFrame):
                model.feature_names_in_ = X_train.columns

            # Save the model
            joblib.dump(model, model_file)
            print(f"[INFO] Model saved: {model_file}")

        # Make predictions and calculate performance metrics
        print("[INFO] Calculating predictions and metrics...")
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        threshold = 0.5  # Default threshold
        y_train_pred = (y_train_proba > threshold).astype(int)
        y_test_pred = (y_test_proba > threshold).astype(int)

        # Calculate performance
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_test_proba)

        result = {
            "model_name": "Logistic Regression",
            "trained_model": model,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy,
            "roc_auc_score": roc_auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "y_test_proba": y_test_proba
        }

        # Print metrics to the console
        print(f"\nTrain Accuracy: {result['train_accuracy']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"ROC-AUC: {result['roc_auc_score']:.4f}")
        print(f"F1-Score: {result['f1_score']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print("=" * 24)

        return result

    except FileNotFoundError as fnfe:
        print(f"[ERROR] File or path not found: {fnfe}")
        raise

    except ValueError as ve:
        print(f"[ERROR] Invalid input data or models: {ve}")
        raise

    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}")
        raise


def train_random_forest(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trains a Random Forest Classifier with adjustable threshold.

    Args:
        X_train, X_test, y_train, y_test: Training and test data.
        use_saved_models: Flag to indicate whether saved models should be used.
        models_dir: Path to save models.

    Returns:
        dict: Result data, including the trained model, scores, and predictions.
    """
    # Ensure that X_train and X_test are DataFrames with identical column names
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("Columns of X_train and X_test do not match.")
    else:
        raise TypeError("X_train and X_test must be DataFrames to ensure consistency.")

    model_file = os.path.join(models_dir, 'random_forest_model.pkl')

    # Check if a saved model exists
    if use_saved_models and os.path.exists(model_file):
        print(f"\nSaved Random Forest model found. Loading...")
        model = joblib.load(model_file)
    else:
        print(f"\nTraining Random Forest...")
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, model_file)
        print(f"Random Forest model saved as '{model_file}'")

    # Calculate probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Adjust threshold for classification
    threshold = 0.5  # You can adjust this value
    y_train_pred = (y_train_proba > threshold).astype(int)
    y_test_pred = (y_test_proba > threshold).astype(int)

    # Calculate additional evaluation metrics
    results = {
        "model_name": "Random Forest",
        "trained_model": model,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "roc_auc_score": roc_auc_score(y_test, y_test_proba),
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "y_test_proba": y_test_proba  # If probabilities need to be saved
    }

    # Print evaluation results
    print(f"\nTrain Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc_score']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print("=" * 24)

    return results

def train_xgboost(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trains an XGBoost classifier.

    Args:
        X_train, X_test, y_train, y_test: Training and test data.
        use_saved_models (bool): Whether to use a saved model.
        models_dir (str): Directory where models should be saved.

    Returns:
        dict: Result data including the trained model, scores, and predictions.
    """
    # Data validation
    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError("Training or test data is not loaded properly. Please check!")
    if len(X_train) == 0 or len(X_test) == 0 or len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Training or test data is empty. Please check!")

    model_file = os.path.join(models_dir, 'xgboost_model.pkl')
    model = None

    # Check if a saved model exists
    if use_saved_models and os.path.exists(model_file):
        try:
            print(f"\nSaved XGBoost model found. Loading...")
            model = xgb.XGBClassifier()
            model.load_model(model_file)
            # Ensure that the loaded model is usable
            if not hasattr(model, 'classes_'):
                print("Warning: The loaded model appears incomplete. Retraining required.")
                model = None
        except Exception as e:
            print(f"Error loading saved model: {e}")
            model = None

    # Train if no valid model was loaded
    if model is None:
        print(f"\nStarting training for XGBoost...")
        model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            raise ValueError(f"Error during model training: {e}")

        # Save the model
        try:
            joblib.dump(model, model_file)  # Save in the specified directory
            print(f"XGBoost model successfully saved in '{model_file}'.")
        except Exception as e:
            print(f"Error saving the model: {e}")

    # Predictions (probabilistic and classification)
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]  # Probabilities for class 1
        y_test_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        raise ValueError(f"Error during probabilistic predictions: {e}")

    # Adjust threshold for classification
    threshold = 0.5  # Default value, can be changed
    y_train_pred = (y_train_proba > threshold).astype(int)
    y_test_pred = (y_test_proba > threshold).astype(int)

    # Check ROC-AUC (only if `predict_proba` is available)
    roc_auc = roc_auc_score(y_test, y_test_proba) if hasattr(model, "predict_proba") else None

    # Evaluation metrics
    results = {
        "model_name": "XGBoost",
        "trained_model": model,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "roc_auc_score": roc_auc,
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "y_test_proba": y_test_proba  # Probabilistic predictions (optional for future analysis)
    }

    # Output evaluation results
    print(f"\nTrain Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {results['roc_auc_score']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print("=" * 24)

    return results


def train_lightgbm(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trains a LightGBM classifier, saves the model and feature names locally.
    If a saved model exists, it is loaded.

    Args:
        X_train, X_test, y_train, y_test: Training and test data.
        use_saved_models: Flag indicating whether to use saved models.
        models_dir: Path to save the model.

    Returns:
        dict: Result data including the trained model, scores, and predictions.
    """

    # Ensure that X_train and X_test are Pandas DataFrames with the same column names
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns)  # Ensure consistent feature names

    # Model and feature name files
    model_file = os.path.join(models_dir, 'lightgbm_model.pkl')
    feature_names_file = os.path.join(models_dir, 'lightgbm_feature_names.pkl')
    model = None

    # Check if the model exists and can be loaded
    if use_saved_models and os.path.exists(model_file):
        try:
            print("\nSaved LightGBM model found. Loading...")
            model = joblib.load(model_file)

            # Load feature names
            if os.path.exists(feature_names_file):
                with open(feature_names_file, 'rb') as f:
                    saved_feature_names = joblib.load(f)

                # Ensure that the loaded feature names match the current ones
                if list(X_train.columns) != saved_feature_names:
                    raise ValueError("Feature names from the saved data do not match X_train!")
        except Exception as e:
            print(f"Error loading the saved model or data: {e}")
            model = None

    # Train the model if none was loaded
    if model is None:
        print("\nNo model found. Starting training for LightGBM...")
        model = LGBMClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)

        # Save the model and feature names
        try:
            joblib.dump(model, model_file)
            print(f"LightGBM model saved as '{model_file}'")

            with open(feature_names_file, 'wb') as f:
                joblib.dump(list(X_train.columns), f)
                print(f"Feature names saved as '{feature_names_file}'")
        except Exception as e:
            print(f"Error saving the model or feature names: {e}")

    # Predictions and probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]  # Class probabilities
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Classification threshold
    threshold = 0.5
    y_train_pred = (y_train_proba > threshold).astype(int)
    y_test_pred = (y_test_proba > threshold).astype(int)

    # Compute additional metrics
    roc_auc = roc_auc_score(y_test, y_test_proba) if hasattr(model, "predict_proba") else None
    results = {
        "model_name": "LightGBM",
        "trained_model": model,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "roc_auc_score": roc_auc,
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "y_test_proba": y_test_proba
    }

    # Output results
    print(f"\nTrain Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc_score']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print("=" * 24)

    return results

def train_neural_network(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trains a neural network or loads a saved model if available.
    """
    model_file = os.path.join(models_dir, 'nn_model.keras')

    # Check if a saved model exists
    if use_saved_models and os.path.exists(model_file):
        print(f"\nSaved Keras model found. Loading...")
        model = load_model(model_file)

        # Check if input dimensions match
        input_shape = model.input_shape[1]
        if input_shape != X_train.shape[1]:
            raise ValueError(f"The saved model expects input dimension {input_shape}, "
                             f"but the current data has {X_train.shape[1]} features.")
    else:
        print(f"\nStarting training for a neural network...")
        feature_count = X_train.shape[1]

        # Step 3: Build the model
        model = Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

        try:
            # Train the model
            model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=0)
        except Exception as e:
            print(f"Error during neural network training: {e}")
            return None

        # Save the model
        model.save(model_file)
        print(f"Keras model saved as '{model_file}'")

    # Calculate probabilities
    y_train_proba = model.predict(X_train).ravel()
    y_test_proba = model.predict(X_test).ravel()

    # Check probability distribution
    plt.hist(y_test_proba, bins=30, alpha=0.8, color='blue')
    plt.title("Histogram of Predicted Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.show()

    # Dynamic threshold adjustment (e.g., based on Precision-Recall)
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
    # Select the threshold that guarantees Precision > 0.6
    threshold = thresholds[np.argmax(precision > 0.6)] if np.any(precision > 0.6) else 0.5
    print(f"Automatically adjusted threshold: {threshold:.2f}")

    y_train_pred = (y_train_proba > threshold).astype(int)
    y_test_pred = (y_test_proba > threshold).astype(int)

    # Model evaluation
    result = {
        "model_name": "Neural Network",
        "trained_model": model,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "roc_auc_score": roc_auc_score(y_test, y_test_proba),
        "y_test_proba": y_test_proba,
        "f1_score": f1_score(y_test, y_test_pred, zero_division=1),
        "precision": precision_score(y_test, y_test_pred, zero_division=1),
        "recall": recall_score(y_test, y_test_pred, zero_division=1),
    }

    # Check output: Validate presence of critical cases
    critical_count = np.sum(y_test_pred)
    if critical_count == 0:
        print("\nWARNING: No critical cases found. Please check data and model!")
    else:
        print(f"Critical cases identified: {critical_count}")

    # Display evaluation results
    print(f"\nTraining Accuracy: {result['train_accuracy']:.4f}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"ROC-AUC: {result['roc_auc_score']:.4f}")
    print(f"F1-Score: {result['f1_score']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print("=" * 24)

    return result


def evaluate_models(models, X_test, y_test, plot_dir="plots"):
    """
    Evaluates ML models and creates confusion matrix subplots.
    """
    evaluation_results = {
        "Model": [],
        "ROC-AUC": [],
        "F1-Score": [],
        "Precision": [],
        "Recall": [],
        "Accuracy": [],
        "MCC": [],
        "Log-Loss": [],
        "TP": [],
        "TN": [],
        "FP": [],
        "FN": []
    }

    # Evaluate models
    for name, model in models.items():
        if model is None:
            print(f"Warning: Model '{name}' is None and will be skipped.")
            continue
        try:
            # Generate predictions
            if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # For models without `predict_proba`:
                y_proba = model.predict(X_test)
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 0]

            # Binary classification based on 0.5 threshold
            y_pred = (y_proba >= 0.5).astype(int)

            # Compute metrics
            roc_auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            log_loss_val = log_loss(y_test, y_proba)

            # Compute confusion matrix -> TN, FP, FN, TP
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Save results
            evaluation_results["Model"].append(name)
            evaluation_results["ROC-AUC"].append(roc_auc)
            evaluation_results["F1-Score"].append(f1)
            evaluation_results["Precision"].append(precision)
            evaluation_results["Recall"].append(recall)
            evaluation_results["Accuracy"].append(accuracy)
            evaluation_results["MCC"].append(mcc)
            evaluation_results["Log-Loss"].append(log_loss_val)
            evaluation_results["TP"].append(tp)
            evaluation_results["TN"].append(tn)
            evaluation_results["FP"].append(fp)
            evaluation_results["FN"].append(fn)

            print(f"Model '{name}' evaluated:")
            print(f"  ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"  Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}, Log-Loss: {log_loss_val:.4f}")
            print(f"  Confusion Matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")

        except Exception as e:
            print(f"Error while evaluating model '{name}': {e}")

    results_df = pd.DataFrame(evaluation_results)

    try:
        model_names = results_df["Model"]
        tp = results_df["TP"]
        tn = results_df["TN"]
        fp = results_df["FP"]
        fn = results_df["FN"]

        num_models = len(model_names)
        fig, axes = plt.subplots(num_models, 1, figsize=(8, 6 * num_models), squeeze=False)

        for i, ax in enumerate(axes.flatten()):
            matrix = np.array([[tn[i], fp[i]], [fn[i], tp[i]]])
            ax.imshow(matrix, interpolation="nearest", cmap="Blues")  # Direct use of color scheme

            ax.set_title(f"Confusion Matrix - {model_names[i]}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Negative", "Positive"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Negative", "Positive"])

            # Add text inside each matrix cell
            for j in range(2):
                for k in range(2):
                    ax.text(k, j, format(matrix[j, k], "d"),
                            horizontalalignment="center",
                            color="white" if matrix[j, k] > matrix.max() / 2 else "black")  # Dynamic color

        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "combined_confusion_matrices.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()

    except Exception as e:
        print(f"Error creating confusion matrix plot: {e}")

    return results_df

def plot_horizontal_comparison(
        model_names,
        train_accuracies,
        test_accuracies,
        roc_auc_scores,
        plot_dir,
        f1_scores=None,
        precisions=None,
        recalls=None,
        sort_by="Test Accuracy",
        include_train=True,
        bar_colors=("skyblue", "steelblue", "lightgreen", "orange", "purple", "brown"),
        xlabel="Performance",
        legend_labels=("Train Accuracy", "Test Accuracy", "ROC AUC", "F1 Score", "Precision", "Recall")
):
    """
    Creates a horizontal bar chart to compare model performance.

    Parameters:
    -----------
    model_names : list
        List of model names.
    train_accuracies : list
        List of training accuracies (optional to display).
    test_accuracies : list
        List of testing accuracies.
    roc_auc_scores : list
        List of ROC AUC scores.
    f1_scores : list, optional
        List of F1-scores, if available.
    precisions : list, optional
        List of precision values, if available.
    recalls : list, optional
        List of recall values, if available.
    plot_dir : str
        Directory to save the chart.
    sort_by : str
        Metric to sort the models by (`"Train Accuracy"`, `"Test Accuracy"`, `"ROC AUC"`,
        `"F1 Score"`, `"Precision"`, `"Recall"`).
    include_train : bool
        Whether to display `Train Accuracy` (optional).
    bar_colors : tuple
        Colors for the bars (Train, Test, ROC AUC, F1 Score, Precision, Recall).
    xlabel : str
        Label of the x-axis (can be adapted for dynamic language).
    legend_labels : tuple
        Legend labels for Train, Test, ROC AUC, F1 Score, Precision, and Recall.
    """
    # Check for the directory
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Prepare data (add new metrics if available)
    data = pd.DataFrame({
        "Model": model_names,
        "Train Accuracy": train_accuracies,
        "Test Accuracy": test_accuracies,
        "ROC AUC": roc_auc_scores,
    })

    # Optionally add additional columns if data is available
    if f1_scores:
        data["F1 Score"] = f1_scores
    if precisions:
        data["Precision"] = precisions
    if recalls:
        data["Recall"] = recalls

    # Dynamic sorting of models
    if sort_by in data.columns:
        data = data.sort_values(by=sort_by, ascending=False)  # Sort descending
    else:
        raise ValueError(f"Invalid sort_by parameter: {sort_by}. Must be one of {', '.join(data.columns)}")

    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.10  # Width of the bars for different metrics
    y_pos = np.arange(len(data))  # Y-positions for the bars

    # Bar plots by metric
    if include_train:
        ax.barh(y_pos - 2 * bar_width, data["Train Accuracy"], bar_width, label=legend_labels[0], color=bar_colors[0])
    ax.barh(y_pos - bar_width, data["Test Accuracy"], bar_width, label=legend_labels[1], color=bar_colors[1])
    ax.barh(y_pos, data["ROC AUC"], bar_width, label=legend_labels[2], color=bar_colors[2])

    # Optionally insert additional metrics
    if "F1 Score" in data.columns:
        ax.barh(y_pos + bar_width, data["F1 Score"], bar_width, label=legend_labels[3], color=bar_colors[3])
    if "Precision" in data.columns:
        ax.barh(y_pos + 2 * bar_width, data["Precision"], bar_width, label=legend_labels[4], color=bar_colors[4])
    if "Recall" in data.columns:
        ax.barh(y_pos + 3 * bar_width, data["Recall"], bar_width, label=legend_labels[5], color=bar_colors[5])

    # Labels, title, and legend
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["Model"])
    ax.set_title("Model Comparison")
    ax.set_xlabel(xlabel)
    ax.legend()
    plt.tight_layout()

    # Save chart
    output_path = os.path.join(plot_dir, "model_performance_comparison.png")
    plt.savefig(output_path)
    plt.show()

    print(f"Chart saved at: {output_path}")


def plot_combined_roc_curves(models, X_test, y_test, plot_dir, nn_model=None, y_proba_nn=None):
    """
    Creates a plot for ROC curves of all models, including the neural network.
    """
    try:
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(16, 10))
        valid_models = False  # Check if at least one model is successful

        # Plot traditional models
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                # Retrieve probabilities
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"ROC-{name} (AUC={roc_auc:.2f})")
                valid_models = True

        # Plot for the neural network
        if nn_model and y_proba_nn is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba_nn)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"ROC-NeuralNet (AUC={roc_auc:.2f})", linestyle="--", color="purple")
            valid_models = True

        # Random classification line
        if valid_models:
            plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")
            plt.legend(loc="lower right")
            plt.title("ROC Curves")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, "all_roc_curves.png"))
            plt.show()
        else:
            print("No valid models found for ROC curves.")

    except Exception as e:
        print(f"Error creating ROC plot: {e}")


def test_under_over_fit(model_names, train_accuracies, test_accuracies, overfit_threshold=0.1, underfit_threshold=0.6):
    """
    Identifies overfitting and underfitting for given models based on accuracy values.

    Args:
        model_names (list): List of model names.
        train_accuracies (list): Training accuracies of the models.
        test_accuracies (list): Testing accuracies of the models.
        overfit_threshold (float): Threshold for overfitting (Train-Test difference).
        underfit_threshold (float): Threshold for underfitting (Train and Test accuracy < Threshold).

    Returns:
        pd.DataFrame: DataFrame with results ("Model", "Overfit", "Underfit").
    """
    results = {"Model": [], "Overfit": [], "Underfit": []}

    for name, train, test in zip(model_names, train_accuracies, test_accuracies):
        overfit = (train - test) > overfit_threshold
        underfit = (train < underfit_threshold) and (test < underfit_threshold)

        results["Model"].append(name)
        results["Overfit"].append(overfit)
        results["Underfit"].append(underfit)

        # Optionally save output in variables (for debugging)
        log_message = (
            f"Model: {name}\n"
            f"  Overfit: {overfit} (Train-Test Difference: {train - test:.3f})\n"
            f"  Underfit: {underfit} (Train Accuracy: {train:.3f}, Test Accuracy: {test:.3f})\n"
        )

    # Convert results to DataFrame
    result_df = pd.DataFrame(results)

    return result_df

def plot_under_over_fit(model_names, train_accuracies, test_accuracies, plot_dir):
    """
    Visualizes training and testing differences to evaluate overfitting or underfitting.

    Args:
        model_names (list): List of model names.
        train_accuracies (list): Training accuracies.
        test_accuracies (list): Testing accuracies.
        plot_dir (str): Directory to save the generated plot.
    """
    try:
        differences = np.array(train_accuracies) - np.array(test_accuracies)
        colors = ["green" if -0.1 <= d <= 0.1 else "red" if d > 0.1 else "orange" for d in differences]

        data = pd.DataFrame({
            "Model": model_names,
            "Train Accuracy": train_accuracies,
            "Test Accuracy": test_accuracies,
            "Train-Test Difference": differences
        }).sort_values(by="Train-Test Difference", ascending=False)

        plt.figure(figsize=(12, 8))
        bars = plt.barh(data["Model"], data["Train-Test Difference"], color=colors)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1, label="Balanced Fit")
        plt.xlabel("Train-Test Accuracy Difference", fontsize=12)
        plt.ylabel("Models", fontsize=12)
        plt.title("Model Analysis: Underfitting vs Overfitting", fontsize=14)

        for bar, diff, train, test in zip(bars, data["Train-Test Difference"], data["Train Accuracy"],
                                          data["Test Accuracy"]):
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f"Diff: {diff:.2f}\nTrain: {train:.2f}\nTest: {test:.2f}",
                     va='center', ha='left', fontsize=9)

        plt.legend()
        plt.tight_layout()
        plt.gca().invert_yaxis()

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "under_over_fit_analysis.png")
        plt.savefig(plot_path)
        plt.show()

        print(f"Under-/Overfitting analysis saved at: {plot_path}")

    except Exception as e:
        print(f"Error in the 'plot_under_over_fit' function: {e}")


def get_best_model(models, evaluation_results, fit_results, primary_metric="ROC-AUC", tie_breaker="F1-Score"):
    """
    Selects the best model based on a primary metric while excluding models with overfitting/underfitting.

    Args:
        models (dict): Dictionary of trained models.
        evaluation_results (pd.DataFrame): DataFrame containing the metrics of all models.
        fit_results (pd.DataFrame): DataFrame containing the fit analysis results ("Overfit", "Underfit").
        primary_metric (str): Primary metric for selecting the best model (e.g., "ROC-AUC").
        tie_breaker (str): Secondary metric used in case of ties.

    Returns:
        tuple: Best model and the name of the model (as String).
    """
    # Combine fit results with model metrics
    if "Model" not in fit_results.columns or "Overfit" not in fit_results.columns or "Underfit" not in fit_results.columns:
        raise ValueError("The fit results must contain the columns 'Model', 'Overfit', and 'Underfit'.")

    combined_results = pd.merge(evaluation_results, fit_results, on="Model", how="inner")

    # Filter out models with overfitting or underfitting
    valid_models = combined_results[(combined_results["Overfit"] == False) & (combined_results["Underfit"] == False)]

    if valid_models.empty:
        raise ValueError("No models meet the criteria: No overfitting and no underfitting.")

    # Select model based on primary metric
    if primary_metric not in valid_models.columns:
        raise ValueError(
            f"The metric '{primary_metric}' does not exist in the results. Available metrics: {valid_models.columns.tolist()}")

    best_model_data = valid_models.loc[valid_models[primary_metric].idxmax()]
    top_candidates = valid_models[valid_models[primary_metric] == best_model_data[primary_metric]]

    # Use secondary metric in case of ties
    if len(top_candidates) > 1 and tie_breaker in valid_models.columns:
        top_candidates = top_candidates.sort_values(by=tie_breaker, ascending=False)
        best_model_data = top_candidates.iloc[0]

    # Extract name and model
    best_model_name = best_model_data["Model"]
    best_model = models[best_model_name]

    print(
        f"The best model based on '{primary_metric}' is '{best_model_name}' with a score of {best_model_data[primary_metric]:.4f}.")
    if len(top_candidates) > 1:
        print(f"  (Tie resolved using '{tie_breaker}').")
    return best_model, best_model_name


def get_critical_employees(model, X_transformed, df,
                           feature_names_file="Models/lightgbm_feature_names.pkl", threshold=0.0):
    """
    Identifies critical employees with a churn probability higher than a threshold.

    Works for LightGBM models and other sklearn-like models.

    Args:
        model: The trained model for prediction.
        X_transformed (np.ndarray): Preprocessed features (unscaled).
        df (pd.DataFrame): DataFrame with additional employee information.

        feature_names_file (str): Path to the saved feature names (for LightGBM).
        threshold (float): The threshold for the churn probability.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Critical employees and the top 15 employees by probability.
    """

    print(f"Shape of X_transformed: {X_transformed.shape}")
    print(f"Shape of df: {df.shape}")

    # Dynamically determine the last month and year from the dataset
    max_year = df["Year"].max()  # Highest year in the dataset
    max_month_in_max_year = df[df["Year"] == max_year]["Month"].max()  # Highest month in the highest year
    print(f"Last year in the dataset: {max_year}, last month in the year: {max_month_in_max_year}")

    # ** Load feature names and apply to input data (only necessary for LightGBM) **
    if isinstance(model, lgb.LGBMClassifier) and feature_names_file:
        try:
            with open(feature_names_file, 'rb') as f:
                feature_names = joblib.load(f)
            print(f"Feature names successfully loaded: {feature_names}")

            # Convert X_transformed to DataFrame with correct column names
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        except Exception as e:
            print(f"Error loading or applying feature names: {e}")
            return None, None

    # ** Calculate predictions **
    print("Calculating predictions...")
    try:
        if hasattr(model, "predict_proba"):
            # For sklearn-like models including LightGBM
            y_proba = model.predict_proba(X_transformed)[:, 1]
        elif hasattr(model, "predict"):
            # For models like Keras (without predict_proba)
            y_proba = model.predict(X_transformed).flatten()
        else:
            raise AttributeError(f"Model type '{type(model)}' is not supported.")
    except Exception as e:
        print(f"Error during predictions: {e}")
        return None, None

    # ** Link probabilities with data **
    try:
        # Convert churn probability to percentages
        df["Churn Probability"] = y_proba * 100
    except ValueError as e:
        print(f"Error adding churn probability: {e}")
        return None, None

    # ** Identify critical employees **
    try:
        # Filter: Only employees active (0 churn) in the last month/year
        df = df[
            (df["Turnover"] == 0) &
            (df["Month"] == max_month_in_max_year) &
            (df["Year"] == max_year)
            ]
        print(f"Shape after filtering by churn, month, and year: {df.shape}")

        critical_employees = df[df["Churn Probability"] > threshold]
        critical_employees = critical_employees.sort_values(
            by="Churn Probability", ascending=False
        )

        # Remove duplicates based on employee ID
        if "Employee_ID" in critical_employees.columns:
            duplicate_count = critical_employees.duplicated(subset=["Employee_ID"]).sum()
            if duplicate_count > 0:
                print(f"Warning: {duplicate_count} duplicate entries based on 'Employee_ID' were removed.")
                critical_employees = critical_employees.drop_duplicates(subset=["Employee_ID"])

        # Top 15 critical employees
        top_15_employees = critical_employees.head(15)
    except Exception as e:
        print(f"Error sorting critical employees: {e}")
        return None, None

    # ** Return results **
    print("Critical employees successfully identified.")
    return critical_employees, top_15_employees

def get_critical_employees_all_models(models, X_transformed, df,
                                      feature_names_file="Models/lightgbm_feature_names.pkl", threshold=0.0):
    """
    Identifies critical employees and gathers results for multiple models.
    Supports LightGBM with stored feature names.

    Args:
        models (dict): A dictionary of models, where keys are model names.
        X_transformed (np.ndarray): The preprocessed features (unscaled).
        df (pd.DataFrame): The DataFrame with employee information.
        feature_names_file (str): Path to the stored feature names (for LightGBM).
        threshold (float): The threshold for churn probability.

    Returns:
        dict: A dictionary with model names as keys and tuple results as values.
    """
    results = {}

    print(f"\nShape of X_transformed: {X_transformed.shape}")
    print(f"Shape of df: {df.shape}")

    # Dynamically determine the last month and year from the DataFrame
    max_year = df["Year"].max()
    max_month_in_max_year = df[df["Year"] == max_year]["Month"].max()
    print(f"Last year in the dataset: {max_year}, last month in the year: {max_month_in_max_year}")

    # ** Synchronize between X_transformed and df **
    if len(X_transformed) != len(df):
        print("WARNING: Dimensions of X_transformed and df do not match. Synchronizing data...")
        df = df.iloc[:len(X_transformed)].reset_index(drop=True).copy()

    # ** Iterate over all models **
    for model_name, model in models.items():
        print(f"\nProcessing model: {model_name}")
        try:
            # Reset transformed features for the model
            X_model_transformed = X_transformed.copy()

            # ** Special handling for LightGBM **
            if isinstance(model, lgb.LGBMClassifier):
                # Apply specific feature names for LightGBM
                if feature_names_file:
                    try:
                        with open(feature_names_file, 'rb') as f:
                            feature_names = joblib.load(f)
                        print(f"Feature names loaded successfully: {feature_names}")

                        # Convert to DataFrame and remove non-numeric columns
                        X_model_transformed = pd.DataFrame(X_model_transformed, columns=feature_names)
                    except Exception as e:
                        print(f"Error loading or applying feature names for '{model_name}': {e}")
                        results[model_name] = (pd.DataFrame(), pd.DataFrame())
                        continue

            # ** Ensure that only numeric data is used for prediction **
            if isinstance(X_model_transformed, pd.DataFrame):
                X_model_transformed = X_model_transformed.select_dtypes(include=[np.number])

            # ** Calculate predictions **
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_model_transformed)[:, 1]
                elif hasattr(model, "predict"):
                    y_proba = model.predict(X_model_transformed).flatten()
                else:
                    print(f"The model '{model_name}' does not support 'predict_proba' or 'predict'.")
                    results[model_name] = (pd.DataFrame(), pd.DataFrame())
                    continue
            except Exception as e:
                print(f"Error during prediction for model '{model_name}': {e}")
                results[model_name] = (pd.DataFrame(), pd.DataFrame())
                continue

            # ** Collect results **
            df_copy = df.copy()
            df_copy["Churn Probability"] = y_proba * 100

            # Filter data for the last month and year
            df_copy = df_copy[
                (df_copy["Turnover"] == 0) &
                (df_copy["Month"] == max_month_in_max_year) &
                (df_copy["Year"] == max_year)
                ]

            # Determine critical employees and top 15
            critical_employees = df_copy[df_copy["Churn Probability"] > threshold]
            critical_employees = critical_employees.sort_values(by="Churn Probability", ascending=False)
            critical_employees = critical_employees.drop_duplicates(subset=["Employee_ID"], keep="first")

            top_15 = critical_employees.head(15)

            # Save results
            results[model_name] = (critical_employees, top_15)

        except Exception as e:
            print(f"Error with model '{model_name}': {e}")
            results[model_name] = (pd.DataFrame(), pd.DataFrame())

    return results


def save_results(data, file_name_base, output_dir):
    """
    Saves results as both CSV and Excel in the specified directory.

    Args:
        data (pd.DataFrame): The data to save.
        file_name_base (str): Base file name (without extension).
        output_dir (str): Destination directory for saving.
    """
    csv_path = os.path.join(output_dir, f"{file_name_base}.csv")
    # xlsx_path = os.path.join(output_dir, f"{file_name_base}.xlsx")

    # Export data
    data.to_csv(csv_path, index=False)
    # data.to_excel(xlsx_path, index=False)

    # Feedback to the user
    print(f"\nData '{file_name_base}' has been saved successfully:")
    print(f"- CSV: {csv_path}")
    # print(f"- Excel: {xlsx_path}")


def compare_model_top_employees(file_paths, output_file="Outputs/Comparison_Top_15.csv"):
    """
    Loads the top-15 employee files for various models, compares names and IDs,
    counts how often a combination of name and ID appears in different models,
    provides churn probabilities per model in percentages, and creates a compact summary.

    Args:
        file_paths (dict): Dictionary with model names as keys and file paths as values.
        output_file (str): Name of the output file to save the comparison.

    Returns:
        pd.DataFrame: Compact DataFrame with Name, ID, Counts, Churn Probabilities (in percentages), and Match.
    """
    model_data = {}

    print("Loading top-15 employee files...")
    for model, file_path in file_paths.items():
        try:
            print(f"Loading file for model '{model}' from {file_path}...")
            df = pd.read_csv(file_path)

            # Check necessary columns
            required_columns = {"Name", "Employee_ID", "Churn Probability"}
            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"The file '{file_path}' does not contain the required columns: {required_columns}."
                )

            # Select only required columns
            df = df[["Name", "Employee_ID", "Churn Probability"]].copy()

            # Rename column to include model information
            df.rename(columns={"Churn Probability": f"Churn_{model} (%)"}, inplace=True)
            model_data[model] = df
        except Exception as e:
            print(f"Error loading from {file_path}: {e}")
            continue

    # Combine all loaded data
    print("\nCombining data from all models...")
    combined_df = None
    for model, df in model_data.items():
        if combined_df is None:
            combined_df = df
        else:
            # Merge based on Name and Employee_ID
            combined_df = pd.merge(combined_df, df, on=["Name", "Employee_ID"], how="outer")

    # Count matches
    print("\nCounting matches...")
    combined_df["Frequency_in_Models"] = combined_df.notna().sum(axis=1) - 2  # Ignore 'Name' and 'Employee_ID'

    # Check for identical entries across multiple models
    combined_df["Match"] = combined_df["Frequency_in_Models"] > 1

    # Sort data alphabetically by Name and ID
    combined_df = combined_df.sort_values(by=["Name", "Employee_ID"]).reset_index(drop=True)

    # Save the results
    print(f"Saving results to '{output_file}'...")
    try:
        combined_df.to_csv(output_file, index=False)
        print("Save process completed.")
    except Exception as e:
        print(f"Error during saving: {e}")

    return combined_df

# Main logic for churn analysis and model training
def main():
    """
    Main logic for churn analysis and model training.
    """
    plot_dir = "plots"
    models_dir = "models"
    output_dir = "outputs"
    output_dir_all = "outputs/all_models"
    os.makedirs(plot_dir, exist_ok=True)  # Create directory for plots
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_all, exist_ok=True)
    # Create directories if they don’t exist

    # 1. Load and prepare data
    print("\n### Step 1: Load data ###")
    print("Loading data...")
    file_path = "../data/HR_cleaned.csv"
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        df = load_data(file_path)
        print("Data loading complete.")

    except Exception as e:
        print(f"Error while loading data: {e}")
        return

    # 2. Preprocess data (One-Hot Encoding and SMOTE)
    # Function removes 'Status' and creates 'Churn'
    print("\n### Step 2: Preprocess data ###")
    print("Preprocessing data...")
    df, X_transformed, X_resampled, y_resampled, preprocessor = preprocess_data(df)
    print(f"Shape of X_transformed: {X_transformed.shape}")
    print(f"Shape of df (original data): {df.shape}")
    print("Preprocessing complete.")

    # 3. Split and scale data (create training/test sets and scale)
    print("\n### Step 3: Split and scale data ###")
    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = split(X_resampled, y_resampled)
    print(f"Shape of X_train_scaled: {X_train.shape}")
    print(f"Shape of X_test_scaled: {X_test.shape}")
    print("Data splitting and scaling complete.")

    # 4. Model selection based on requirements
    print("\n### Step 4: Model selection ###")
    print("Performing model selection...")
    include_models = model_selection()  # Function determines which models are to be trained
    include_models_all = include_models
    if len(include_models) == 1:
        print(f"Selected model: {include_models[0]}")
    else:
        print(f"Selected models: {', '.join(include_models)}")

    # 5. Query user choice
    print("\n### Step 5: Choose: Saved models or new training ###")
    use_saved_models = get_user_choice(models_dir)

    # 6. Train models based on the choice
    print("\n### Step 6: Train models ###")
    print("Training models...")

    # Train or load models
    trained_models_results = train_models(
        X_train, X_test, y_train, y_test, include_models, use_saved_models, models_dir)

    # Ensure results exist
    if not trained_models_results or len(trained_models_results) == 0:
        # Check if any models exist at all
        if use_saved_models:
            print("WARNING: Saved models were requested, but no saved models are available.")
            print("Starting new training...")
            use_saved_models = False  # Switch to training new models
            trained_models_results = train_models(
                X_train, X_test, y_train, y_test, include_models, use_saved_models, models_dir)

        # Check again after retry
        if not trained_models_results or len(trained_models_results) == 0:
            print("ERROR: No models were successfully trained or loaded. Program will exit.")
            return

    # Extract models into a structured dictionary
    models = {}
    y_probas = {}  # To store probabilities
    nn_model = None  # Holds the Neural Network model if available

    for result in trained_models_results:
        # Check model-specific results
        model_name = result.get("model_name")
        trained_model = result.get("trained_model")
        roc_auc_score_nn = result.get("y_test_proba", None)

        # Save model
        if model_name and trained_model:
            models[model_name] = trained_model

        # Save probabilities specifically for Neural Network
        if model_name == "Neural Network":
            nn_model = trained_model  # Assign the neural model
            if roc_auc_score_nn is not None:
                y_probas["Neural Network"] = result["y_test_proba"]  # Save ROC probabilities
        else:
            y_probas[model_name] = result.get("roc_auc_score")  # Check for ROC directly for other models

    # Check if any models have been trained
    if not models:
        print("WARNING: No models are ready for analysis.")
        return  # Exit if no model is available

    print(f"{len(models)} models successfully trained.")

    # 8. Overfitting/Underfitting analysis
    print("\n### Step 8: Over-/Underfitting Analysis and Create Plots ###")

    # Define variables
    model_names = [result["model_name"] for result in trained_models_results]  # Model names
    train_accuracies = [result["train_accuracy"] for result in trained_models_results]  # Training accuracies
    test_accuracies = [result["test_accuracy"] for result in trained_models_results]  # Test accuracies
    roc_auc_scores = [result["roc_auc_score"] for result in trained_models_results]  # ROC AUC values

    # Optional metrics extraction
    f1_scores = [result.get("f1_score", None) for result in trained_models_results]
    precisions = [result.get("precision", None) for result in trained_models_results]
    recalls = [result.get("recall", None) for result in trained_models_results]

    # Create horizontal bar chart for model comparison
    print("\nCreating horizontal bar chart for model comparisons...")
    plot_horizontal_comparison(
        model_names=model_names,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        roc_auc_scores=roc_auc_scores,
        f1_scores=f1_scores,
        precisions=precisions,
        recalls=recalls,
        plot_dir=plot_dir,
        sort_by="Test Accuracy",  # Sort by test accuracy
        include_train=True,  # Include training accuracy
        xlabel="Model Performance (Higher is Better)"
    )
    print("Horizontal plots successfully created.")

    # Test and visualize over-/underfitting results
    print("\nAnalyzing overfitting and underfitting...")
    fit_results = test_under_over_fit(model_names, train_accuracies, test_accuracies)
    plot_under_over_fit(model_names, train_accuracies, test_accuracies, plot_dir)
    print("Over-/Underfitting results:")
    print(fit_results)

    # 9. Create ROC curves for all models
    print("\n### Step 9: Create ROC Curves ###")
    print("Creating ROC curves...")
    try:
        plot_combined_roc_curves(
            models=models,
            X_test=X_test,
            y_test=y_test,
            nn_model=nn_model,  # Neural network model
            y_proba_nn=y_probas.get("Neural Network", None),  # Probabilities for nn
            plot_dir=plot_dir
        )
    except Exception as e:
        print(f"Error creating ROC curves: {e}")

    # 10. Evaluate models
    print("\n### Step 10: Model Evaluation ###")
    print("Evaluating models...")
    evaluation_results = evaluate_models(
        models=models,
        X_test=X_test,  # Original test data
        y_test=y_test,  # True labels
        plot_dir=plot_dir  # Directory for plots
    )

    # 11. Select the best model
    print("\n### Step 11: Model Selection ###")
    print("\nSelecting the best model...")
    try:
        best_model, best_model_name = get_best_model(models, evaluation_results, fit_results, primary_metric="ROC-AUC",
                                                     tie_breaker="F1-Score")
        print(f"\nThe best model is: {best_model_name}")
    except ValueError as e:
        print(f"Error during model selection: {e}")
        return

    # 12. Identify critical employees and top 15
    print("\n### Step 12: Identify Critical Employees and Top 15 ###")
    print("Identifying critical employees and top 15...")
    try:
        # Function call for the selected best model
        critical_employees, top_15_employees = get_critical_employees(
            best_model,
            X_transformed,
            df
        )

        if critical_employees.empty:
            print("WARNING: No critical employees found (no probabilities > 70%).")
        else:
            print(f"Number of critical employees: {len(critical_employees)}")

        if top_15_employees.empty:
            print("WARNING: No top 15 employees found.")
        else:
            print(
                f"Top 15 Employees:\n{top_15_employees[['Employee_ID', 'Name', 'Churn Probability']]}")
    except Exception as e:
        print(f"Error identifying critical employees: {e}")
        return

    # 13. Identify critical employees and top 15 for each model
    print("\n### Step 13: Identify Critical Employees and Top 15 for Each Model ###")
    print("Identifying critical employees and top 15 for each model...")

    try:
        for model_name in include_models_all:  # Use model names from the selection
            try:
                print(f"\nProcessing model: {model_name}")
                # Function call for critical employees
                # Pass models as a dictionary with only the current model
                critical_employees_data = get_critical_employees_all_models(
                    models={model_name: models[model_name]},  # Process only the current model
                    X_transformed=X_transformed,
                    df=df
                )

                # Extract results
                if model_name in critical_employees_data:
                    critical_employees, top_15_employees = critical_employees_data[model_name]

                    # Validate and display results
                    if critical_employees.empty:
                        print(f"WARNING: No critical employees found for {model_name}.")
                    else:
                        print(f"Number of critical employees for {model_name}: {len(critical_employees)}")
                        print(critical_employees.head(5))

                    if top_15_employees.empty:
                        print(f"WARNING: No top 15 employees found for {model_name}.")
                    else:
                        print(f"Top 15 Employees for {model_name}:\n"
                              f"{top_15_employees[['Employee_ID', 'Name', 'Churn Probability']]}")

                    # Save results
                    save_results(
                        data=critical_employees,
                        file_name_base=f"Critical_Employees_{model_name.replace(' ', '_')}",
                        output_dir=output_dir_all
                    )
                    save_results(
                        data=top_15_employees,
                        file_name_base=f"Top_15_Employees_{model_name.replace(' ', '_')}",
                        output_dir=output_dir_all
                    )
                else:
                    print(f"WARNING: No results available for {model_name}.")

            except KeyError:
                print(f"ERROR: Model '{model_name}' was not found in 'models'.")
            except Exception as e:
                print(f"Error processing model '{model_name}': {e}")

    except Exception as e:
        print(f"Error identifying critical employees and top 15 for all models: {e}")

    # 13. Save results
    print("\n### Step 13: Save Results ###")
    print("Saving results...")
    save_results(critical_employees, "Critical_Employees", output_dir)
    save_results(top_15_employees, "Top_15_Employees", output_dir)

    print("### Process Completed: Results Saved ###")
    print("### Program Terminated ###")

    # 14. Compare models
    print("\n### Step 14: Compare Models ###")
    file_paths = {
        "LightGBM": "Outputs/all_models/Top_15_Employees_LightGBM.csv",
        "Logistic_Regression": "Outputs/all_models/Top_15_Employees_Logistic_Regression.csv",
        "Neural_Network": "Outputs/all_models/Top_15_Employees_Neural_Network.csv",
        "Random_Forest": "Outputs/all_models/Top_15_Employees_Random_Forest.csv",
        "XGBoost": "Outputs/all_models/Top_15_Employees_XGBoost.csv"
    }

    result = compare_model_top_employees(file_paths)
    print(result)


# Execute the main function
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    start_time = time.time()
    main()
    plt.close("all")
    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds.")