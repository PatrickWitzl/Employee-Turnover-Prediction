import os
import joblib
import pandas as pd
import xgboost as xgb
from data_loading import load_dataset
from data_cleaning import clean_dataset
from ML1_Fluctuation_best_model_6_ohne_pca import preprocess_data
from ML1_Fluctuation_best_model_6_ohne_pca import get_critical_employees

def process_and_identify_critical_employees(
        file_path,  # Eingabedatei
        save_filtered_path= None,  # Speicherort für gefilterte Daten
        models_dir= "Models",  # Verzeichnis, in dem das Modell gespeichert ist
        model_file_name="xgboost_model.pkl",  # Name der Modell-Datei
        feature_names_file="Models/lightgbm_feature_names.pkl",  # Datei mit Feature-Namen
        threshold=0.0  # Schwellenwert für kritische Mitarbeiter
):
    """
    Diese Funktion übernimmt die Datenvorverarbeitung, das Modell-Laden und die Identifikation
    kritischer Mitarbeiter basierend auf einem Vorhersagemodell.

    Args:
        file_path (str): Pfad zur Eingabedatei (CSV).
        save_filtered_path (str): Speicherort für die gefilterte Eingabedatei.
        models_dir (str): Verzeichnis, in dem das trainierte Modell gespeichert ist.
        model_file_name (str): Name der Datei, in der das Modell gespeichert ist.
        feature_names_file (str): Datei mit den Feature-Namen für die Interpretation.
        threshold (float): Schwellenwert, um kritische Mitarbeiter zu identifizieren.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Datenframe mit kritischen Mitarbeitern und den Top 15 Mitarbeitern.
    """
    try:
        # Daten laden und bereinigen
        df = load_dataset(file_path, save_filtered_path)
        df_cleaned = clean_dataset(df)

        # Daten vorverarbeiten
        df_processed, X_transformed, X_resampled, y_resampled, preprocessor = preprocess_data(df_cleaned)

        # Modell-Pfad erstellen und Modell laden
        model_file = os.path.join(models_dir, model_file_name)
        try:
            model = joblib.load(model_file)  # Modell laden
            print("Das Modell wurde erfolgreich geladen.")
        except FileNotFoundError:
            print(f"Fehler: Die Modell-Datei '{model_file}' wurde nicht gefunden.")
            return None, None
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            return None, None

        # Kritische Mitarbeiter identifizieren
        critical_employees, top_15_mitarbeiter = get_critical_employees(
            model, X_transformed, df_processed, feature_names_file=feature_names_file, threshold=threshold
        )

        # Ergebnisse ausgeben
        print(
            f"Top 15 Mitarbeiter:\n{top_15_mitarbeiter[['Mitarbeiter_ID', 'Name', 'Fluktuationswahrscheinlichkeit']]}")

        return critical_employees, top_15_mitarbeiter
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return None, None