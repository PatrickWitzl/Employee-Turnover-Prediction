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
from sklearn.decomposition import PCA
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
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as k
from tensorflow.keras import Sequential
from tensorflow.keras import layers

def load_data(file_path):
    """
    Lädt die Daten aus einer CSV-Datei und verarbeitet sie entsprechend den Anforderungen.
    Erstellt eine neue Spalte 'Fluktuation', die 1 für 'Ausgeschieden' und 0 für andere Werte enthält,
    und entfernt die Spalte 'Status'.

    Args:
        file_path (str): Der Dateipfad zur CSV-Datei.

    Returns:
        pd.DataFrame: Der vorverarbeitete DataFrame.
    """

    # Daten laden
    df = pd.read_csv(file_path, low_memory=False)

    return df

def preprocess_data(df, output_file="X_transformed.csv"):
    """
    Vorverarbeitung der Daten:
    - Zielvariable "Fluktuation" ableiten
    - Feature-Auswahl basierend auf Analyse
    - One-Hot-Encoding für kategorische Features
    - Klassenungleichgewicht mit SMOTE behandeln
    - Anpassung von "Fehlzeiten_Krankheitstage" auf Basis von "Abwesenheitsgrund"

    Args:
        df (pd.DataFrame): Der Original-Dataframe.

    Returns:
        pd.DataFrame: Original transformierte Features (X_transformed).
        pd.DataFrame: Resampled Features (X_resampled).
        np.ndarray: Zielvariable (y_resampled) nach SMOTE.
        ColumnTransformer: Preprocessor für spätere Transformationen.
    """
    # Mitarbeiter mit Status "Ruhestand" entfernen
    df = df.copy()  # Vermeide Original zu modifizieren
    df = df[df['Status'] != 'Ruhestand']
    print(f"Shape nach Entfernen von 'Ruhestand': {df.shape}")

    # Bereinige "Fehlzeiten_Krankheitstage", wenn der "Abwesenheitsgrund" nicht "Krankheit" ist
    df['Fehlzeiten_Krankheitstage'] = df.apply(
        lambda row: row['Fehlzeiten_Krankheitstage'] if row['Abwesenheitsgrund'] == "Krankheit" else 0,
        axis=1
    )
    # Zielvariable "Fluktuation" ableiten
    df['Fluktuation'] = df['Status'].apply(lambda x: 1 if x == "Ausgeschieden" else 0)

    # Features kombinieren (manuelle Auswahl nach Analyse)
    selected_features = [
        'Jahr','Monat', 'Alter', 'Überstunden', 'Fehlzeiten_Krankheitstage',
        'Gehalt', 'Zufriedenheit', 'Fortbildungskosten',
        'Position', 'Geschlecht', 'Standort',
        'Arbeitszeitmodell', 'Verheiratet',
        'Kinder', 'Job Role Progression', 'Job Level', 'Tenure'
    ]
    X = df[selected_features]
    y = df['Fluktuation']

    print(f"Shape von X: {X.shape}")
    print(f"Shape von y: {y.shape}")

    # Kategorische und numerische Spalten erkennen
    categorical_columns = X.select_dtypes(include='object').columns
    numerical_columns = X.select_dtypes(exclude='object').columns

    # Preprocessor definieren
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_columns),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
        ]
    )

    # Features transformieren
    X_transformed = preprocessor.fit_transform(X)

    # Neue Spaltennamen nach Transformation
    transformed_columns = list(numerical_columns) + \
                          list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns))

    # In DataFrame zurückschreiben
    X_transformed = pd.DataFrame(X_transformed, columns=transformed_columns, index=X.index)

    # Synchronisation von df mit X_transformed sicherstellen
    # Prüfe die Indizes, um sicherzustellen, dass nur relevante Zeilen enthalten sind
    if not X_transformed.index.equals(df.index):
        print("Index nicht identisch! Sync wird vorgenommen.")
        df = df.loc[X_transformed.index]  # Synchronisierung basierend auf dem Index von X_transformed

    # SMOTE für Klassenungleichgewicht
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_transformed, y)

    # Zurück als DataFrame für interpretierbare Spalten
    X_resampled = pd.DataFrame(X_resampled, columns=X_transformed.columns)

    # Transformierte Daten X_transformed als CSV speichern
    X_transformed.to_csv(output_file, index=False)  # Speichert die Datei
    print(f"Transformierte Daten wurden gespeichert unter '{output_file}'")

    return df, X_transformed, X_resampled, y_resampled, preprocessor

def split_and_scale(X_resampled, y_resampled, test_size=0.2, scaler_file="Models/scaler.pkl"):
    """
    Teilt die Daten in Trainings- und Testsets und skaliert sie. Speichert den Scaler für spätere Skalierungen.

    Args:
        X_resampled (np.ndarray): Features nach Resampling.
        y_resampled (np.ndarray): Zielvariable nach Resampling.
        test_size (float): Anteil der Testdaten.
        scaler_file (str): Dateiname für das Speichern des Scalers.

    Returns:
        pd.DataFrame: Skalierte Trainingsdaten (X_train_scaled).
        pd.DataFrame: Skalierte Testdaten (X_test_scaled).
        np.ndarray: Trainings-Zielvariablen.
        np.ndarray: Test-Zielvariablen.
    """

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=42, stratify=y_resampled
    )

    # Daten skalieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scaler speichern
    joblib.dump(scaler, scaler_file)  # Speichert den Scaler in einer Datei
    print(f"Scaler wurde gespeichert unter '{scaler_file}'")

    # Daten zurück in DataFrame konvertieren
    X_train = pd.DataFrame(X_train_scaled, columns=X_resampled.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_resampled.columns)

    return X_train, X_test, y_train, y_test, scaler

def reduce_dimensions(X_train, X_test, X_resampled, n_components=0.95):
    """
    Führt PCA durch, während Spaltennamen erhalten bleiben, falls DataFrames genutzt werden.

    Args:
        X_train (pd.DataFrame oder np.ndarray): Trainingsdatensatz.
        X_test (pd.DataFrame oder np.ndarray): Testdatensatz.
        X_full (pd.DataFrame oder np.ndarray): Gesamtdatensatz.
        n_components (float oder int): Anzahl der Hauptkomponenten oder Anteil der erklärten Varianz.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PCA]: PCA-Transformation als DataFrames und das PCA-Modell.
    """
    X_full = X_resampled
    try:
        # Prüfen, ob Eingaben DataFrames sind
        is_dataframe = isinstance(X_train, pd.DataFrame)

        if is_dataframe:
            # Spaltennamen und Index speichern
            if not (X_train.columns.equals(X_test.columns) and X_train.columns.equals(X_full.columns)):
                raise ValueError("Die Spalten der Eingaben (X_train, X_test, X_full) sind inkonsistent.")

            X_train_columns = X_train.columns
            index_train, index_test, index_full = X_train.index, X_test.index, X_full.index
        else:
            # Sicherstellen, dass alle Eingaben NumPy-Arrays sind
            if not isinstance(X_train, np.ndarray) or not isinstance(X_test, np.ndarray) or not isinstance(X_full,
                                                                                                           np.ndarray):
                raise ValueError("Alle Eingabedaten müssen entweder 'pd.DataFrame' oder 'np.ndarray' sein.")
            if X_train.shape[1] != X_test.shape[1] or X_train.shape[1] != X_full.shape[1]:
                raise ValueError("Alle Eingabedatensätze müssen die gleiche Anzahl von Spalten haben.")

        # PCA initialisieren und Trainingsdatensatz fitten
        pca = PCA(n_components=n_components, svd_solver='full')
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_full_pca = pca.transform(X_full)

        # Debugging der erklärten Varianz (nur bei Verhältnis)
        if isinstance(n_components, float):
            variance_explained = np.sum(pca.explained_variance_ratio_) * 100
            print(f"PCA erklärt {variance_explained:.2f}% der Varianz durch {pca.n_components_} Komponenten.")

        print(f"PCA erfolgreich abgeschlossen.")
        print(f"- Originaldimensionen vor PCA: {X_train.shape[1]}")
        print(f"- Reduzierte Dimensionen nach PCA: {X_train_pca.shape[1]}")

        # Falls DataFrame, Daten in DataFrame zurückgeben
        if is_dataframe:
            pca_columns = [f"PCA_{i + 1}" for i in range(X_train_pca.shape[1])]
            X_train_pca = pd.DataFrame(X_train_pca, columns=pca_columns, index=index_train)
            X_test_pca = pd.DataFrame(X_test_pca, columns=pca_columns, index=index_test)
            X_full_pca = pd.DataFrame(X_full_pca, columns=pca_columns, index=index_full)

        return X_train_pca, X_test_pca, X_full_pca, pca

    except Exception as e:
        print(f"Fehler während der PCA-Berechnung: {e}")
        raise

def model_selection():
    """
    Ermöglicht dem Benutzer, Modelle für die Analyse auszuwählen. Dabei können entweder
    spezifische Modelle oder alle Modelle ausgewählt werden. Wenn der Benutzer innerhalb
    von 10 Sekunden keine Auswahl trifft, werden automatisch alle Modelle ausgewählt.

    Returns:
        include_models (list): Liste der auszuwählenden Modelle.
    """
    # Mapping für Modelle
    model_mapping = {
        1: "Logistic Regression",
        2: "Random Forest",
        3: "XGBoost",
        4: "LightGBM",
        5: "Neural Network"
    }

    print("\nBitte wählen Sie Modelle für die Analyse aus:")
    print("1: Logistic Regression")
    print("2: Random Forest")
    print("3: XGBoost")
    print("4: LightGBM")
    print("5: Neural Network")
    print("6: Alle Modelle")

    def input_with_timeout(prompt, timeout=0.10):
        """
        Funktion, die Benutzereingaben mit Timeout verarbeitet. Wenn der Benutzer nicht innerhalb
        der angegebenen Zeit reagiert, wird None zurückgegeben.
        """
        print(prompt, end='', flush=True)  # Eingabeaufforderung anzeigen
        inputs, _, _ = select.select([sys.stdin], [], [], timeout)
        if inputs:
            return sys.stdin.readline().strip()  # Eingabe lesen und zurückgeben
        else:
            print("\nTimeout abgelaufen. Keine Eingabe erkannt.")
            return None

    # Warten auf Benutzereingabe mit Timeout
    selected_input = input_with_timeout(
        "Geben Sie die Nummern der gewünschten Modelle durch Kommas getrennt ein (z. B.: 1,3,5 oder 6 für alle Modelle):\n",
        timeout=0.10  # Timeout in Sekunden
    )

    # Wenn keine Eingabe erfolgt ist, werden alle Modelle gewählt
    if selected_input is None:
        print("\nKeine Auswahl getroffen. Es werden automatisch alle Modelle ausgewählt.")
        return list(model_mapping.values())  # Gibt alle Modelnamen zurück

    # Verarbeiten der Benutzereingabe
    try:
        selected_numbers = [int(n) for n in selected_input.split(",")]  # Benutzereingaben in Liste umwandeln
    except ValueError:
        raise ValueError("Ungültige Eingabe. Bitte geben Sie gültige Modellnummern ein (z. B.: 1,2 oder 6).")

    if 6 in selected_numbers:  # 'Alle Modelle' wurde ausgewählt
        include_models = list(model_mapping.values())  # Alle Modelle zurückgeben
    else:
        # Nur die ausgewählten Modelle zusammenstellen
        include_models = [model_mapping[n] for n in selected_numbers if n in model_mapping]

    if not include_models:  # Sicherstellen, dass mindestens ein Modell ausgewählt wurde
        raise ValueError("Keine gültigen Modelle ausgewählt. Bitte wählen Sie mindestens ein Modell.")

    return include_models

def get_user_choice(models_dir, timeout=0.10):
    """
    Fragt den Benutzer, ob gespeicherte Modelle verwendet oder neue Modelle trainiert werden sollen.

    Args:
        models_dir (str): Verzeichnis, in dem die Modelle gespeichert werden.
        timeout (int): Zeit in Sekunden, die der Benutzer hat, um eine Eingabe zu tätigen.

    Returns:
        bool: True, wenn gespeicherte Modelle verwendet werden sollen und diese existieren, sonst False.
    """
    def input_with_timeout(prompt, timeout=0.10):
        print(prompt, end='', flush=True)
        inputs, _, _ = select.select([sys.stdin], [], [], timeout)
        if inputs:
            return sys.stdin.readline().strip().lower()  # Eingabe lesen
        else:
            print("\nTimeout abgelaufen. Standard: 'y' (gespeicherte Modelle verwenden).")
            return "y"  # Standardoption ist jetzt "y"

    # Benutzer auswählen lassen
    user_choice = input_with_timeout(
        "Verwenden Sie gespeicherte Modelle? [y/n] (Standard: y): ", timeout
    )

    # Prüfung, ob Modelle existieren
    def models_exist(models_dir):
        # Sucht nach vorhandenen Modelldateien im `model_dir`
        for file in os.listdir(models_dir):
            if file.endswith(".pkl") or file.endswith(".json") or file.endswith(".keras"):
                return True
        return False

    # Wenn Benutzeroption 'y' angibt, prüfen, ob Modelle vorhanden sind
    if user_choice == "y":
        if models_exist(models_dir):
            print("Gespeicherte Modelle werden verwendet.")
            return True
        else:
            print("Keine gespeicherten Modelle gefunden. Neues Training wird gestartet.")
            return False
    elif user_choice == "n":
        print("Neue Modelle werden trainiert.")
        return False
    else:
        print("Ungültige Eingabe. Standard: 'y' (gespeicherte Modelle verwenden).")
        if models_exist(models_dir):
            print("Gespeicherte Modelle werden verwendet.")
            return True
        else:
            print("Keine gespeicherten Modelle gefunden. Neues Training wird gestartet.")
            return False

def train_models(X_train, X_test, y_train, y_test, include_models, use_saved_models, models_dir):
    """
    Trainiert dynamisch Modelle basierend auf der Auswahl `include_models`.

    Args:
        X_train, X_test, y_train, y_test: Trainings- und Test-Daten.
        include_models (list): Liste der zu trainierenden Modelle.

    Returns:
        list: Liste von Ergebnissen (Scores und Modelle für jedes Modell).
    """
    trained_models_results = []

    if "Logistic Regression" in include_models:
        try:
            result = train_logistic_regression(X_train, X_test, y_train, y_test,use_saved_models, models_dir)
            if "model_name" in result:  # Sicherstellen, dass das Ergebnis valide ist
                trained_models_results.append(result)
            else:
                print(f"Warnung: Logistic Regression hat kein valides Ergebnis zurückgegeben.")
        except Exception as e:
            print(f"Fehler beim Training der Logistic Regression: {e}")

    if "Random Forest" in include_models:
        try:
            result = train_random_forest(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:
                trained_models_results.append(result)
            else:
                print(f"Warnung: Random Forest hat kein valides Ergebnis zurückgegeben.")
        except Exception as e:
            print(f"Fehler beim Training des Random Forest: {e}")

    if "XGBoost" in include_models:
        try:
            result = train_xgboost(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:
                trained_models_results.append(result)
            else:
                print(f"Warnung: XGBoost hat kein valides Ergebnis zurückgegeben.")
        except Exception as e:
            print(f"Fehler beim Training von XGBoost: {e}")

    if "LightGBM" in include_models:
        try:
            result = train_lightgbm(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:
                trained_models_results.append(result)
            else:
                print(f"Warnung: LightGBM hat kein valides Ergebnis zurückgegeben.")
        except Exception as e:
            print(f"Fehler beim Training des LightGBM: {e}")

    if "Neural Network" in include_models:
        try:
            result = train_neural_network(X_train, X_test, y_train, y_test, use_saved_models, models_dir)
            if "model_name" in result:
                trained_models_results.append(result)
            else:
                print(f"Warnung: Neural Network hat kein valides Ergebnis zurückgegeben.")
        except Exception as e:
            print(f"Fehler beim Training des Neural Networks: {e}")

    return trained_models_results

def train_logistic_regression(X_train, X_test, y_train, y_test, use_saved_models, models_dir, pca=None):
    """
    Trainiert oder lädt Logistic Regression mit optionaler PCA und behandelt Dimension-Mismatches.
    """
    model_file = os.path.join(models_dir, 'logistic_regression_with_pca.pkl')

    try:
        # Verzeichnis für Modelle erstellen, falls es nicht existiert
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Prüfen, ob ein gespeichertes Modell geladen werden soll
        model = None
        if use_saved_models and os.path.exists(model_file):
            print("[INFO] Lade gespeichertes Logistic Regression Modell...")
            model, loaded_pca = joblib.load(model_file)

            # Prüfen, ob PCA geladen wurde und sollte verwendet werden
            if pca is None and loaded_pca is not None:
                print("[INFO] Verwende gespeichertes PCA-Objekt.")
                pca = loaded_pca

            # Prüfen, ob die Trainingsdaten zu den Modell-Features passen
            if X_train.shape[1] != model.n_features_in_:
                print(
                    f"[WARNUNG] Die Anzahl der Features im Modell ({model.n_features_in_}) stimmt nicht mit den Eingabedaten überein ({X_train.shape[1]}).")
                print("[INFO] Versuche automatische Anpassung der Features...")

                # Fehlende Spalten ergänzen
                missing_columns = set(model.feature_names_in_) - set(X_train.columns)
                for col in missing_columns:
                    X_train[col] = 0
                    X_test[col] = 0

                # Zusätzliche Spalten entfernen
                X_train = X_train[model.feature_names_in_]
                X_test = X_test[model.feature_names_in_]

        # Anwenden von PCA, falls vorhanden
        if pca is not None:
            print("[INFO] PCA auf Eingabedaten anwenden...")
            if hasattr(pca, "n_components_"):
                if pca.n_components_ != X_train.shape[1] and model is not None:
                    raise ValueError(
                        f"[FEHLER] PCA erzeugt {pca.n_components_} Features, aber das Modell erwartet {model.n_features_in_} Features.")
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

        # Falls kein Modell geladen wurde, Training starten
        if model is None:
            print("[INFO] Kein gespeichertes Modell gefunden. Starte neues Training...")
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

            # Training des Modells
            print("[INFO] Starte GridSearchCV-Suche für Logistic Regression...")
            logreg_pipeline.fit(X_train, y_train)
            model = logreg_pipeline.best_estimator_
            print("[INFO] Training erfolgreich abgeschlossen.")

            # Speichern der Spaltennamen, um Dimension-Mismatches in Zukunft zu vermeiden
            if isinstance(X_train, pd.DataFrame):
                model.feature_names_in_ = X_train.columns

            # Modell und PCA-Objekt speichern
            joblib.dump((model, pca), model_file)
            print(f"[INFO] Modell gespeichert: {model_file}")

        # Vorhersagen und Performancemetriken berechnen
        print("[INFO] Berechne Vorhersagen und Metriken...")
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        threshold = 0.5  # Standardmäßiger Schwellenwert
        y_train_pred = (y_train_proba > threshold).astype(int)
        y_test_pred = (y_test_proba > threshold).astype(int)

        # Performance berechnen
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

        # Metriken auf der Konsole ausgeben
        print(f"\nTrainings-Genauigkeit: {result['train_accuracy']:.4f}")
        print(f"Test-Genauigkeit: {result['test_accuracy']:.4f}")
        print(f"ROC-AUC: {result['roc_auc_score']:.4f}")
        print(f"F1-Score: {result['f1_score']:.4f}")
        print(f"Präzision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print("=" * 24)

        return result

    except FileNotFoundError as fnfe:
        print(f"[FEHLER] Speicherpfad oder Datei nicht gefunden: {fnfe}")
        raise

    except ValueError as ve:
        print(f"[FEHLER] Ungültige Eingabedaten oder Modelle: {ve}")
        raise

    except Exception as e:
        print(f"[UNERWARTETER FEHLER] {e}")
        raise

def train_random_forest(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trainiert einen Random Forest Classifier mit anpassbarem Threshold.

    Args:
        X_train, X_test, y_train, y_test: Trainings- und Test-Daten.
        use_saved_models: Flag, ob gespeicherte Modelle verwendet werden sollen.
        models_dir: Pfad zum Speichern der Modelle.

    Returns:
        dict: Ergebnisdaten, einschließlich trainiertem Modell, Scores und Predictions.
    """
    # Sicherstellen, dass X_train und X_test DataFrames mit identischen Spaltennamen sind
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("Die Spalten von X_train und X_test stimmen nicht überein.")
    else:
        raise TypeError("X_train und X_test müssen DataFrames sein, um Konsistenz sicherzustellen.")

    model_file = os.path.join(models_dir, 'random_forest_model.pkl')

    # Prüfen, ob ein gespeichertes Modell vorhanden ist
    if use_saved_models and os.path.exists(model_file):
        print(f"\nGespeichertes Random Forest-Modell gefunden. Laden...")
        model = joblib.load(model_file)
    else:
        print(f"\nTraining für Random Forest startet...")
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        # Modell speichern
        joblib.dump(model, model_file)
        print(f"Random Forest-Modell gespeichert als '{model_file}'")

    # Wahrscheinlichkeiten berechnen
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Schwellenwert (Threshold) für Klassifikation anpassen
    threshold = 0.5  # Sie können diesen Wert anpassen
    y_train_pred = (y_train_proba > threshold).astype(int)
    y_test_pred = (y_test_proba > threshold).astype(int)

    # Zusätzliche Bewertungsmethoden berechnen
    results = {
        "model_name": "Random Forest",
        "trained_model": model,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "roc_auc_score": roc_auc_score(y_test, y_test_proba),
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "y_test_proba": y_test_proba  # Falls Wahrscheinlichkeiten gespeichert werden sollen
    }

    # Bewertungsergebnisse ausgeben
    print(f"\nTrainings-Genauigkeit: {results['train_accuracy']:.4f}")
    print(f"Test-Genauigkeit: {results['test_accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc_score']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Präzision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print("=" * 24)

    return results

def train_xgboost(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trainiert einen XGBoost Classifier.

    Args:
        X_train, X_test, y_train, y_test: Trainings- und Test-Daten.
        use_saved_models (bool): Ob ein gespeichertes Modell verwendet werden soll.
        models_dir (str): Verzeichnis, in dem Modelle gespeichert werden sollen.

    Returns:
        dict: Ergebnisdaten, einschließlich trainiertem Modell, Scores und Predictions.
    """
    # Datenvalidierung
    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError("Trainings- oder Testdaten sind nicht korrekt geladen. Bitte überprüfen!")
    if len(X_train) == 0 or len(X_test) == 0 or len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Trainings- oder Testdaten sind leer. Bitte überprüfen!")

    model_file = os.path.join(models_dir, 'xgboost_model.pkl')
    model = None

    # Prüfen, ob ein gespeichertes Modell vorhanden ist
    if use_saved_models and os.path.exists(model_file):
        try:
            print(f"\nGespeichertes XGBoost-Modell gefunden. Laden...")
            model = xgb.XGBClassifier()
            model.load_model(model_file)
            # Sicherstellen, dass das geladene Modell verwendbar ist
            if not hasattr(model, 'classes_'):
                print("Warnung: Das geladene Modell scheint unvollständig zu sein. Neu-Training notwendig.")
                model = None
        except Exception as e:
            print(f"Fehler beim Laden des gespeicherten Modells: {e}")
            model = None

    # Trainieren, falls kein gültiges Modell geladen wurde
    if model is None:
        print(f"\nTraining für XGBoost startet...")
        model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            raise ValueError(f"Fehler beim Trainieren des Modells: {e}")

        # Modell speichern
        try:
            joblib.dump(model, model_file)  # Speichern im festgelegten Verzeichnis
            print(f"XGBoost-Modell erfolgreich in '{model_file}' gespeichert.")
        except Exception as e:
            print(f"Fehler beim Speichern des Modells: {e}")

    # Vorhersagen (Probabilistisch und Klassifikation)
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]  # Wahrscheinlichkeiten für Klasse 1
        y_test_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        raise ValueError(f"Fehler bei probabilistischen Vorhersagen: {e}")

    # Schwellenwert (Threshold) für Klassifikation anpassen
    threshold = 0.5  # Standardwert, kann angepasst werden
    y_train_pred = (y_train_proba > threshold).astype(int)
    y_test_pred = (y_test_proba > threshold).astype(int)

    # ROC-AUC prüfen (Nur falls `predict_proba` verfügbar ist)
    roc_auc = roc_auc_score(y_test, y_test_proba) if hasattr(model, "predict_proba") else None

    # Evaluations-Metriken
    results = {
        "model_name": "XGBoost",
        "trained_model": model,
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "roc_auc_score": roc_auc,
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "y_test_proba": y_test_proba  # Probabilistische Vorhersagen (optional für spätere Auswertungen)
    }

    # Bewertungsergebnisse ausgeben
    print(f"\nTrainings-Genauigkeit: {results['train_accuracy']:.4f}")
    print(f"Test-Genauigkeit: {results['test_accuracy']:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {results['roc_auc_score']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Präzision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print("=" * 24)

    return results

def train_lightgbm(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trainiert einen LightGBM Classifier, speichert das Modell und die Feature-Namen lokal.
    Falls ein gespeichertes Modell vorhanden ist, wird es geladen.

    Args:
        X_train, X_test, y_train, y_test: Trainings- und Test-Daten.
        use_saved_models: Flag, ob gespeicherte Modelle verwendet werden sollen.
        models_dir: Speicherpfad für das Modell.

    Returns:
        dict: Ergebnisdaten, inklusive trainiertem Modell, Scores und Predictions.
    """

    # Sicherstellen, dass X_train und X_test Pandas-DataFrames mit gleichen Spaltennamen sind
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns)  # Konsistente Feature-Namen sicherstellen

    # Modelle und Feature-Namen speichern in diesen Dateien
    model_file = os.path.join(models_dir, 'lightgbm_model.pkl')
    feature_names_file = os.path.join(models_dir, 'lightgbm_feature_names.pkl')
    model = None

    # Prüfe, ob das Modell schon existiert und geladen werden kann
    if use_saved_models and os.path.exists(model_file):
        try:
            print("\nGespeichertes LightGBM-Modell gefunden. Laden...")
            model = joblib.load(model_file)

            # Lade auch Feature-Namen
            if os.path.exists(feature_names_file):
                with open(feature_names_file, 'rb') as f:
                    saved_feature_names = joblib.load(f)

                # Sicherstellen, dass die gelesenen Feature-Namen zu den jetzigen passen
                if list(X_train.columns) != saved_feature_names:
                    raise ValueError("Die Feature-Namen aus den gespeicherten Daten passen nicht zu X_train!")
        except Exception as e:
            print(f"Fehler beim Laden des gespeicherten Modells oder der Daten: {e}")
            model = None

    # Modell trainieren, wenn keines geladen werden konnte
    if model is None:
        print("\nKein Modell gefunden. Starte Training für LightGBM...")
        model = LGBMClassifier(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)

        # Modell und Feature-Namen speichern
        try:
            joblib.dump(model, model_file)
            print(f"LightGBM-Modell gespeichert als '{model_file}'")

            with open(feature_names_file, 'wb') as f:
                joblib.dump(list(X_train.columns), f)
                print(f"Feature-Namen gespeichert als '{feature_names_file}'")
        except Exception as e:
            print(f"Fehler beim Speichern des Modells oder der Feature-Namen: {e}")

    # Vorhersagen und Wahrscheinlichkeiten berechnen
    y_train_proba = model.predict_proba(X_train)[:, 1]  # Klassenwahrscheinlichkeiten
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Klassenschwellenwert
    threshold = 0.5
    y_train_pred = (y_train_proba > threshold).astype(int)
    y_test_pred = (y_test_proba > threshold).astype(int)

    # Zusätzliche Metriken berechnen
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

    # Ergebnisse ausgeben
    print(f"\nTrainings-Genauigkeit: {results['train_accuracy']:.4f}")
    print(f"Test-Genauigkeit: {results['test_accuracy']:.4f}")
    print(f"ROC-AUC: {results['roc_auc_score']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Präzision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print("=" * 24)

    return results

def train_neural_network(X_train, X_test, y_train, y_test, use_saved_models, models_dir):
    """
    Trainiert ein neuronales Netzwerk
    oder lädt ein gespeichertes Modell, falls vorhanden.
    """
    model_file = os.path.join(models_dir, 'nn_model.keras')

    # Prüfen, ob ein gespeichertes Modell vorhanden ist
    if use_saved_models and os.path.exists(model_file):
        print(f"\nGespeichertes Keras-Modell gefunden. Laden...")
        model = load_model(model_file)

        # Prüfen, ob Input-Dimensionen übereinstimmen
        input_shape = model.input_shape[1]
        if input_shape != X_train.shape[1]:
            raise ValueError(f"Das gespeicherte Modell erwartet Eingabedimension {input_shape}, "
                             f"aber die aktuellen Daten haben {X_train.shape[1]} Features.")
    else:
        print(f"\nTraining für ein neuronales Netzwerk startet...")
        feature_count = X_train.shape[1]

        # Schritt 3: Modell erstellen
        model = Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # Kompilierung des Modells
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

        try:
            # Modell trainieren
            model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=0)
        except Exception as e:
            print(f"Fehler beim Training des neuronalen Netzwerks: {e}")
            return None

        # Modell speichern
        model.save(model_file)
        print(f"Keras-Modell gespeichert als '{model_file}'")

    # Wahrscheinlichkeiten berechnen
    y_train_proba = model.predict(X_train).ravel()
    y_test_proba = model.predict(X_test).ravel()

    # Wahrscheinlichkeitsverteilung prüfen
    plt.hist(y_test_proba, bins=30, alpha=0.8, color='blue')
    plt.title("Histogramm der vorhergesagten Wahrscheinlichkeiten")
    plt.xlabel("Wahrscheinlichkeit")
    plt.ylabel("Häufigkeit")
    plt.show()

    # Dynamische Schwellenwertanpassung (z. B. anhand Precision-Recall)
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
    # Wähle Schwellenwert, der Precision > 0.6 garantiert
    threshold = thresholds[np.argmax(precision > 0.6)] if np.any(precision > 0.6) else 0.5
    print(f"Automatisch angepasster Schwellenwert: {threshold:.2f}")

    y_train_pred = (y_train_proba > threshold).astype(int)
    y_test_pred = (y_test_proba > threshold).astype(int)

    # Bewertung des Modells
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

    # Ausgabe überprüfen: Prüfung auf leere Ergebnisse
    critical_count = np.sum(y_test_pred)
    if critical_count == 0:
        print("\nWARNUNG: Keine kritischen Mitarbeiter gefunden. Prüfen Sie Daten und Modell!")
    else:
        print(f"Kritische Mitarbeiter identifiziert: {critical_count}")

    # Bewertungsergebnisse ausgeben
    print(f"\nTrainings-Genauigkeit: {result['train_accuracy']:.4f}")
    print(f"Test-Genauigkeit: {result['test_accuracy']:.4f}")
    print(f"ROC-AUC: {result['roc_auc_score']:.4f}")
    print(f"F1-Score: {result['f1_score']:.4f}")
    print(f"Präzision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print("=" * 24)

    return result

def evaluate_models(models, X_test, y_test, plot_dir="plots"):
    """
    Bewertet ML-Modelle und erstellt Verwirrungsmatrix-Subplots.
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

    # Modelle evaluieren
    for name, model in models.items():
        if model is None:
            print(f"Warnung: Modell '{name}' ist None und wird übersprungen.")
            continue
        try:
            # Vorhersagen erstellen
            if hasattr(model, "predict_proba") and callable(getattr(model, "predict_proba")):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # Für Modelle ohne `predict_proba`:
                y_proba = model.predict(X_test)
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 0]

            # Binäre Klassifikation basierend auf 0.5-Schwellenwert
            y_pred = (y_proba >= 0.5).astype(int)

            # Metriken berechnen
            roc_auc = roc_auc_score(y_test, y_proba)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            log_loss_val = log_loss(y_test, y_proba)

            # Verwirrungsmatrix berechnen -> TN, FP, FN, TP
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            # Ergebnisse speichern
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

            print(f"Modell '{name}' bewertet:")
            print(f"  ROC-AUC: {roc_auc:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"  Accuracy: {accuracy:.4f}, MCC: {mcc:.4f}, Log-Loss: {log_loss_val:.4f}")
            print(f"  Confusion Matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")

        except Exception as e:
            print(f"Fehler beim Evaluieren des Modells '{name}': {e}")

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
            ax.imshow(matrix, interpolation="nearest", cmap="Blues")  # Direkte Farbschema-Verwendung

            ax.set_title(f"Confusion Matrix - {model_names[i]}")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Negative", "Positive"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Negative", "Positive"])

            # Hinzufügen von Text innerhalb jeder Matrix-Zelle
            for j in range(2):
                for k in range(2):
                    ax.text(k, j, format(matrix[j, k], "d"),
                            horizontalalignment="center",
                            color="white" if matrix[j, k] > matrix.max() / 2 else "black")  # Dynamische Farbe

        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "combined_confusion_matrices.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()


    except Exception as e:
        print(f"Fehler bei der Erstellung des Verwirrungsmatrix-Plots: {e}")

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
    Erstellt ein horizontales Balkendiagramm, um Modellleistung zu vergleichen.

    Parameter:
    -----------
    model_names : list
        Liste der Modellnamen.
    train_accuracies : list
        Liste der Trainingsgenauigkeiten (optional einblendbar).
    test_accuracies : list
        Liste der Testgenauigkeiten.
    roc_auc_scores : list
        Liste der ROC-AUC-Werte.
    training_times : list
        Liste der Trainingszeiten in Sekunden (werden als Annotation angezeigt).
    f1_scores : list, optional
        Liste der F1-Scores, falls vorhanden.
    precisions : list, optional
        Liste der Präzisionswerte, falls vorhanden.
    recalls : list, optional
        Liste der Recall-Werte, falls vorhanden.
    plot_dir : str
        Verzeichnis zum Speichern des Diagramms.
    sort_by : str
        Metrik, nach der die Modelle sortiert werden (`"Train Accuracy"`, `"Test Accuracy"`, `"ROC AUC"`,
        `"Training Time (s)"`, `"F1 Score"`, `"Precision"`, `"Recall"`).
    include_train : bool
        Ob `Train Accuracy` dargestellt werden soll (optional).
    bar_colors : tuple
        Farben der Balken für Train, Test, ROC AUC, F1 Score, Precision und Recall.
    xlabel : str
        Beschriftung der x-Achse (dynamische Sprache möglich).
    legend_labels : tuple
        Legendenbeschriftung für Train, Test, ROC AUC, F1 Score, Precision und Recall.
    """
    # Prüfen auf Verzeichnis
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Daten vorbereiten (neue Metriken optional hinzufügen)
    data = pd.DataFrame({
        "Model": model_names,
        "Train Accuracy": train_accuracies,
        "Test Accuracy": test_accuracies,
        "ROC AUC": roc_auc_scores,
    })

    # Optional: Zusätzliche Spalten hinzufügen, wenn Daten verfügbar
    if f1_scores:
        data["F1 Score"] = f1_scores
    if precisions:
        data["Precision"] = precisions
    if recalls:
        data["Recall"] = recalls

    # Dynamische Sortierung der Modelle
    if sort_by in data.columns:
        data = data.sort_values(by=sort_by, ascending=False)  # Absteigend sortieren
    else:
        raise ValueError(f"Invalid sort_by parameter: {sort_by}. Must be one of {', '.join(data.columns)}")

    # Diagrammerstellung
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.10  # Breite der Balken für verschiedene Metriken
    y_pos = np.arange(len(data))  # Y-Positionen für die Balken

    # Balkendiagramme nach Metrik
    if include_train:
        ax.barh(y_pos - 2 * bar_width, data["Train Accuracy"], bar_width, label=legend_labels[0], color=bar_colors[0])
    ax.barh(y_pos - bar_width, data["Test Accuracy"], bar_width, label=legend_labels[1], color=bar_colors[1])
    ax.barh(y_pos, data["ROC AUC"], bar_width, label=legend_labels[2], color=bar_colors[2])

    # Zusätzliche Metriken optional einfügen
    if "F1 Score" in data.columns:
        ax.barh(y_pos + bar_width, data["F1 Score"], bar_width, label=legend_labels[3], color=bar_colors[3])
    if "Precision" in data.columns:
        ax.barh(y_pos + 2 * bar_width, data["Precision"], bar_width, label=legend_labels[4], color=bar_colors[4])
    if "Recall" in data.columns:
        ax.barh(y_pos + 3 * bar_width, data["Recall"], bar_width, label=legend_labels[5], color=bar_colors[5])


    # Achsenbeschriftung, Titel und Legende
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data["Model"])
    ax.set_title("Modellvergleich")
    ax.set_xlabel(xlabel)
    ax.legend()
    plt.tight_layout()

    # Diagramm speichern
    output_path = os.path.join(plot_dir, "model_performance_comparison.png")
    plt.savefig(output_path)
    plt.show()

    print(f"Diagramm gespeichert unter: {output_path}")

def plot_combined_roc_curves(models, X_test, y_test, plot_dir, nn_model=None, y_proba_nn=None ):
    """
    Plot für ROC-Kurven aller Modelle erstellen, einschließlich des neuronalen Netzwerks.
    """
    try:
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(16, 10))
        valid_models = False  # Prüfen, ob zumindest ein Modell erfolgreich

        # Plot der traditionellen Modelle
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                # Wahrscheinlichkeiten abrufen
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"ROC-{name} (AUC={roc_auc:.2f})")
                valid_models = True

        # Plot für das neuronale Netzwerk
        if nn_model and y_proba_nn is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba_nn)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"ROC-NeuralNet (AUC={roc_auc:.2f})", linestyle="--", color="purple")
            valid_models = True

        # Zufallsklassifikationsgrenze
        if valid_models:
            plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")
            plt.legend(loc="lower right")
            plt.title("ROC-Kurven")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, "all_roc_curves.png"))
            plt.show()
        else:
            print("Keine validen Modelle für ROC-Kurven gefunden.")

    except Exception as e:
        print(f"Fehler bei der Erstellung des ROC-Plots: {e}")

def test_under_over_fit(model_names, train_accuracies, test_accuracies, overfit_threshold=0.1, underfit_threshold=0.6):
    """
    Identifiziert Overfitting und Underfitting für gegebene Modelle basierend auf Genauigkeitswerten.

    Args:
        model_names (list): Liste der Modellnamen.
        train_accuracies (list): Trainingsgenauigkeiten der Modelle.
        test_accuracies (list): Testgenauigkeiten der Modelle.
        overfit_threshold (float): Schwellenwert für Overfitting (Train-Test-Differenz).
        underfit_threshold (float): Schwellenwert für Underfitting (Train- und Testgenauigkeit < Threshold).

    Returns:
        pd.DataFrame: DataFrame mit den Ergebnissen ("Model", "Overfit", "Underfit").
    """
    results = {"Model": [], "Overfit": [], "Underfit": []}

    for name, train, test in zip(model_names, train_accuracies, test_accuracies):
        overfit = (train - test) > overfit_threshold
        underfit = (train < underfit_threshold) and (test < underfit_threshold)

        results["Model"].append(name)
        results["Overfit"].append(overfit)
        results["Underfit"].append(underfit)

        # Ausgabe nur in Variablen speichern (optional für Debugging)
        log_message = (
            f"Model: {name}\n"
            f"  Overfit: {overfit} (Train-Test Difference: {train - test:.3f})\n"
            f"  Underfit: {underfit} (Train Accuracy: {train:.3f}, Test Accuracy: {test:.3f})\n"
        )

    # Ergebnisse in DataFrame umwandeln
    result_df = pd.DataFrame(results)

    return result_df

def plot_under_over_fit(model_names, train_accuracies, test_accuracies, plot_dir):
    """
    Visualisiert Trainings- und Testabweichungen, um Overfitting oder Underfitting zu bewerten.

    Args:
        model_names (list): Liste von Modellnamen.
        train_accuracies (list): Trainingsgenauigkeiten.
        test_accuracies (list): Testgenauigkeiten.
        plot_dir (str): Speicherort des generierten Plots.
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

        print(f"Under-/Overfitting-Analyse gespeichert unter: {plot_path}")

    except Exception as e:
        print(f"Fehler in der Funktion 'plot_under_over_fit': {e}")

def get_best_model(models, evaluation_results, fit_results, primary_metric="ROC-AUC", tie_breaker="F1-Score"):
    """
    Wählt das beste Modell basierend auf einer Hauptmetrik aus, schließt aber Modelle mit Over-/Underfitting aus.

    Args:
        models (dict): Dictionary der trainierten Modelle.
        evaluation_results (pd.DataFrame): DataFrame mit den Metriken aller Modelle.
        fit_results (pd.DataFrame): DataFrame mit den FIT-Analyseergebnissen ("Overfit", "Underfit").
        primary_metric (str): Hauptmetrik, nach der das beste Modell ausgewählt wird (z.B.: "ROC-AUC").
        tie_breaker (str): Sekundärmetrik, die bei einem Gleichstand verwendet wird.

    Returns:
        tuple: Bestes Modell und der Name des Modells (als String).
    """
    # Kombiniere die FIT-Ergebnisse mit den Modell-Metriken
    if "Model" not in fit_results.columns or "Overfit" not in fit_results.columns or "Underfit" not in fit_results.columns:
        raise ValueError("Die FIT-Ergebnisse müssen die Spalten 'Model', 'Overfit' und 'Underfit' enthalten.")

    combined_results = pd.merge(evaluation_results, fit_results, on="Model", how="inner")

    # Filtere alle Modelle aus, die Overfitting oder Underfitting aufweisen
    valid_models = combined_results[(combined_results["Overfit"] == False) & (combined_results["Underfit"] == False)]

    if valid_models.empty:
        raise ValueError("Keine Modelle erfüllen die Kriterien: Kein Overfit und kein Underfit.")

    # Modell basierend auf der Hauptmetrik auswählen
    if primary_metric not in valid_models.columns:
        raise ValueError(
            f"Die Metrik '{primary_metric}' ist nicht in den Ergebnissen vorhanden. Verfügbare Metriken: {valid_models.columns.tolist()}")

    best_model_data = valid_models.loc[valid_models[primary_metric].idxmax()]
    top_candidates = valid_models[valid_models[primary_metric] == best_model_data[primary_metric]]

    # Bei Gleichstand verwende die Sekundärmetrik
    if len(top_candidates) > 1 and tie_breaker in valid_models.columns:
        top_candidates = top_candidates.sort_values(by=tie_breaker, ascending=False)
        best_model_data = top_candidates.iloc[0]

    # Namen und Modell extrahieren
    best_model_name = best_model_data["Model"]
    best_model = models[best_model_name]

    print(
        f"Das beste Modell basierend auf '{primary_metric}' ist '{best_model_name}' mit einem Wert von {best_model_data[primary_metric]:.4f}.")
    if len(top_candidates) > 1:
        print(f"  (Gleichstand gelöst durch '{tie_breaker}'.)")
    return best_model, best_model_name

def get_critical_employees(model, X_transformed, df, scaler_file="Models/scaler.pkl",
                           feature_names_file="Models/lightgbm_feature_names.pkl", pca=None, threshold=0.0):
    """
    Identifiziert kritische Mitarbeiter mit einer Fluktuationswahrscheinlichkeit höher
    als der angegebenen Schwelle unter Verwendung eines gespeicherten Scalers.
    Optional wird PCA angewendet. Funktioniert für LightGBM-Modelle und andere sklearn-ähnliche Modelle.

    Args:
        model: Das trainierte Modell für die Vorhersage.
        X_transformed (np.ndarray): Die preprocessierten Features (unskaliert).
        df (pd.DataFrame): Der DataFrame mit zusätzlichen Informationen (z. B. Mitarbeiterdaten).
        scaler_file (str): Dateiname des gespeicherten Scalers.
        feature_names_file (str): Pfad zu den gespeicherten Feature-Namen (für LightGBM).
        pca (PCA, optional): Ein optionales PCA-Objekt für Dimensionenreduktion.
        threshold (float): Der Schwellenwert für die Fluktuationswahrscheinlichkeit.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Kritische Mitarbeiter und die Top 15 Mitarbeiter nach Wahrscheinlichkeit.
    """

    print(f"Shape von X_transformed: {X_transformed.shape}")
    print(f"Shape von df: {df.shape}")

    # Dynamisch letzten Monat und letztes Jahr aus dem Datensatz bestimmen
    max_year = df["Jahr"].max()  # Höchstes Jahr im Datensatz
    max_month_in_max_year = df[df["Jahr"] == max_year]["Monat"].max()  # Höchster Monat im höchsten Jahr
    print(f"Letztes Jahr im Datensatz: {max_year}, letzter Monat im Jahr: {max_month_in_max_year}")

    # ** Scaler laden und anwenden **
    print(f"Lade Scaler aus Datei '{scaler_file}'...")
    try:
        scaler = joblib.load(scaler_file)  # Scaler laden
        X_transformed = scaler.transform(X_transformed)  # Daten skalieren
        print("Testdaten erfolgreich skaliert.")
    except Exception as e:
        print(f"Fehler beim Laden oder Anwenden des Scalers: {e}")
        return None, None

    # ** Feature-Namen laden und auf die Eingabedaten anwenden (nur für LightGBM notwendig) **
    if isinstance(model, lgb.LGBMClassifier) and feature_names_file:
        try:
            with open(feature_names_file, 'rb') as f:
                feature_names = joblib.load(f)
            print(f"Feature-Namen erfolgreich geladen: {feature_names}")

            # Konvertiere X_transformed in einen DataFrame mit den korrekten Spaltennamen
            X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        except Exception as e:
            print(f"Fehler beim Laden oder Anwenden der Feature-Namen: {e}")
            return None, None

    # ** PCA nur dann laden, wenn das Modell Logistic Regression ist **
    if pca is None and isinstance(model, LogisticRegression):
        pca_path = "Models/Back/pca_model.pkl"
        if os.path.exists(pca_path):
            try:
                print(f"Lade gespeichertes PCA-Modell aus '{pca_path}'...")
                pca = joblib.load(pca_path)
                X_transformed = pca.transform(X_transformed)
                print(f"PCA-Transformation erfolgreich angewendet. Shape: {X_transformed.shape}")
            except Exception as e:
                print(f"Fehler beim Laden oder Anwenden von PCA: {e}")
                return None, None
        else:
            print(f"[WARNUNG] Kein gespeichertes PCA-Modell unter '{pca_path}' gefunden. Fortfahren ohne PCA.")

    # ** Berechnung der Vorhersagen **
    print("Berechnung der Vorhersagen...")
    try:
        if hasattr(model, "predict_proba"):
            # Für sklearn-ähnliche Modelle, einschließlich LightGBM
            y_proba = model.predict_proba(X_transformed)[:, 1]
        elif hasattr(model, "predict"):
            # Für Modelle wie Keras (z. B. ohne predict_proba)
            y_proba = model.predict(X_transformed).flatten()
        else:
            raise AttributeError(f"Modelltyp '{type(model)}' wird nicht unterstützt.")
    except Exception as e:
        print(f"Fehler bei den Vorhersagen: {e}")
        return None, None

    # ** Verknüpfung der Wahrscheinlichkeiten mit den Daten **
    try:
        # Fluktuationswahrscheinlichkeit in Prozent umrechnen
        df["Fluktuationswahrscheinlichkeit"] = y_proba * 100
    except ValueError as e:
        print(f"Fehler beim Hinzufügen der Fluktuationswahrscheinlichkeit: {e}")
        return None, None

    # ** Kritische Mitarbeiter identifizieren **
    try:
        # Filterung: Nur Mitarbeiter mit Fluktuation aktiv (0) und aus dem letzten Monat/Jahr
        df = df[
            (df["Fluktuation"] == 0) &
            (df["Monat"] == max_month_in_max_year) &
            (df["Jahr"] == max_year)
            ]
        print(f"Shape nach Filtern nach Fluktuation, Monat und Jahr: {df.shape}")

        critical_employees = df[df["Fluktuationswahrscheinlichkeit"] > threshold]
        critical_employees = critical_employees.sort_values(
            by="Fluktuationswahrscheinlichkeit", ascending=False
        )

        # Dubletten anhand von Mitarbeiter_ID entfernen
        if "Mitarbeiter_ID" in critical_employees.columns:
            duplicate_count = critical_employees.duplicated(subset=["Mitarbeiter_ID"]).sum()
            if duplicate_count > 0:
                print(f"Warnung: {duplicate_count} doppelte Einträge basierend auf 'Mitarbeiter_ID' wurden entfernt.")
                critical_employees = critical_employees.drop_duplicates(subset=["Mitarbeiter_ID"])

        # Top 15 kritische Mitarbeiter
        top_15 = critical_employees.head(15)
    except Exception as e:
        print(f"Fehler bei der Sortierung der kritischen Mitarbeiter: {e}")
        return None, None

    # ** Rückgabe der Ergebnisse **
    print("Kritische Mitarbeiter erfolgreich identifiziert.")
    return critical_employees, top_15

def get_critical_employees_all_models(models, X_transformed, df, scaler_file="Models/scaler.pkl",
                                      feature_names_file="Models/lightgbm_feature_names.pkl", pca=None, threshold=0.0):
    """
    Identifiziert kritische Mitarbeiter und sammelt Ergebnisse für mehrere Modelle,
    unter Berücksichtigung eines gespeicherten Scalers und optionaler PCA.
    Unterstützt LightGBM mit gespeicherten Feature-Namen.

    Args:
        models (dict): Ein Dictionary der Modelle, wobei die Keys die Modellnamen sind.
        X_transformed (np.ndarray): Die preprocessierten Features (unskaliert).
        df (pd.DataFrame): Der DataFrame mit Mitarbeiterinformationen.
        scaler_file (str): Dateiname des gespeicherten Scalers.
        feature_names_file (str): Pfad zu den gespeicherten Feature-Namen (für LightGBM).
        pca (PCA, optional): Ein optionales PCA-Objekt für Dimensionenreduktion.
        threshold (float): Der Schwellenwert für die Fluktuationswahrscheinlichkeit.

    Returns:
        dict: Ein Dictionary mit Modellnamen als Keys und Tuple-Resultaten als Values.
    """
    results = {}

    print(f"\nShape von X_transformed: {X_transformed.shape}")
    print(f"Shape von df: {df.shape}")

    # Dynamisch letzten Monat und Jahr aus dem DataFrame bestimmen
    max_year = df["Jahr"].max()
    max_month_in_max_year = df[df["Jahr"] == max_year]["Monat"].max()
    print(f"Letztes Jahr im Datensatz: {max_year}, letzter Monat im Jahr: {max_month_in_max_year}")

    # ** Scaler laden und Testdaten skalieren **
    print(f"Lade Scaler aus '{scaler_file}'...")
    try:
        scaler = joblib.load(scaler_file)
        X_transformed = scaler.transform(X_transformed)
        print("Testdaten erfolgreich skaliert.")
    except Exception as e:
        print(f"Fehler beim Laden oder Anwenden des Scalers: {e}")
        return None

    # ** Synchronisierung zwischen X_transformed und df **
    if len(X_transformed) != len(df):
        print("WARNUNG: Dimensionen von X_transformed und df stimmen nicht überein. Synchronisiere Daten...")
        df = df.iloc[:len(X_transformed)].reset_index(drop=True).copy()

    # ** Iteration über alle Modelle **
    for model_name, model in models.items():
        print(f"\nBearbeite Modell: {model_name}")
        try:
            # Zurücksetzen der transformierten Features für das Modell
            X_model_transformed = X_transformed.copy()

            # ** Spezialbehandlung für PCA oder LightGBM **
            if isinstance(model, LogisticRegression):
                if pca is not None:
                    try:
                        print(f"PCA wird für das Modell '{model_name}' angewendet...")
                        X_model_transformed = pca.transform(X_model_transformed)
                        print(f"PCA-Transformation erfolgreich abgeschlossen. Shape: {X_model_transformed.shape}")
                    except Exception as e:
                        print(f"Fehler bei PCA für Modell '{model_name}': {e}. Fortfahren ohne PCA.")

            elif isinstance(model, lgb.LGBMClassifier):
                # Für LightGBM spezielle Feature-Namen anwenden
                if feature_names_file:
                    try:
                        with open(feature_names_file, 'rb') as f:
                            feature_names = joblib.load(f)
                        print(f"Feature-Namen erfolgreich geladen: {feature_names}")

                        # Konvertiere in DataFrame und entferne nicht-numerische Spalten
                        X_model_transformed = pd.DataFrame(X_model_transformed, columns=feature_names)
                    except Exception as e:
                        print(f"Fehler beim Laden oder Anwenden der Feature-Namen für '{model_name}': {e}")
                        results[model_name] = (pd.DataFrame(), pd.DataFrame())
                        continue

            # ** Sicherstellen, dass nur numerische Daten für die Vorhersage verwendet werden **
            if isinstance(X_model_transformed, pd.DataFrame):
                X_model_transformed = X_model_transformed.select_dtypes(include=[np.number])

            # ** Vorhersagen berechnen **
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_model_transformed)[:, 1]
                elif hasattr(model, "predict"):
                    y_proba = model.predict(X_model_transformed).flatten()
                else:
                    print(f"Das Modell '{model_name}' unterstützt weder 'predict_proba' noch 'predict'.")
                    results[model_name] = (pd.DataFrame(), pd.DataFrame())
                    continue
            except Exception as e:
                print(f"Fehler bei Vorhersage für Modell '{model_name}': {e}")
                results[model_name] = (pd.DataFrame(), pd.DataFrame())
                continue

            # ** Ergebnisse sammeln **
            df_copy = df.copy()
            df_copy["Fluktuationswahrscheinlichkeit"] = y_proba * 100

            # Daten für den letzten Monat und das letzte Jahr filtern
            df_copy = df_copy[
                (df_copy["Fluktuation"] == 0) &
                (df_copy["Monat"] == max_month_in_max_year) &
                (df_copy["Jahr"] == max_year)
                ]

            # Kritische Mitarbeiter und Top-15 bestimmen
            critical_employees = df_copy[df_copy["Fluktuationswahrscheinlichkeit"] > threshold]
            critical_employees = critical_employees.sort_values(by="Fluktuationswahrscheinlichkeit", ascending=False)
            critical_employees = critical_employees.drop_duplicates(subset=["Mitarbeiter_ID"], keep="first")

            top_15 = critical_employees.head(15)

            # Ergebnisse speichern
            results[model_name] = (critical_employees, top_15)

        except Exception as e:
            print(f"Fehler bei Modell '{model_name}': {e}")
            results[model_name] = (pd.DataFrame(), pd.DataFrame())

    return results

def save_results(data, file_name_base, output_dir):
    """
    Speichert Ergebnisse sowohl als CSV als auch als Excel im angegebenen Verzeichnis.

    Args:
        data (pd.DataFrame): Die zu speichernden Daten.
        file_name_base (str): Basename der Datei (ohne Endung).
        output_dir (str): Zielverzeichnis für die Speicherung.
    """
    csv_path = os.path.join(output_dir, f"{file_name_base}.csv")
    #xlsx_path = os.path.join(output_dir, f"{file_name_base}.xlsx")

    # Daten exportieren
    data.to_csv(csv_path, index=False)
    #data.to_excel(xlsx_path, index=False)

    # Feedback an den Nutzer
    print(f"\nDaten '{file_name_base}' wurden erfolgreich gespeichert:")
    print(f"- CSV: {csv_path}")
    #print(f"- Excel: {xlsx_path}")

def compare_model_top_employees(file_paths, output_file="Outputs/Vergleich_Top_15.csv"):
    """
    Lädt die Top-15-Mitarbeiter-Dateien für verschiedene Modelle, vergleicht Namen und IDs,
    zählt, wie oft eine Kombination aus Name und ID in den verschiedenen Modellen vorkommt,
    gibt Fluktuationswahrscheinlichkeiten je Modell in Prozent an und erstellt eine kompakte Zusammenfassung.

    Args:
        file_paths (dict): Dictionary mit Modellnamen als Key und Datei-Pfaden als Value.
        output_file (str): Name der Ausgabedatei, in der der Vergleich gespeichert wird.

    Returns:
        pd.DataFrame: Kompakter DataFrame mit Name, ID, Häufigkeiten, Fluktuationswahrscheinlichkeiten (in Prozent) und Übereinstimmung.
    """
    model_data = {}

    print("Lade Top-15-Mitarbeiter-Dateien...")
    for model, file_path in file_paths.items():
        try:
            print(f"Lade Datei für Modell '{model}' aus {file_path}...")
            df = pd.read_csv(file_path)

            # Prüfe notwendige Spalten
            required_columns = {"Name", "Mitarbeiter_ID", "Fluktuationswahrscheinlichkeit"}
            if not required_columns.issubset(df.columns):
                raise ValueError(
                    f"Die Datei '{file_path}' enthält nicht die nötigen Spalten: {required_columns}."
                )

            # Wähle nur benötigte Spalten aus
            df = df[["Name", "Mitarbeiter_ID", "Fluktuationswahrscheinlichkeit"]].copy()

            # Spalte mit Modellinformationen umbenennen
            df.rename(columns={"Fluktuationswahrscheinlichkeit": f"Fluktuation_{model} (%)"}, inplace=True)
            model_data[model] = df
        except Exception as e:
            print(f"Fehler beim Laden von {file_path}: {e}")
            continue

    # Kombiniere alle geladenen Daten
    print("\nKombiniere Daten aus allen Modellen...")
    combined_df = None
    for model, df in model_data.items():
        if combined_df is None:
            combined_df = df
        else:
            # Merge basierend auf Name und Mitarbeiter_ID
            combined_df = pd.merge(combined_df, df, on=["Name", "Mitarbeiter_ID"], how="outer")

    # Zähle Übereinstimmungen
    print("\nZähle Übereinstimmungen...")
    combined_df["Häufigkeit_in_Modellen"] = combined_df.notna().sum(axis=1) - 2  # Ignoriere 'Name' und 'Mitarbeiter_ID'

    # Überprüfen auf gleiche Einträge in mehreren Modellen
    combined_df["Übereinstimmung"] = combined_df["Häufigkeit_in_Modellen"] > 1

    # Sortiere die Daten alphabetisch nach Name und ID
    combined_df = combined_df.sort_values(by=["Name", "Mitarbeiter_ID"]).reset_index(drop=True)

    # Speichere die Ergebnisse
    print(f"Speichere die Ergebnisse in '{output_file}'...")
    try:
        combined_df.to_csv(output_file, index=False)
        print("Speicherprozess abgeschlossen.")
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

    return combined_df

# Hauptlogik für Fluktuationsanalyse und Modelltraining
def main():
    """
    Hauptlogik für die Fluktuationsanalyse und das Modelltraining.
    """
    plot_dir = "Plots"
    models_dir = "Models"
    output_dir = "Outputs"
    output_dir_all = "Outputs/all_models"
    os.makedirs(plot_dir, exist_ok=True)  # Verzeichnis für Plots erstellen
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_all, exist_ok=True)
    # Verzeichnis erstellen, falls nicht vorhanden

    # 1. Daten laden und vorbereiten
    print("\n### Schritt 1: Daten laden ###")
    print("Daten laden...")
    file_path = "HR_cleaned.csv"
    if not os.path.exists(file_path):
        print(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden.")
        return

    try:
        df = load_data(file_path)
        print("Daten laden abgeschlossen.")

    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return

    # 2. Daten vorverarbeiten (Preprocessing: One-Hot-Encoding und SMOTE)
    # Funktion entfernt 'Status', erstellt 'Fluktuation'
    print("\n### Schritt 2: Daten vorverarbeiten ###")
    print("Daten vorverarbeiten...")
    df, X_transformed, X_resampled, y_resampled, preprocessor = preprocess_data(df)
    print(f"Shape von X_transformed: {X_transformed.shape}")
    print(f"Shape von df (Originaldaten): {df.shape}")
    print("Preprocessing abgeschlossen.")

    # 3. Daten aufteilen und skalieren (Training/Test-Sets erstellen und skalieren)
    print("\n### Schritt 3: Daten aufteilen und skalieren ###")
    print("Daten aufteilen und skalieren...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X_resampled, y_resampled)
    print(f"Shape von X_train_scaled: {X_train.shape}")
    print(f"Shape von X_test_scaled: {X_test.shape}")
    print("Datenaufteilung und Skalierung abgeschlossen.")

    # 4. PCA-Dimensionalitätsreduktion anwenden (auf Trainings-, Test- und Gesamtdaten)
    print("\n### Schritt 4: PCA-Dimensionalitätsreduktion ###")
    pca_file = os.path.join(models_dir, "pca_model.pkl")
    if os.path.exists(pca_file):
        print("Gespeichertes PCA-Modell gefunden. Laden...")
        pca = joblib.load(pca_file)

        # Transformieren der existierenden Daten
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_full_pca = pca.transform(X_resampled)

    else:
        print("PCA-Dimensionalitätsreduktion durchführen...")
        X_train_pca, X_test_pca, X_full_pca, pca = reduce_dimensions(X_train, X_test, X_resampled)

        # Speichern der PCA-Transformation
        joblib.dump(pca, pca_file)
        print(f"PCA-Modell gespeichert: {pca_file}")
    print(f"Shape nach PCA: {X_full_pca.shape}")
    print("PCA abgeschlossen.")

    # 5. Modellauswahl basierend auf den Anforderungen
    print("\n### Schritt 5: Modellauswahl ###")
    print("Modellauswahl durchführen...")
    include_models = model_selection()  # Die Funktion bestimmt, welche Modelle trainiert werden sollen
    include_models_all = include_models
    if len(include_models) == 1:
        print(f"Ausgewähltes Modell: {include_models[0]}")
    else:
        print(f"Ausgewählte Modelle: {', '.join(include_models)}")

    #6. Benutzerwahl abfragen
    print("\n### Schritt 6 Wählen: Gespeicherte Modelle oder neues Training ###")
    use_saved_models = get_user_choice(models_dir)

    # 7. Modelle trainieren basierend auf der Auswahl
    print("\n### Schritt 7: Modelle trainieren ###")
    print("Modelle trainieren...")

    # Modelle trainieren oder laden
    trained_models_results = train_models(
        X_train, X_test, y_train, y_test, include_models, use_saved_models, models_dir)

    # Sicherstellen, dass Ergebnisse verfügbar sind
    if not trained_models_results or len(trained_models_results) == 0:
        # Überprüfen, ob Modelle überhaupt existieren
        if use_saved_models:
            print("WARNUNG: Gespeicherte Modelle wurden angefordert, aber es gibt keine gespeicherten Modelle.")
            print("Starte neues Training...")
            use_saved_models = False  # Umschalten auf Training neuer Modelle
            trained_models_results = train_models(
                X_train_pca, X_test_pca, y_train, y_test, include_models, use_saved_models, models_dir)

        # Nach erneutem Versuch prüfen, ob Training erfolgreich war
        if not trained_models_results or len(trained_models_results) == 0:
            print("FEHLER: Keine Modelle wurden erfolgreich trainiert oder geladen. Programm wird beendet.")
            return

    # Extrahieren der Modelle in ein übersichtliches Dictionary
    models = {}
    y_probas = {}  # Zum Speichern der Wahrscheinlichkeiten
    nn_model = None  # Haltemechanismus für Neural Network-Modell, falls vorhanden

    for result in trained_models_results:
        # Model spezifisch prüfen
        model_name = result.get("model_name")
        trained_model = result.get("trained_model")
        roc_auc_score_nn = result.get("y_test_proba", None)

        # Speichern des Modells
        if model_name and trained_model:
            models[model_name] = trained_model

        # Für Neural Network Wahrscheinlichkeiten speziell speichern
        if model_name == "Neural Network":
            nn_model = trained_model  # Setzt das neuronale Modell
            if roc_auc_score_nn is not None:
                y_probas["Neural Network"] = result["y_test_proba"]  # Speichern der ROC-Wahrscheinlichkeit
        else:
            y_probas[model_name] = result.get("roc_auc_score")  # Für andere Modelle prüfen: ROC direkt speichern

    # Prüfen, ob überhaupt ein Modell trainiert wurde
    if not models:
        print("WARNUNG: Keine Modelle stehen für die Analyse bereit.")
        return  # Kein Modell - Abbruch

    print(f"{len(models)} Modelle erfolgreich trainiert.")

    # 8. Analyse von Overfitting/Underfitting
    print("\n### Schritt 8: Over-/Underfitting-Analyse und Plots erstellen ###")

    # Variablen definieren
    model_names = [result["model_name"] for result in trained_models_results]  # Namen der Modelle
    train_accuracies = [result["train_accuracy"] for result in trained_models_results]  # Trainingsgenauigkeiten
    test_accuracies = [result["test_accuracy"] for result in trained_models_results]  # Testgenauigkeiten
    roc_auc_scores = [result["roc_auc_score"] for result in trained_models_results]  # ROC AUC-Werte

    # Zusätzliche Metriken optional extrahieren
    f1_scores = [result.get("f1_score", None) for result in trained_models_results]  # Optional: F1-Scores
    precisions = [result.get("precision", None) for result in trained_models_results]  # Optional: Präzision
    recalls = [result.get("recall", None) for result in trained_models_results]  # Optional: Recall-Werte


    # Horizontales Balkendiagramm für Modellvergleiche erstellen
    print(print("\nHorizontales Balkendiagramm für Modellvergleiche erstellen ..."))
    plot_horizontal_comparison(
        model_names=model_names,
        train_accuracies=train_accuracies,
        test_accuracies=test_accuracies,
        roc_auc_scores=roc_auc_scores,
        f1_scores=f1_scores,
        precisions=precisions,
        recalls=recalls,
        plot_dir=plot_dir,
        sort_by="Test Accuracy",  # Sortieren nach Testgenauigkeit
        include_train=True,  # Trainingsgenauigkeiten einbeziehen
        xlabel="Model Performance (Higher is Better)"
    )
    print("Horizontale Plots wurden erfolgreich erstellt.")

    # Over-/Underfitting testen und Ergebnisse visualisieren
    print("\nAnalyse von Overfitting und Underfitting ...")
    fit_results = test_under_over_fit(model_names, train_accuracies, test_accuracies)
    plot_under_over_fit(model_names, train_accuracies, test_accuracies, plot_dir)
    print("Over-/Underfitting-Ergebnisse:")
    print(fit_results)



    # 9. ROC-Kurven für alle Modelle erstellen
    print("\n### Schritt 9: ROC-Kurven erstellen ###")
    print("ROC-Kurven erstellen...")
    try:
        # Dictionary `models` enthält alle trainierten Modelle
        # Dictionary `y_probas` enthält die Wahrscheinlichkeiten

        # Aufruf der ROC-Plot-Funktion
        plot_combined_roc_curves(
            models=models,
            X_test=X_test,
            y_test=y_test,
            nn_model=nn_model,  # Das neuronale Netzwerk-Modell
            y_proba_nn=y_probas.get("Neural Network", None),  # Wahrscheinlichkeiten für nn
            plot_dir=plot_dir
        )

    except Exception as e:
        print(f"Fehler bei der Erstellung der ROC-Kurven: {e}")

    # 10. Modelle bewerten
    print("\n### Schritt 10: Modellbewertung ###")
    print("Modelle bewerten...")
    # Übergabe der PCA-Daten zusätzlich an die Funktion
    evaluation_results = evaluate_models(
        models=models,
        X_test=X_test,  # Original-Testdaten
        y_test=y_test,  # Wahre Labels
        plot_dir=plot_dir  # Verzeichnis für Plots
    )

    # 11. Bestes Modell auswählen
    print("\n### Schritt 11: Modellauswahl ###")
    print("\nBestes Modell auswählen...")
    try:
        best_model, best_model_name = get_best_model(models, evaluation_results, fit_results, primary_metric="ROC-AUC",
                                                     tie_breaker="F1-Score")
        print(f"\nDas beste Modell ist: {best_model_name}")
    except ValueError as e:
        print(f"Fehler bei der Modellauswahl: {e}")
        return

    # 12. Kritische Mitarbeiter und Top 15 ermitteln
    print("\n### Schritt 12: Kritische Mitarbeiter und Top 15 ermitteln ###")
    print("Kritische Mitarbeiter und Top 15 ermitteln...")
    try:
        # Überprüfen, ob PCA angewendet werden soll (z. B. für die logistische Regression)
        apply_pca = best_model_name == "Logistic Regression"  # PCA nur bei log. Regression anwenden

        # Funktionsaufruf, mit oder ohne PCA, abhängig vom Modell
        critical_employees, top_15_mitarbeiter = get_critical_employees(
            best_model,
            X_transformed,
            df,
            pca=pca if apply_pca else None  # PCA nur übergeben, wenn logistische Regression
        )

        if critical_employees.empty:
            print("WARNUNG: Keine kritischen Mitarbeiter gefunden (keine Wahrscheinlichkeiten > 70%).")
        else:
            print(f"Anzahl der kritischen Mitarbeiter: {len(critical_employees)}")

        if top_15_mitarbeiter.empty:
            print("WARNUNG: Keine Top 15 Mitarbeiter gefunden.")
        else:
            print(
                f"Top 15 Mitarbeiter:\n{top_15_mitarbeiter[['Mitarbeiter_ID', 'Name', 'Fluktuationswahrscheinlichkeit']]}")
    except Exception as e:
        print(f"Fehler beim Ermitteln der kritischen Mitarbeiter: {e}")
        return

    # 13. Kritische Mitarbeiter und Top 15 für jedes Modell ermitteln
    print("\n### Schritt 13: Kritische Mitarbeiter und Top 15 für jedes Modell ermitteln ###")
    print("Kritische Mitarbeiter und Top 15 ermitteln...")

    try:
        for model_name in include_models_all:  # Verwende die Modellnamen aus der Auswahl
            try:
                print(f"\nBearbeite Modell: {model_name}")

                apply_pca = model_name == "Logistic Regression"  # Prüfen, ob PCA erforderlich ist

                # Funktionsaufruf für kritische Mitarbeiter
                # Übergib die Modelle als Dictionary mit nur dem aktuellen Modell
                critical_employees_data = get_critical_employees_all_models(
                    models={model_name: models[model_name]},  # Nur das aktuelle Modell verarbeiten
                    X_transformed=X_transformed,
                    df=df,
                    pca=pca if apply_pca else None
                )

                # Ergebnisse extrahieren
                if model_name in critical_employees_data:
                    critical_employees, top_15_employees = critical_employees_data[model_name]

                    # Ergebnisse validieren und anzeigen
                    if critical_employees.empty:
                        print(f"WARNUNG: Keine kritischen Mitarbeiter für {model_name} gefunden.")
                    else:
                        print(f"Anzahl der kritischen Mitarbeiter für {model_name}: {len(critical_employees)}")
                        print(critical_employees.head(5))

                    if top_15_employees.empty:
                        print(f"WARNUNG: Keine Top 15 Mitarbeiter für {model_name} gefunden.")
                    else:
                        print(f"Top 15 Mitarbeiter für {model_name}:\n"
                              f"{top_15_employees[['Mitarbeiter_ID', 'Name', 'Fluktuationswahrscheinlichkeit']]}")

                    # Ergebnisse speichern
                    save_results(
                        data=critical_employees,
                        file_name_base=f"Critical_Mitarbeiter_{model_name.replace(' ', '_')}",
                        output_dir=output_dir_all
                    )
                    save_results(
                        data=top_15_employees,
                        file_name_base=f"Top_15_Mitarbeiter_{model_name.replace(' ', '_')}",
                        output_dir=output_dir_all
                    )
                else:
                    print(f"WARNUNG: Keine Ergebnisse für {model_name} vorhanden.")

            except KeyError:
                print(f"FEHLER: Modell '{model_name}' wurde nicht in 'models' gefunden.")
            except Exception as e:
                print(f"Fehler bei der Verarbeitung des Modells '{model_name}': {e}")

    except Exception as e:
        print(f"Fehler beim Ermitteln der kritischen Mitarbeiter und Top 15 für alle Modelle: {e}")

    # 13. Ergebnisse speichern
    print("\n### Schritt 13: Ergebnisse speichern ###")
    print("Ergebnisse speichern...")
    save_results(critical_employees, "Critical_Mitarbeiter", output_dir)
    save_results(top_15_mitarbeiter, "Top_15_Mitarbeiter", output_dir)

    print("### Komprimiert: Komprimiert")
    print("### Programm beendet.")

    #14. Vergleich der Modelle
    print("\n### Schritt 14: Vergleich der Modelle ###")
    file_paths = {
        "LightGBM": "Outputs/all_models/Top_15_Mitarbeiter_LightGBM.csv",
        "Logistic_Regression": "Outputs/all_models/Top_15_Mitarbeiter_Logistic_Regression.csv",
        "Neural_Network": "Outputs/all_models/Top_15_Mitarbeiter_Neural_Network.csv",
        "Random_Forest": "Outputs/all_models/Top_15_Mitarbeiter_Random_Forest.csv",
        "XGBoost": "Outputs/all_models/Top_15_Mitarbeiter_XGBoost.csv"
    }

    result = compare_model_top_employees(file_paths)
    print(result)


# Ausführung der Hauptfunktion
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    start_time = time.time()
    main()
    plt.close("all")
    k.clear_session()
    end_time = time.time()
    print(f"\nAnalyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")