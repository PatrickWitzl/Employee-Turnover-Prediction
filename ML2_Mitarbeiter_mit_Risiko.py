import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans


def setup_directories():
    """ Erstellt Verzeichnisse für Outputs und Plots. """
    plots_dir = Path("Plots")
    plots_dir.mkdir(exist_ok=True)

    output_dir = Path("/Users/Patrick/Library/Mobile Documents/com~apple~CloudDocs/PycharmProjects/HR_BEM/Outputs/ML2")
    output_dir.mkdir(parents=True, exist_ok=True)

    return plots_dir, output_dir


def load_and_prepare_data():
    """ Lädt und bereitet den HR-Datensatz vor. """
    try:
        df_original = pd.read_csv("HR_cleaned.csv")  # Lade den HR-Datensatz
        print("Datensatz erfolgreich geladen.")
    except FileNotFoundError:
        print("Fehler: Datei 'HR_cleaned.csv' nicht gefunden.")
        exit()

    # Daten kopieren und Mitarbeiter mit Tenure < 0 entfernen
    df = df_original.copy()
    df = df[df['Tenure'] >= 0]

    # 'Fehlzeiten_Krankheitstage' als numerisch sicherstellen
    df['Fehlzeiten_Krankheitstage'] = pd.to_numeric(df['Fehlzeiten_Krankheitstage'], errors='coerce')

    # NaN-Werte entfernen
    df.dropna(subset=['Fehlzeiten_Krankheitstage'], inplace=True)

    # Zielvariable: Fehlzeiten kategorisieren
    df['Fehlzeiten_Kategorie'] = pd.cut(
        df['Fehlzeiten_Krankheitstage'],
        bins=[0, 6, 12, 21],  # Kategorien für Fehlzeiten
        labels=["Niedrig", "Mittel", "Hoch"]
    )

    # Zielvariable "Krankheitsrisiko"
    df['Krankheitsrisiko'] = (df['Fehlzeiten_Kategorie'] == 'Hoch').astype(int)

    return df, df_original


def check_target_variable_distribution(data):
    """ Prüft die Verteilung der Zielvariable. """
    print("Verteilung der Zielvariablen vor Datenaufteilung:")
    print(data['Krankheitsrisiko'].value_counts())

    if data['Krankheitsrisiko'].nunique() < 2:
        raise ValueError("Der gefilterte Datensatz enthält nicht beide Klassen.")
    elif data['Krankheitsrisiko'].value_counts().min() < 5:
        print("Warnung: Eine Klasse hat sehr wenige Datenpunkte!")


def split_data(df, features, target):
    """ Teilt Daten in Trainings- und Testdaten auf. """
    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def train_random_forest(X_train, y_train, numerical_features, categorical_features):
    """ Trainiert ein Random Forest Modell mit Preprocessing. """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    print("Modelltraining läuft...")
    model_pipeline.fit(X_train, y_train)
    print("Modell erfolgreich trainiert.")

    return model_pipeline


def evaluate_model(model_pipeline, X_test, y_test):
    """ Bewertet das trainierte Modell. """
    y_test_pred = model_pipeline.predict(X_test)
    y_test_prob = model_pipeline.predict_proba(X_test)[:, 1]

    print("Auswertung des Modells:")
    print(confusion_matrix(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))
    roc_auc = roc_auc_score(y_test, y_test_prob)
    print(f"ROC-AUC-Score: {roc_auc:.2f}")

    return roc_auc, y_test_pred, y_test_prob


def save_high_risk_employees(df, model_pipeline, features, output_dir):
    """ Speichert High-Risk-Mitarbeiter in eine Datei. """
    df['Krankheitsrisiko_Prognose'] = model_pipeline.predict(df[features])
    high_risk_employees = df[df['Krankheitsrisiko_Prognose'] == 1]

    # Speichern der High-Risk-Mitarbeiter
    high_risk_filename = output_dir / "High_Risk_Mitarbeiter.csv"
    high_risk_employees.to_csv(high_risk_filename, index=False)
    print(f"CSV-Datei mit High-Risk-Mitarbeitern gespeichert unter: {high_risk_filename}")

    return high_risk_employees


def save_top_5_employees(high_risk_employees, output_dir):
    """ Speichert die Top-5-Risikomitarbeiter. """
    top_5_employees = high_risk_employees.nlargest(5, 'Fehlzeiten_Krankheitstage')
    top_5_filename = output_dir / "Top_5_High_Risk_Mitarbeiter.csv"
    top_5_employees.to_csv(top_5_filename, index=False)

    print("Top 5 Mitarbeiter mit dem höchsten Krankheitsrisiko:")
    print(top_5_employees[['Name', 'Alter', 'Gehalt', 'Standort', 'Zufriedenheit', 'Fehlzeiten_Krankheitstage']])


def generate_plots(df, numerical_features, plots_dir):
    """ Erstellt Boxplots für numerische Features. """
    for feature in numerical_features:
        plt.figure(figsize=(8, 6))
        df.boxplot(column=feature, by='Krankheitsrisiko_Prognose', grid=False)
        plt.title(f'Boxplot {feature} nach Krankheitsrisiko')
        plt.suptitle("")
        plt.savefig(plots_dir / f"{feature}ML2_Boxplot.png")
        plt.close()

    print("Visualisierungen gespeichert im Verzeichnis:", plots_dir)


def perform_clustering(df, numerical_features, output_dir):
    """ Führt K-Means-Clustering für numerische Features durch. """
    print("K-Means Clustering läuft...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[numerical_features])

    clustered_filename = output_dir / "HR_Analysis_with_Clusters.csv"
    df.to_csv(clustered_filename, index=False)
    print(f"Cluster-Ergebnisse gespeichert unter: {clustered_filename}")


def main():
    # Timer starten
    start_time = time.time()

    # Verzeichnisse vorbereiten
    plots_dir, output_dir = setup_directories()

    # Daten laden und vorbereiten
    df, df_original = load_and_prepare_data()
    check_target_variable_distribution(df)

    # Features und Zielvariable
    features = ['Alter', 'Gehalt', 'Geschlecht', 'Standort', 'Job Level', 'Arbeitszeitmodell', 'Zufriedenheit',
                'Überstunden', 'Tenure']
    numerical_features = ['Alter', 'Gehalt', 'Überstunden', 'Tenure']
    categorical_features = ['Geschlecht', 'Standort', 'Job Level', 'Arbeitszeitmodell']
    target = 'Krankheitsrisiko'

    # Daten aufteilen
    X_train, X_test, y_train, y_test = split_data(df, features, target)

    # Modell trainieren
    model_pipeline = train_random_forest(X_train, y_train, numerical_features, categorical_features)

    # Modell bewerten
    evaluate_model(model_pipeline, X_test, y_test)

    # Ergebnisse speichern
    high_risk_employees = save_high_risk_employees(df, model_pipeline, features, output_dir)
    save_top_5_employees(high_risk_employees, output_dir)

    # Visualisierungen erstellen
    generate_plots(df, numerical_features, plots_dir)

    # Clustering durchführen
    perform_clustering(df, numerical_features, output_dir)

    # Timer stoppen
    end_time = time.time()
    print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")


if __name__ == "__main__":
    main()