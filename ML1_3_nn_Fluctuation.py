import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model
from sklearn.decomposition import PCA
import time

# Start-Timer
start_time = time.time()

# Ausgabeordner für Daten und Plots definieren
PLOT_DIR = os.path.join("Outputs","ML1")
os.makedirs(PLOT_DIR, exist_ok=True)

# Modell laden
nn_model = load_model('nn_model.keras')

# Ursprüngliche Daten vorbereiten
df = pd.read_csv("HR_cleaned.csv")
df_copy = df.copy()


df_copy['Fluktuation'] = df_copy['Status'].apply(lambda x: 1 if 'Ausgeschieden' in x else 0)
df_copy.drop(columns=['Status'], inplace=True)

X = df_copy.drop(columns=['Fluktuation', 'Mitarbeiter_ID'], errors='ignore')

# Preprocessing
categorical_columns = X.select_dtypes(include='object').columns
numerical_columns = X.select_dtypes(exclude='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_columns),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
    ]
)
X_transformed = preprocessor.fit_transform(X)

# PCA anwenden
pca = PCA(n_components=50, svd_solver='arpack')
X_transformed_pca = pca.fit_transform(X_transformed)

# Funktion definieren, um alle relevanten Mitarbeiter zu speichern und die Top 15 auszugeben
def get_critical_employees(nn_model, X_transformed, df_copy):
    """
    Gibt alle Mitarbeiter mit einer Fluktuationswahrscheinlichkeit über 80% zurück
    und die Top 15 mit der höchsten Kündigungswahrscheinlichkeit.

    Args:
        nn_model: Das trainierte neuronale Netzwerkmodell.
        X_transformed: Der transformierte Merkmalsvektor des ursprünglichen Datensatzes.
        df_copy: Der ursprüngliche DataFrame mit den Mitarbeiterinformationen.

    Returns:
        Ein DataFrame mit den Mitarbeitern > 80% Wahrscheinlichkeiten
        sowie einen DataFrame mit den Top 15 Mitarbeitern.
    """
    # Wahrscheinlichkeiten für das neuronale Netz berechnen
    y_proba = nn_model.predict(X_transformed).ravel()

    # Wahrscheinlichkeiten in den DataFrame hinzufügen
    df_copy['Fluktuations_Wahrscheinlichkeit'] = y_proba

    # Alle Mitarbeiter mit einer Fluktuationswahrscheinlichkeit über 80%
    critical_employees = df_copy[df_copy['Fluktuations_Wahrscheinlichkeit'] > 0.80].copy()


    # DataFrame nach den Wahrscheinlichkeiten sortieren und die Top 15 auswählen
    top_15_mitarbeiter = critical_employees.sort_values(
        by='Fluktuations_Wahrscheinlichkeit', ascending=False
    ).head(15)

    return critical_employees, top_15_mitarbeiter


# Kritische Mitarbeiter und Top 15 abrufen
critical_employees, top_15 = get_critical_employees(
    nn_model, X_transformed_pca, df_copy
)

# Explizite Kopie von critical_employees, um SettingWithCopyWarning zu vermeiden
critical_employees = critical_employees.copy()

# Mitarbeiter_ID zu den Ergebnissen mit loc hinzufügen
critical_employees.loc[:, 'Mitarbeiter_ID'] = df_copy['Mitarbeiter_ID']
top_15.loc[:, 'Mitarbeiter_ID'] = df_copy.loc[top_15.index, 'Mitarbeiter_ID']

# Die Top 15 Mitarbeiter ausgeben
print("\nTop 15 Mitarbeiter mit der höchsten Fluktuationswahrscheinlichkeit:")
print(top_15[['Mitarbeiter_ID', 'Name','Fluktuations_Wahrscheinlichkeit']])

# Dateien speichern
critical_csv_path = os.path.join(PLOT_DIR, 'Critical_Mitarbeiter.csv')
critical_xlsx_path = os.path.join(PLOT_DIR, 'Critical_Mitarbeiter.xlsx')
top15_csv_path = os.path.join(PLOT_DIR, 'Top_15_Mitarbeiter.csv')
top15_xlsx_path = os.path.join(PLOT_DIR, 'Top_15_Mitarbeiter.xlsx')
# Alle kritischen Mitarbeiter (wahrscheinlich > 80%) speichern
critical_employees.to_csv(critical_csv_path, index=False)
print(f"Datei '{critical_csv_path}' wurde erfolgreich gespeichert.")

critical_employees.to_excel(critical_xlsx_path, index=False)
print(f"Datei '{critical_xlsx_path}' wurde erfolgreich gespeichert.")

# Die Top 15 Mitarbeiter separat speichern
top_15.to_csv(top15_csv_path, index=False)
print(f"Datei '{top15_csv_path}' wurde erfolgreich gespeichert.")

top_15.to_excel(top15_xlsx_path, index=False)
print(f"Datei '{top15_xlsx_path}' wurde erfolgreich gespeichert.")

# Erfolgreiche Speicherung zusammenfassen
print("\nDie Ergebnisse wurden erfolgreich gespeichert in 'HR_BEM/Output':")
print(f"- Alle kritischen Mitarbeiter: '{critical_csv_path}' und '{critical_xlsx_path}'")
print(f"- Top 15 Mitarbeiter: '{top15_csv_path}' und '{top15_xlsx_path}'")

# Dauerberechnung
end_time = time.time()
print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")
