import pandas as pd
import time

from data_loading import load_dataset
# Start-Timer
start_time = time.time()

def clean_dataset(df):
    """
    Clean the dataset by handling missing values, working on object-based columns, and removing duplicates.

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

    # Fülle fehlende Werte in 'Austrittsdatum' für aktive Mitarbeiter
    df["Austrittsdatum"] = df["Austrittsdatum"].fillna("Kein Austritt")  # Oder als None belassen

    # Fehlende Werte bearbeiten
    df = df.fillna({
        "Interne Weiterbildungen": 0,
        "Fortbildungskosten": 0,
        "Fehlzeiten_Krankheitstage": df["Fehlzeiten_Krankheitstage"].median(),
        "Überstunden": df["Überstunden"].median(),
    })

    # 'Einstellungsdatum' ohne Konvertierung bereinigen
    df["Einstellungsdatum"] = df["Einstellungsdatum"].astype(str)  # Sicherstellen, dass alle Werte Strings sind
    df["Einstellungsdatum"] = df["Einstellungsdatum"].str.strip()  # Führende/nachfolgende Leerzeichen entfernen
    df["Einstellungsdatum"] = df["Einstellungsdatum"].replace(["", "null", "N/A"], pd.NA)

    # Weitere Spalten bereinigen und konvertieren
    df["Wechselbereitschaft"] = df["Wechselbereitschaft"].astype(float)
    df["Fehlzeiten_Krankheitstage"] = df["Fehlzeiten_Krankheitstage"].astype(int)
    df["Überstunden"] = df["Überstunden"].astype(int)

    # Doppelte Einträge entfernen
    df.drop_duplicates(inplace=True)

    print("\nDataset cleaned successfully!")
    print(df.info())
    print(df.head())

    return df


# Dauerberechnung
end_time = time.time()
print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")


# Bereinigten Datensatz speichern
if __name__ == "__main__":
    file_path = "HR_Testdatensatz_filtered.csv"
    df = pd.read_csv(file_path)  # Datensatz laden
    cleaned_df = clean_dataset(df)


    # Bereinigten Datensatz speichern
    cleaned_file_path = "HR_cleaned.csv"
    cleaned_df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned dataset saved to {cleaned_file_path}")
