import pandas as pd
import os
import time

def load_dataset(file_path, save_filtered_path):
    """
    Load dataset from CSV file, filter data based on year range (2015–2025),
    and save filtered data to a new CSV file.

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
        # CSV-Datei laden
        df = pd.read_csv(file_path)

        # Prüfen, ob eine Spalte "Jahr" vorhanden ist oder erstmalig erzeugen
        if "Jahr" not in df.columns:
            if "Einstellungsdatum" in df.columns:
                # Datum nach Jahr extrahieren, falls Datumsspalte vorhanden ist
                df["Einstellungsdatum"] = pd.to_datetime(df["Einstellungsdatum"])  # Datum parsen
                df["Jahr"] = df["Einstellungsdatum"].dt.year
            else:
                print("Fehler: Die Spalte 'Jahr' oder 'Einstellungsdatum' fehlt.")
                return None

        # Daten filtern: Jahr zwischen 2015 und 2025
        start_year = 2015
        end_year = 2025
        df_filtered = df[(df["Jahr"] >= start_year) & (df["Jahr"] <= end_year)]

        # Gefilterte Daten anzeigen
        print("Filtered Dataset (2015-2025):")
        print(df_filtered.info())
        print("\nFirst 5 rows preview of filtered data:")
        print(df_filtered.head())

        # Gefilterten Datensatz speichern
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


# Optional: Laufzeit messen
start_time = time.time()
end_time = time.time()
print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")

if __name__ == "__main__":
    # Beispiel: Eingabe- und Ausgabepfade
    file_path = "../data/HR_Testdatensatz.csv"  # Eingabedatei
    save_filtered_path = "../data/HR_Testdatensatz_filtered.csv"  # Speicherort für gefilterte Daten
    df = load_dataset(file_path, save_filtered_path)