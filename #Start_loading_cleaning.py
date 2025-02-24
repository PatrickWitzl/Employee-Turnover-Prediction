import subprocess
import os
import time

# Start-Timer
start_time = time.time()

# Liste der Python-Skripte, die ausgeführt werden sollen
all_pys = [
    "data_loading.py",
    "data_cleaning.py"
]

def start_loading_cleaning():
    processes = []  # Hier speichern wir alle gestarteten Prozesse
    for py in all_pys:
        # Prüfen, ob das Skript existiert
        if not os.path.exists(py):
            print(f"Datei '{py}' wurde nicht gefunden. Überspringen...")
            continue

        print(f"Starte {py}...")
        # Skript ausführen
        process = subprocess.Popen(["python3", py], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((py, process))

    print("\nAlle verfügbaren Machine-Learning-Skripte wurden gestartet.\n")

    # Ausgabe der Prozesse überwachen
    for py, process in processes:
        try:
            stdout, stderr = process.communicate()  # Warten auf Abschluss
            if process.returncode == 0:
                print(f"{py} wurde erfolgreich ausgeführt:\n{stdout.decode()}")
            else:
                print(f"Fehler beim Ausführen von {py}:\n{stderr.decode()}")
        except Exception as e:
            print(f"Unerwarteter Fehler beim Ausführen von {py}: {str(e)}")

# Dauerberechnung
end_time = time.time()
print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")



# Hauptfunktion starten
if __name__ == "__main__":
    start_loading_cleaning()