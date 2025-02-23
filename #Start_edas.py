import subprocess
import os
import time

# Start-Timer
start_time = time.time()

# Liste der Python-Skripte, die ausgeführt werden sollen
scripts = [
    "eda_1.py",
    "eda_2_ausscheiden.py",
    "eda_3_Fehlzeiten_Krankheitstage.py",
    "eda_4_Fehlzeiten.py",
    "eda_5_Krankheitstage.py",
]

def start_scripts():
    processes = []  # Hier speichern wir alle gestarteten Prozesse
    for script in scripts:
        # Prüfen, ob das Skript existiert
        if not os.path.exists(script):
            print(f"Datei '{script}' wurde nicht gefunden. Überspringen...")
            continue

        print(f"Starte {script}...")
        # Skript ausführen
        process = subprocess.Popen(["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((script, process))

    print("\n✨ Alle verfügbaren Skripte wurden gestartet!\n")

    # Ausgabe der Prozesse überwachen
    for script, process in processes:
        try:
            stdout, stderr = process.communicate()  # Warten auf den Abschluss
            if process.returncode == 0:
                print(f" {script} wurde erfolgreich ausgeführt:\n{stdout.decode()}")
            else:
                print(f" Fehler beim Ausführen von {script}:\n{stderr.decode()}")
        except Exception as e:
            print(f" Unerwarteter Fehler beim Ausführen von {script}: {str(e)}")

# Dauerberechnung
end_time = time.time()
print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")

# Hauptfunktion starten
if __name__ == "__main__":
    start_scripts()
