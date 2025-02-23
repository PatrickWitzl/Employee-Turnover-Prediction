import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time

# Start-Timer
start_time = time.time()

# Ordner für Plots erstellen
PLOTS_DIR = Path("Plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Daten laden
df = pd.read_csv("HR_cleaned.csv")  # Datei muss im Arbeitsverzeichnis sein
df_copy = df.copy()

# Filtere die Daten für Abwesenheitsgrund == "Krankheit" und summiere die Fehlzeiten
fehlzeiten_krankheit = df_copy.loc[df_copy["Abwesenheitsgrund"] == "Krankheit", "Fehlzeiten_Krankheitstage"]

# Ausgabe der berechneten Summe
print(f"Fehlzeiten für Abwesenheitsgrund 'Krankheit': {fehlzeiten_krankheit}")

unique_counts = fehlzeiten_krankheit.value_counts()

# Ausgabe der einzigartigen Werte und ihrer Häufigkeiten
print(unique_counts)


# Zielvariable: Fehlzeiten kategorisieren
df_copy['Fehlzeiten_Kategorie'] = pd.cut(
    df_copy['Fehlzeiten_Krankheitstage'],
    bins=[0, 6, 12, 21],  # Drei Kategorien: 0-6, 7-12, 13-21
    labels=["Niedrig", "Mittel", "Hoch"]
)

# Daten filtern: Nur Ausfallgrund == "Krankheit"
df_krankheit = df_copy[df_copy['Abwesenheitsgrund'] == 'Krankheit']

print(df_krankheit)


# Gruppierung und Aggregation (z.B. Krankheitstage pro Jahr summieren, falls IDs nicht relevant sind)
df_krankheit_aggregated = df_krankheit.groupby('Jahr')['Fehlzeiten_Krankheitstage'].sum().reset_index()

print("Gefilterte und gruppierte Daten:")
print(df_krankheit_aggregated)

# Fehlzeiten nach Kategorien analysieren
print("\nVerteilung der Krankheitszeiten-Kategorien:")
print(df_krankheit['Fehlzeiten_Kategorie'].value_counts())

# 1. Visualisierung: Fehlzeiten nach Alter (Boxplot)
print("\nDaten für Boxplot überprüfen:")
print(df_krankheit[['Alter', 'Fehlzeiten_Krankheitstage']].dropna().head())  # Vorschau auf die Daten

if df_krankheit[['Alter', 'Fehlzeiten_Krankheitstage']].dropna().shape[0] > 0:  # Sicherstellen, dass Daten vorhanden sind
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_krankheit['Alter'], y=df_krankheit['Fehlzeiten_Krankheitstage'],hue=df_krankheit['Fehlzeiten_Kategorie'], palette="coolwarm")
    plt.title("Krankheitstage nach Alter")
    plt.xlabel("Alter")
    plt.ylabel("Krankheitstage")
    plt.savefig(PLOTS_DIR / "eda_5_01_boxplot_alter_krankheitstage.png")
    plt.show()
else:
    print("Keine ausreichenden Daten für den Boxplot vorhanden.")

# 2. Visualisierung: Heatmap der Korrelationen mit Fehlzeiten
plt.figure(figsize=(8, 12))  # Größeres Plot-Fenster für bessere Darstellung
numerische_spalten = df_copy.select_dtypes(include=["float64", "int64"]).columns
heatmap_corr = df_copy[numerische_spalten].corr()['Fehlzeiten_Krankheitstage'].sort_values(ascending=False)
sns.heatmap(heatmap_corr.to_frame(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            cbar=True,
            linewidths=0.5
            )
plt.title("Korrelationen: Einfluss auf Krankheitstage", fontsize=14)
plt.yticks(rotation=0, fontsize=10)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "eda_5_02_korrelationen_krankheitstage_beschriftung_lesbar.png")
plt.show()

print('\nKorrelationen: Einfluss auf Krankheitstage')
print(heatmap_corr)

# 3. Analyse: Extremgruppen (Wenig vs. Viele Krankheitstage)
extrem_groessen = df_krankheit[(df_krankheit['Fehlzeiten_Krankheitstage'] <= 3) | (df_krankheit['Fehlzeiten_Krankheitstage'] >= 15)]
print("\nAnalyse der Extremgruppen (<= 3 oder >= 15 Krankheitstage):")
print(extrem_groessen.describe())

# 4. Cluster-Analyse: Ähnlichkeiten zwischen Mitarbeitern
print("\nFühre Cluster-Analyse durch...")
cluster_features = ['Alter', 'Fehlzeiten_Krankheitstage', 'Gehalt', 'Überstunden']

# Sicherstellen, dass Cluster-Features keine NaN-Werte enthalten
df_krankheit_copy = df_krankheit.dropna(subset=cluster_features)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_krankheit_copy[cluster_features])

# KMeans-Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_krankheit_copy['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualisierung der Cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df_krankheit_copy['Alter'], y=df_krankheit_copy['Fehlzeiten_Krankheitstage'],
    hue=df_krankheit_copy['Cluster'], palette="Set1")
plt.title("Cluster-Analyse nach Alter und Krankheitstage")
plt.xlabel("Alter")
plt.ylabel("Krankheitstage")
plt.legend(title="Cluster")
plt.savefig(PLOTS_DIR / "eda_5_03_cluster_analyse.png")
plt.show()

# Berechnung der Mittelwerte der Merkmale je Cluster
cluster_means = df_krankheit_copy.groupby('Cluster')[cluster_features].mean()

print("\nDurchschnittswerte der Cluster:")
print(cluster_means)



# Weitere Visualisierung: Cluster-Boxplots
for feature in cluster_features:
    print(f"\nDaten für Boxplot von {feature} überprüfen:")
    print(df_krankheit_copy[['Cluster', feature]].dropna().head())  # Debugging-Daten anzeigen

    if df_krankheit_copy[['Cluster', feature]].dropna().shape[0] > 0:  # Sicherstellen, dass Daten vorhanden sind
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=df_krankheit_copy, hue='Cluster', palette="Set2")
        plt.title(f"Verteilung von {feature} pro Cluster")
        plt.xlabel("Cluster")
        plt.ylabel(feature)
        plt.savefig(PLOTS_DIR / "eda_5_03_Verteilung pro Cluster.png")
        plt.show()
    else:
        print(f"Keine ausreichenden Daten für den Boxplot von {feature} vorhanden.")

# Original-Scatterplot mit Mittelwert-Anmerkungen
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df_krankheit_copy['Alter'],
    y=df_krankheit_copy['Fehlzeiten_Krankheitstage'],
    hue=df_krankheit_copy['Cluster'],
    palette="Set1"
)
plt.title("Cluster-Analyse nach Alter und Krankheitstage")
plt.xlabel("Alter")
plt.ylabel("Krankheitstage")
plt.legend(title="Cluster")

# Mittelwerte auf dem Scatterplot anzeigen
for cluster, row in cluster_means.iterrows():
    plt.text(
        row['Alter'],
        row['Fehlzeiten_Krankheitstage'],
        f"Cluster {cluster}\n{row['Alter']:.1f}, {row['Fehlzeiten_Krankheitstage']:.1f}",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', alpha=0.5)
    )

plt.savefig(PLOTS_DIR / "eda_5_04_cluster_analyse_mit_mittelwerten.png")
plt.show()

# 5. Zeitliche Analyse, falls Daten vorhanden
if 'Jahr' in df_krankheit_copy.columns:
    plt.figure(figsize=(10, 6))
    durchschnittliche_krankheitstage = df_krankheit_copy.groupby('Jahr')['Fehlzeiten_Krankheitstage'].mean()
    plt.plot(durchschnittliche_krankheitstage.index, durchschnittliche_krankheitstage.values, marker='o', color='blue')
    plt.title("Durchschnittliche Krankheitstage pro Jahr")
    plt.xlabel("Jahr")
    plt.ylabel("Durchschnittliche Krankheitstage")
    plt.grid()
    plt.savefig(PLOTS_DIR / "eda_5_05_zeitliche_analyse.png")
    plt.show()

# Dauerberechnung
end_time = time.time()
print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")
