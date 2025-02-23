import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud


# Funktion: Sicherstellen, dass ein Verzeichnis existiert
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Verzeichnis '{directory}' wurde erstellt.")
    else:
        print(f"Verzeichnis '{directory}' existiert bereits.")


def plot_combined_unique_employee_ids_by_year(df, column_name, year_column, plots_dir, plot_name):
    if "Mitarbeiter_ID" in df.columns:
        # Sicherstellen, dass die Jahr-Spalte existiert
        if year_column not in df.columns:
            raise KeyError(f"Spalte '{year_column}' nicht im DataFrame gefunden.")

        # Nur eindeutige Mitarbeiter-IDs
        df_unique = df.drop_duplicates(subset="Mitarbeiter_ID")

        # Sicherstellen, dass die gewünschte Spalte vorhanden ist
        if column_name in df.columns:
            # Daten nach Jahr filtern
            years = sorted(df_unique[year_column].dropna().unique())

            # Anzahl der Subplots definieren (einer pro Jahr)
            fig, axes = plt.subplots(len(years), 1, figsize=(8, 6 * len(years)), sharex=True)
            fig.suptitle(f"Kategorische Verteilung: {column_name} nach Jahr", fontsize=16)

            for i, year in enumerate(years):
                df_year = df_unique[df_unique[year_column] == year]
                ax = axes[i] if len(years) > 1 else axes
                sns.countplot(x=column_name, data=df_year, ax=ax)
                ax.set_title(f"Jahr: {year}")
                ax.set_xlabel(column_name)
                ax.set_ylabel("Anzahl eindeutiger IDs")

            # Layout und Speicherung
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Platz für den Haupttitel reservieren
            plt.savefig(plots_dir / plot_name, bbox_inches="tight")
            plt.show()
        else:
            raise KeyError(f"Spalte '{column_name}' nicht im DataFrame gefunden.")
    else:
        raise KeyError("Spalte 'Mitarbeiter_ID' nicht im DataFrame gefunden.")

# Funktion: Berechne und visualisiere Skewness und Kurtosis
def plot_skewness_kurtosis(df, numerische_spalten):
    print("\nSkewness und Kurtosis für numerische Spalten:")
    for spalte in numerische_spalten:
        skew = df[spalte].skew()
        kurt = df[spalte].kurtosis()
        print(f"{spalte}: Skewness = {skew:.2f}, Kurtosis = {kurt:.2f}")


# Funktion: Erstelle Histogramme der numerischen Spalten
def plot_histograms(df, numerische_spalten, plots_dir, plot_name):
    print("\n️Erstelle Histogramme für numerische Spalten...")
    num_cols = 3
    num_rows = -(-len(numerische_spalten) // num_cols)  # Aufrunden
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
    axes = axes.flatten()

    for i, spalte in enumerate(numerische_spalten):
        sns.histplot(df[spalte].dropna(), bins=30, kde=True, ax=axes[i], edgecolor="black")
        axes[i].set_title(f"Histogramm: {spalte}")

    for i in range(len(numerische_spalten), len(axes)):  # Leere Plots entfernen
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(plots_dir / plot_name, bbox_inches="tight")
    plt.show()


# Funktion: Scatterplots zwischen Spaltenpaaren generieren
def plot_scatterplots(df, relevante_spalten, plots_dir, plot_name):
    if len(relevante_spalten) < 2:
        print("Nicht genügend numerische Spalten für Scatterplots.")
        return

    print("\nErstelle eine kompakte Ansicht von Scatterplots...")
    num_cols = 3
    spalten_paare = [(relevante_spalten[i], relevante_spalten[j])
                     for i in range(len(relevante_spalten))
                     for j in range(i + 1, len(relevante_spalten))]
    num_rows = -(-len(spalten_paare) // num_cols)  # Aufrunden
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
    axes = axes.flatten()

    for i, (spalte_x, spalte_y) in enumerate(spalten_paare):
        sns.scatterplot(data=df, x=spalte_x, y=spalte_y, alpha=0.7, ax=axes[i])
        axes[i].set_title(f"{spalte_x} vs {spalte_y}")
        axes[i].set_xlabel(spalte_x)
        axes[i].set_ylabel(spalte_y)

    for i in range(len(spalten_paare), len(axes)):  # Leere Plätze entfernen
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(plots_dir / plot_name, bbox_inches="tight")
    plt.show()


# Funktion: Erstelle Heatmap der Korrelationsmatrix
def plot_correlation_heatmap(df, numerische_spalten, plots_dir, plot_name):
    if len(numerische_spalten) > 1:
        print("\nErstelle Heatmap der Korrelationsmatrix...")
        corr_matrix = df[numerische_spalten].corr()
        mask = np.eye(corr_matrix.shape[0], dtype=bool)

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1, mask=mask)
        plt.title("Korrelationsmatrix mit ausgeschlossener Hauptdiagonale")
        plt.savefig(plots_dir / plot_name, bbox_inches="tight")
        plt.show()


# Funktion: Dichteplot (KDE) Beispiel
def plot_kde(df, column, plots_dir, plot_name):
    if column in df.columns:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(df[column], fill=True, color="green")
        plt.title(f"Dichteverteilung von {column}")
        plt.savefig(plots_dir / plot_name, bbox_inches="tight")
        plt.show()


# Funktion: Wordcloud für Textspalten
def plot_wordcloud(df, text_column, plots_dir, plot_name):
    if text_column in df.columns:
        print(f"\nErstelle WordCloud für '{text_column}'...")
        text_data = " ".join(df[text_column].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(plots_dir / plot_name, bbox_inches="tight")
        plt.show()


def prepare_data(df):
    """
    Gruppiert Mitarbeiter in drei Hauptkategorien:
    - Aktiv
    - Ruhestand
    - Ausgeschieden
    """
    if 'Status' in df.columns:
        df["Ausscheiden_Kategorie"] = df["Status"].apply(
            lambda x: "Ruhestand" if x == "Ruhestand"
            else ("Ausgeschieden" if x == "Ausgeschieden"
                  else "Aktiv")
        )
    return df

def filter_status_changes(df):
    """
    Statusänderungen filtern. Beinhaltet die Kategorien:
    - Aktiv
    - Ausgeschieden
    - Ruhestand (optional, je nach Analyse)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Erwartet wurde ein pd.DataFrame, erhalten: {}".format(type(df)))

    # Filter für spezifische Analyse: Schließe "Ruhestand" aus, falls nicht relevant.
    filtered_df = df[df['Status'].isin(['Aktiv', 'Ausgeschieden', 'Ruhestand'])]
    return filtered_df


def plot_active_and_terminated(df, save_path):
    """
    Ergänzt Plots für 'Ruhestand' zu den existierenden 'Aktiv' und 'Ausgeschieden'.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    required_columns = {'Jahr', 'Mitarbeiter_ID', 'Ausscheiden_Kategorie'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Der DataFrame muss die Spalten {required_columns} enthalten.")

    # Berechnung von Ausgeschiedenen, Ruheständlern und Aktiven pro Jahr
    ausschiedene_per_year = (
        df[df['Ausscheiden_Kategorie'] == 'Ausgeschieden']
        .groupby('Jahr')['Mitarbeiter_ID']
        .nunique()
        .reset_index()
    )
    ausschiedene_per_year.rename(columns={'Mitarbeiter_ID': 'Anzahl_Ausgeschiedene'}, inplace=True)

    ruhestand_per_year = (
        df[df['Ausscheiden_Kategorie'] == 'Ruhestand']
        .groupby('Jahr')['Mitarbeiter_ID']
        .nunique()
        .reset_index()
    )
    ruhestand_per_year.rename(columns={'Mitarbeiter_ID': 'Anzahl_Ruhestand'}, inplace=True)

    aktive_per_year = (
        df[df['Ausscheiden_Kategorie'] == 'Aktiv']
        .groupby('Jahr')['Mitarbeiter_ID']
        .nunique()
        .reset_index()
    )
    aktive_per_year.rename(columns={'Mitarbeiter_ID': 'Anzahl_Aktive'}, inplace=True)

    # Plot: Ausgeschiedene
    plt.figure(figsize=(8, 6))
    plt.bar(ausschiedene_per_year['Jahr'], ausschiedene_per_year['Anzahl_Ausgeschiedene'], color='#FF5733', alpha=0.8,
            edgecolor='black')
    plt.title('Anzahl der Ausgeschiedenen pro Jahr')
    plt.xlabel('Jahr')
    plt.ylabel('Anzahl der Ausgeschiedenen')
    plt.tight_layout()
    plt.savefig(save_path / "ausgeschiedene_pro_jahr.png", dpi=300)
    plt.close()

    # Plot: Ruhestand
    plt.figure(figsize=(8, 6))
    plt.bar(ruhestand_per_year['Jahr'], ruhestand_per_year['Anzahl_Ruhestand'], color='#3498DB', alpha=0.8,
            edgecolor='black')
    plt.title('Anzahl der in Ruhestand getretenen Mitarbeiter pro Jahr')
    plt.xlabel('Jahr')
    plt.ylabel('Anzahl Ruhestand')
    plt.tight_layout()
    plt.savefig(save_path / "ruhestand_pro_jahr.png", dpi=300)
    plt.close()

    # Plot: Aktive
    plt.figure(figsize=(8, 6))
    plt.bar(aktive_per_year['Jahr'], aktive_per_year['Anzahl_Aktive'], color='#3CB371', alpha=0.8, edgecolor='black')
    plt.title('Anzahl der Aktiven pro Jahr')
    plt.xlabel('Jahr')
    plt.ylabel('Anzahl der Aktiven')
    plt.tight_layout()
    plt.savefig(save_path / "aktive_pro_jahr.png", dpi=300)
    plt.close()

def group_and_summarize_by_year(df):
    """
    Gruppiert Daten nach Jahr und Kategorie:
    - Aktiv
    - Ausgeschieden
    - Ruhestand
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Erwartet wurde ein pd.DataFrame, aber erhalten wurde: {}".format(type(df)))

    df = df.drop_duplicates(subset=['Jahr', 'Mitarbeiter_ID'])

    grouped = df.groupby(['Jahr', 'Ausscheiden_Kategorie'])['Mitarbeiter_ID'].count().reset_index()
    grouped_pivot = grouped.pivot(index='Jahr', columns='Ausscheiden_Kategorie', values='Mitarbeiter_ID').fillna(0)
    grouped_pivot = grouped_pivot.reset_index()

    return grouped_pivot


def create_boxplot(df_filtered, variable, save_path):
    """
     Kombinierter Boxplot für mehrere Variablen erstellen.

     Parameter:
         df_filtered (DataFrame): Gefilterter DataFrame für die Analyse.
         variables (List): Liste von Variablen, die visualisiert werden sollen.
         save_path (Path): Speicherpfad für den kombinierten Plot.
     """
    # Überprüfung, ob die notwendigen Spalten vorhanden sind
    if 'Ausscheiden_Kategorie' not in df_filtered.columns:
        print("Fehler: 'Ausscheiden_Kategorie'-Spalte fehlt im DataFrame.")
        return

    # DataFrame in langes Format umwandeln
    df_long = df_filtered.melt(
        id_vars='Ausscheiden_Kategorie',
        value_vars=variable,
        var_name='Variable',
        value_name='Wert'
    )

    # Plot erstellen
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df_long,
        x='Variable',
        y='Wert',
        hue='Ausscheiden_Kategorie',
        palette="coolwarm"
    )
    plt.title("Kombinierter Boxplot: Verteilung der Variablen nach Ausscheiden-Kategorie")
    plt.xlabel("Variable")
    plt.ylabel("Wert")
    plt.legend(title="Ausscheiden-Kategorie", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Verhindert Überlappungen
    plt.savefig(save_path)
    plt.show()


def analyze_extreme_groups(df_filtered):
    """Analyse der Extremgruppen (hohe/niedrige Gehälter)."""
    extreme_high = df_filtered[df_filtered['Gehalt'] > df_filtered['Gehalt'].quantile(0.95)]
    extreme_low = df_filtered[df_filtered['Gehalt'] < df_filtered['Gehalt'].quantile(0.05)]

    print("\nExtremgruppen: Hohe Gehälter:")
    print(extreme_high.describe())
    print("\nExtremgruppen: Niedrige Gehälter:")
    print(extreme_low.describe())



def enhanced_cluster_visualizations(df_filtered, cluster_features, save_dir):
    """
    Erweiterte Visualisierung für die Cluster-Analyse.

    Erstellt Scatterplots, Paarplots, 3D-Plots, Heatmaps, Silhouette-Analysen sowie
    interaktive Diagramme, basierend auf den Cluster-Analyse-Ergebnissen.
    """

    # Überprüfung auf gültige Daten für die Analyse
    df = df_filtered.dropna(subset=cluster_features)
    if df.empty:
        print("Fehler: Keine geeigneten Daten für die Cluster-Analyse verfügbar.")
        return

    # Standardisierung der Features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_features])

    # K-Means-Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Cluster-Mittelwerte berechnen
    cluster_means = df.groupby('Cluster')[cluster_features].mean()
    print("\nCluster-Mittelwerte:")
    print(cluster_means)


    # 2. Pairplot
    sns.pairplot(df[cluster_features + ['Cluster']], hue='Cluster', palette="Set2", diag_kind="kde", height=2.5)
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Paarplot der Cluster", fontsize=16)  # Titel hinzufügen
    plt.savefig(save_dir / "pairplot_cluster.png", bbox_inches="tight")
    plt.show()

    # 3. 3D Scatterplot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Alter'], df['Gehalt'], df['Tenure'], c=df['Cluster'], cmap='Set1', s=60)
    ax.set_title("3D-Cluster-Analyse", fontsize=16)
    ax.set_xlabel("Alter", fontsize=12)
    ax.set_ylabel("Gehalt", fontsize=12)
    ax.set_zlabel("Tenure", fontsize=12)
    legend_labels = [f"Cluster {i}" for i in range(kmeans.n_clusters)]
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Cluster", loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "cluster_3d.png", bbox_inches="tight")
    plt.show()

    # 4. Heatmap der Cluster-Mittelwerte
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Heatmap der Cluster-Mittelwerte", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Cluster", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "cluster_heatmap.png", bbox_inches="tight")
    plt.show()

#    # 5. Silhouette-Analyse
#   silhouette_vals = silhouette_samples(scaled_data, df['Cluster'])
#    y_ticks = []
#    y_lower, y_upper = 0, 0
#
#    plt.figure(figsize=(10, 6))
#    for cluster in range(kmeans.n_clusters):
#        cluster_silhouette_vals = np.sort(silhouette_vals[df['Cluster'] == cluster])
#        y_upper += len(cluster_silhouette_vals)
#        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
#        y_ticks.append((y_lower + y_upper) / 2)
#        y_lower += len(cluster_silhouette_vals)
#
#    avg_silhouette = np.mean(silhouette_vals)
#    plt.axvline(avg_silhouette, color="red", linestyle="--", label="Durchschnittlicher Silhouette-Wert")
#    plt.yticks(y_ticks, [f"Cluster {i}" for i in range(kmeans.n_clusters)], fontsize=12)
#    plt.ylabel("Cluster", fontsize=14)
#    plt.xlabel("Silhouette-Koeffizient", fontsize=14)
#    plt.title("Silhouette-Plot der Cluster", fontsize=16)
#    plt.legend(loc="upper right", fontsize=12)
#    plt.tight_layout()
#    plt.savefig(save_dir / "silhouette_plot.png", bbox_inches="tight")
#    plt.show()

#    # 6. Interaktiver Scatterplot
#    fig = px.scatter(
#        df, x='Alter', y='Gehalt', color='Cluster',
#        title="Cluster-Analyse: Alter vs. Gehalt (interaktiv)",
#        labels={'Cluster': 'Cluster'},
#        hover_data=cluster_features
#    )
#    fig.write_html(str(save_dir / "cluster_interactive.html"))
#    fig.show()

def yearly_active_and_terminated_analysis(df_filtered, save_path):
    """Analyse der Anzahl der aktiven und ausgeschiedenen Mitarbeiter pro Jahr."""
    if {'Jahr', 'Mitarbeiter_ID', 'Status'}.issubset(df_filtered.columns):
        # Entferne doppelte Einträge basierend auf Jahr und Mitarbeiter-ID
        unique_df = df_filtered.drop_duplicates(subset=['Jahr', 'Mitarbeiter_ID'])

        # Gruppiere und zähle die Anzahl der Ausgeschiedenen ("Ausgeschieden") und Aktiven ("Aktiv") nach Jahr
        yearly_data = (
            unique_df
            .groupby(['Jahr', 'Status'])['Mitarbeiter_ID']
            .nunique()
            .reset_index(name='Anzahl')
            .pivot(index='Jahr', columns='Status', values='Anzahl')
        )
        yearly_data = yearly_data.fillna(0).reset_index()  # Fehlende Werte mit 0 auffüllen

        # Ergebnisse anzeigen
        print("Jährliche Analyse (Anzahl Aktive und Ausgeschiedene):")
        print(yearly_data)

        # Visualisierung
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_data['Jahr'], yearly_data['Ausgeschieden'], marker='o', label="Ausgeschiedene", color='red')
        plt.plot(yearly_data['Jahr'], yearly_data['Aktiv'], marker='o', label="Aktive", color='green')
        plt.title("Anzahl Aktive und Ausgeschiedene Mitarbeiter pro Jahr")
        plt.xlabel("Jahr")
        plt.ylabel("Anzahl der Mitarbeiter")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
    else:
        print("Die Spalten 'Jahr', 'Mitarbeiter_ID' und 'Status' fehlen im DataFrame.")

def yearly_hired_analysis(df, save_path):
    """Analyse der Anzahl der eingestellten Mitarbeiter pro Jahr."""
    if 'Einstellungsdatum' in df.columns:
        # Stelle sicher, dass `Einstellungsdatum` ein Datumsformat ist
        df['Einstellungsjahr'] = pd.to_datetime(df['Einstellungsdatum']).dt.year

        # Gruppierung nach Jahr und Anzahl
        hires_per_year = df.groupby('Einstellungsjahr')['Mitarbeiter_ID'].nunique().reset_index()
        hires_per_year.columns = ['Jahr', 'Eingestellt']

        # Ergebnisse anzeigen
        print("Jährliche Analyse (Anzahl Eingestellter):")
        print(hires_per_year)

        # Visualisierung
        plt.figure(figsize=(12, 6))
        plt.bar(hires_per_year['Jahr'], hires_per_year['Eingestellt'], color='blue', alpha=0.7, label="Eingestellt")
        plt.title("Jährliche Anzahl eingestellter Mitarbeiter")
        plt.xlabel("Jahr")
        plt.ylabel("Anzahl der eingestellten Mitarbeiter")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc="upper right")
        plt.savefig(save_path)
        plt.show()
    else:
        print("Die Spalte 'Einstellungsdatum' fehlt im DataFrame.")

def yearly_terminated_analysis(df, save_path):
    """Analyse der Anzahl der ausgeschiedenen Mitarbeiter pro Jahr."""
    if {'Jahr', 'Mitarbeiter_ID', 'Status'}.issubset(df.columns):
        # Filtere ausgeschiedene Mitarbeiter
        terminated_df = df[df['Status'] == 'Ausgeschieden']

        # Gruppierung und Zählen der Mitarbeiter pro Jahr
        terminated_per_year = terminated_df.groupby('Jahr')['Mitarbeiter_ID'].nunique().reset_index()
        terminated_per_year.columns = ['Jahr', 'Ausgeschieden']

        # Ergebnisse anzeigen
        print("Jährliche Analyse (Anzahl Ausgeschiedene):")
        print(terminated_per_year)

        # Visualisierung
        plt.figure(figsize=(12, 6))
        plt.bar(terminated_per_year['Jahr'], terminated_per_year['Ausgeschieden'], color='red', alpha=0.7,
                label="Ausgeschieden")
        plt.title("Jährliche Anzahl ausgeschiedener Mitarbeiter")
        plt.xlabel("Jahr")
        plt.ylabel("Anzahl der ausgeschiedenen Mitarbeiter")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc="upper right")
        plt.savefig(save_path)
        plt.show()
    else:
        print("Die Spalten 'Jahr', 'Mitarbeiter_ID' und 'Status' fehlen im DataFrame.")

def yearly_active_trend(df, save_path):
    """Verlauf der aktiven Mitarbeiter pro Jahr."""
    if {'Jahr', 'Mitarbeiter_ID', 'Status'}.issubset(df.columns):
        # Filtere aktive Mitarbeiter
        active_df = df[df['Status'] == 'Aktiv']

        # Gruppierung und Zählen der Mitarbeiter pro Jahr
        active_per_year = active_df.groupby('Jahr')['Mitarbeiter_ID'].nunique().reset_index()
        active_per_year.columns = ['Jahr', 'Aktive']

        # Ergebnisse anzeigen
        print("Jährlicher Verlauf der Aktiven:")
        print(active_per_year)

        # Visualisierung
        plt.figure(figsize=(12, 6))
        plt.plot(active_per_year['Jahr'], active_per_year['Aktive'], marker='o', color='green', label="Aktive")
        plt.title("Verlauf der aktiven Mitarbeiter pro Jahr")
        plt.xlabel("Jahr")
        plt.ylabel("Anzahl der aktiven Mitarbeiter")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(loc="upper right")
        plt.savefig(save_path)
        plt.show()
    else:
        print("Die Spalten 'Jahr', 'Mitarbeiter_ID' und 'Status' fehlen im DataFrame.")

        # Hauptfunktion

def main():
    # Timer starten
    start_time = time.time()

    # Basisverzeichnis für Plots festlegen
    PLOTS_DIR = Path("Plots")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Daten laden
    data_file_path = Path("HR_cleaned.csv")
    if not data_file_path.exists():
        print(f"ERROR: Datei '{data_file_path}' existiert nicht.")
        exit(1)

    df = pd.read_csv(data_file_path)
    print("Bereinigter Datensatz erfolgreich geladen!")

    # Spalten analysieren
    numerische_spalten = df.select_dtypes(include=["float64", "int64"]).columns
    kategorische_spalten = df.select_dtypes(include=["object"]).columns
    print("Datenarten:")
    print(df.dtypes)
    print("\nFehlende Werte:")
    print(df.isnull().sum())

    # Visualisierung: Eindeutige IDs pro Jahr
    plot_combined_unique_employee_ids_by_year(df, "Geschlecht", "Jahr", PLOTS_DIR, "02_geschlecht_by_year.png")
    plot_combined_unique_employee_ids_by_year(df, "Job Level", "Jahr", PLOTS_DIR, "03_job_level_by_year.png")

    plot_skewness_kurtosis(df, numerische_spalten)

    plot_histograms(df, numerische_spalten, PLOTS_DIR, "03_histograms.png")

    relevante_spalten = [spalte for spalte in numerische_spalten if spalte not in [
        "Mitarbeiter_ID", "Zufriedenheit", "Fortbildungskosten",
        "Team Size", "Interne Weiterbildungen", "Urlaubstage_genommen",
        "Jährliche Leistungsbewertung", "Jahr", "Anzahl Untergebene", "Zeit bis zur Rente", "Wechselbereitschaft",
        "Monat"]]
    plot_scatterplots(df, relevante_spalten, PLOTS_DIR, "04_scatterplots.png")

    plot_correlation_heatmap(df, numerische_spalten, PLOTS_DIR, "05_correlation_heatmap.png")

    plot_kde(df, "Wechselbereitschaft", PLOTS_DIR, "06_kde_wechselbereitschaft.png")

    plot_wordcloud(df, "Job Role Progression", PLOTS_DIR, "07_wordcloud_job_role_progression.png")

    # 1. Daten vorbereiten
    df = prepare_data(df)
    print("Daten vorbereitet.")

    # 2. Statusänderungen filtern (nur relevante Kategorien behalten)
    df_filtered = filter_status_changes(df)
    print("Daten gefiltert (nur 'Aktiv', 'Ausgeschieden', 'Ruhestand').")

    # 3. Gruppieren und Daten zusammenfassen
    grouped_data = group_and_summarize_by_year(df_filtered)
    print("Zusammengefasste Daten nach Jahr und Kategorie:")
    print(grouped_data)

    # 4. Plots erstellen

    # a) Verlauf der aktiven/ausgeschiedenen Mitarbeiter
    print("Erstelle Verlauf der aktiven/ausgeschiedenen Mitarbeiter...")
    yearly_active_and_terminated_analysis(df_filtered, PLOTS_DIR / "aktive_und_ausgeschiedene_verlauf.png")

    # b) Plots: Eingestellte und ausgeschiedene Mitarbeiter
    print("Erstelle Plot für eingestellte Mitarbeiter pro Jahr...")
    yearly_hired_analysis(df, PLOTS_DIR / "eingestellt_pro_jahr.png")

    print("Erstelle Plot für ausgeschiedene Mitarbeiter pro Jahr...")
    yearly_terminated_analysis(df, PLOTS_DIR / "ausgeschieden_pro_jahr.png")

    print("Erstelle Verlauf der aktiven Mitarbeiter...")
    yearly_active_trend(df, PLOTS_DIR / "aktive_verlauf.png")

    # c) Kombinierter Plot (Aktiv, Ausgeschieden, Ruhestand)
    print("Erstelle kombinierten Plot für 'Aktiv', 'Ausgeschieden' und 'Ruhestand'...")
    plot_active_and_terminated(df_filtered, PLOTS_DIR)

    # 5. Boxplots erstellen
    variables = [ 'Gehalt', 'Überstunden', 'Wechselbereitschaft', 'Zufriedenheit']  # Beispielvariablen
    for var in variables:
        if var in df_filtered.columns:
            print(f"Erstelle Boxplot für {var}...")
            create_boxplot(df_filtered, var, PLOTS_DIR / f"boxplot_{var.lower()}.png")

    # 7. Analyse der Extremgruppen
    print("Analysiere Extremgruppen (hohe/niedrige Gehälter)...")
    analyze_extreme_groups(df_filtered)


    # 9. Erweiterte Cluster-Analyse und Visualisierungen
    cluster_features = ['Alter', 'Gehalt', 'Überstunden', 'Wechselbereitschaft', 'Zufriedenheit']  # Beispiel
    if all(feature in df_filtered.columns for feature in cluster_features):
        print("Führe erweiterte Cluster-Visualisierungen durch...")
        enhanced_cluster_visualizations(df_filtered, cluster_features, PLOTS_DIR)

    # Zeit stoppen
    end_time = time.time()
    print(f"Analyse abgeschlossen in {end_time - start_time:.2f} Sekunden.")

if __name__ == "__main__":
    main()
