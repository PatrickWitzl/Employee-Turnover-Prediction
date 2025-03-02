import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px

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


def create_correlation_heatmap(df_filtered, save_path):
    """
    Korrelations-Heatmap für numerische Spalten erstellen, wobei nur die Diagonale maskiert wird.
    """
    # Numerische Spalten auswählen
    numerische_spalten = df_filtered.select_dtypes(include=['float64', 'int64']).columns
    correlations = df_filtered[numerische_spalten].corr()

    # Maske für die Hauptdiagonale erstellen
    mask = np.eye(correlations.shape[0], dtype=bool)  # Nur Hauptdiagonale maskieren

    # Heatmap zeichnen
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlations,
        mask=mask,  # Maske anwenden, um Hauptdiagonale auszublenden
        annot=True,  # Korrelationswerte anzeigen
        cmap="coolwarm",  # Farbpalette
        fmt=".2f",  # Formatierung der Werte
        cbar=True  # Farbskala anzeigen
    )
    plt.title("Korrelationen von numerischen Variablen (ohne Diagonale)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return correlations


def analyze_extreme_groups(df_filtered):
    """Analyse der Extremgruppen (hohe/niedrige Gehälter)."""
    extreme_high = df_filtered[df_filtered['Gehalt'] > df_filtered['Gehalt'].quantile(0.95)]
    extreme_low = df_filtered[df_filtered['Gehalt'] < df_filtered['Gehalt'].quantile(0.05)]

    print("\nExtremgruppen: Hohe Gehälter:")
    print(extreme_high.describe())
    print("\nExtremgruppen: Niedrige Gehälter:")
    print(extreme_low.describe())


#def cluster_analysis(df_filtered, cluster_features, save_dir):
    #"""Cluster-Analyse durchführen mit kombinierter Boxplot-Darstellung."""
    #df = df_filtered.dropna(subset=cluster_features)
    #if df.empty:
    #    print("Fehler: Keine geeigneten Daten für die Cluster-Analyse verfügbar.")
    #    return

    # Standardisierung der Features
    #scaler = StandardScaler()
    #scaled_data = scaler.fit_transform(df[cluster_features])

    # K-Means-Clustering
    #kmeans = KMeans(n_clusters=3, random_state=42)
    #df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Cluster-Mittelwerte anzeigen
    #cluster_means = df.groupby('Cluster')[cluster_features].mean()
    #print("\nCluster-Mittelwerte:\n", cluster_means)
    #
    ## Cluster-Scatterplot (Alter vs. Gehalt)
    #plt.figure(figsize=(10, 6))
    #sns.scatterplot(data=df, x='Alter', y='Gehalt', hue='Cluster', palette="Set1")
    #plt.title("Cluster-Analyse: Alter vs Gehalt")
    #plt.xlabel("Alter")
    #plt.ylabel("Gehalt")
    #plt.legend(title="Cluster")
    #plt.savefig(save_dir / "cluster_alter_gehalt.png")
    #plt.show()

    # Kombinierte Boxplots für Cluster-Features
    #plt.figure(figsize=(14, 8))
    #df_long = df.melt(id_vars='Cluster', value_vars=cluster_features,
    #                  var_name="Feature", value_name="Wert")
    #sns.boxplot(
    #    data=df_long,
    #    x='Feature',
    #    y='Wert',
    #    hue='Cluster',
    #    palette="Set2"
    #)
    #plt.title("Kombinierter Boxplot: Verteilung der Features nach Cluster")
    #plt.xlabel("Feature")
    #plt.ylabel("Wert")
    #plt.legend(title="Cluster")
    #plt.savefig(save_dir / "combined_boxplot.png")
    #plt.show()


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

    # 1. Scatterplot mit Centroiden
    #centroids = scaler.inverse_transform(kmeans.cluster_centers_)  # Zentroiden zurücktransformieren
    #plt.figure(figsize=(10, 6))
    #sns.scatterplot(data=df, x='Alter', y='Gehalt', hue='Cluster', palette="Set1", s=50)
    #plt.scatter(centroids[:, cluster_features.index('Alter')],
    #            centroids[:, cluster_features.index('Gehalt')],
    #            s=200, c='black', marker='X', label='Centroid')
    #plt.title("Cluster-Analyse: Alter vs. Gehalt mit Centroiden", fontsize=16)
    #plt.xlabel("Alter", fontsize=14)
    #plt.ylabel("Gehalt", fontsize=14)
    #plt.legend(title="Cluster", fontsize=12)
    #plt.tight_layout()
    #plt.savefig(save_dir / "cluster_alter_gehalt_centroid.png", bbox_inches="tight")
    #plt.show()

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

    # 5. Silhouette-Analyse
    silhouette_vals = silhouette_samples(scaled_data, df['Cluster'])
    y_ticks = []
    y_lower, y_upper = 0, 0

    plt.figure(figsize=(10, 6))
    for cluster in range(kmeans.n_clusters):
        cluster_silhouette_vals = np.sort(silhouette_vals[df['Cluster'] == cluster])
        y_upper += len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    avg_silhouette = np.mean(silhouette_vals)
    plt.axvline(avg_silhouette, color="red", linestyle="--", label="Durchschnittlicher Silhouette-Wert")
    plt.yticks(y_ticks, [f"Cluster {i}" for i in range(kmeans.n_clusters)], fontsize=12)
    plt.ylabel("Cluster", fontsize=14)
    plt.xlabel("Silhouette-Koeffizient", fontsize=14)
    plt.title("Silhouette-Plot der Cluster", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "silhouette_plot.png", bbox_inches="tight")
    plt.show()

    # 6. Interaktiver Scatterplot
    fig = px.scatter(
        df, x='Alter', y='Gehalt', color='Cluster',
        title="Cluster-Analyse: Alter vs. Gehalt (interaktiv)",
        labels={'Cluster': 'Cluster'},
        hover_data=cluster_features
    )
    fig.write_html(str(save_dir / "cluster_interactive.html"))
    fig.show()

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
        plt.legend()
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
        plt.legend()
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
        plt.legend()
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
        plt.legend()
        plt.savefig(save_path)
        plt.show()
    else:
        print("Die Spalten 'Jahr', 'Mitarbeiter_ID' und 'Status' fehlen im DataFrame.")

def main():
    import time
    import pandas as pd
    from pathlib import Path

    # Timer starten
    start_time = time.time()

    # Verzeichnis für Plots erstellen
    PLOTS_DIR = Path("Plots")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Datei laden
    file_path = "../data/HR_cleaned.csv"
    if not Path(file_path).exists():
        print(f"ERROR: Datei '{file_path}' existiert nicht.")
        return

    print(f"Lade Datei '{file_path}'...")
    df = pd.read_csv(file_path)

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

    # 6. Korrelations-Heatmap
    print("Erstelle Korrelations-Heatmap...")
    correlations = create_correlation_heatmap(df_filtered, PLOTS_DIR / "heatmap_korrelationen.png")
    print("Korrelationsmatrix:")
    print(correlations)

    # 7. Analyse der Extremgruppen
    print("Analysiere Extremgruppen (hohe/niedrige Gehälter)...")
    analyze_extreme_groups(df_filtered)

    # 8. Cluster-Analyse
    #cluster_features = ['Alter', 'Gehalt', 'Überstunden', 'Wechselbereitschaft', 'Zufriedenheit']  # Beispiel
    #if all(feature in df_filtered.columns for feature in cluster_features):
        #print("Führe Cluster-Analyse durch...")
        #cluster_analysis(df_filtered, cluster_features, PLOTS_DIR)

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