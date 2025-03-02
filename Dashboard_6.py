import io
import seaborn as sns
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import dash
from dash import Input, Output, State, ctx, dcc, html, dash_table
import plotly.express as px
from data_loading import load_dataset
from data_cleaning import clean_dataset
from ML1_Fluctuation_best_model_6_ohne_pca import preprocess_data
from dash import dcc, html, Input, Output, State, ctx, dash_table

import os

from ML1_Fluctuation_best_model_6_ohne_pca import get_critical_employees
from model_for_dash import process_and_identify_critical_employees

# Beispielhafte Modellpfade
model_paths = {
    "Random Forest": "models/random_forest_model.pkl",
    "XGBoost": "models/xgboost_model.pkl",
    "LightGBM": "models/lightgbm_model.pkl"
}

try:
    df = pd.read_csv("HR_cleaned.csv", low_memory=False)
except FileNotFoundError:
    df = pd.DataFrame()  # Leerer DataFrame als Fallback


color_mapping = {
    "Aktiv": "#1f77b4",  # Blau
    "Ausgeschieden": "#2ca02c",  # Grün
    "Ruhestand": "#ff7f0e",  # Orange
}

# App-Konfiguration
# Tailwind CSS einbinden
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css",
]

# App-Konfiguration
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "HR Analytics Dashboard"


# Layout der App
app.layout = html.Div(
    className="container mx-auto my-4 px-4",
    children=[
        # Header
        html.Div(
            className="text-center text-blue-600 font-bold text-4xl mb-6",
            children="HR Analytics Dashboard"
        ),

        # Buttons direkt unter dem Header
        html.Div(
            className="flex justify-center space-x-4 mb-6",
            children=[
                html.Button("KPIs & Trends", id="btn-to-page-1",
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
                html.Button("Analysen & Details", id="btn-to-page-2",
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
                html.Button("Korrelationsmatrix", id="btn-to-page-3",  # Neuer Button für Korrelationsmatrix
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
                html.Button("Kritische Mitarbeiter", id="btn-to-page-4",
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
            ],
        ),

        # Platzhalter für dynamische Inhalte (Seiteninhalt)
        html.Div(id="page-content", className="mt-4"),

        # Footer mit fixer Position
        html.Div(
            className="text-center text-gray-500 text-sm mt-6",
            children="© 2025 Patrick Witzl",
            style={
                "position": "fixed",
                "width": "100%",
                "bottom": "0",
                "left": "0",
                "backgroundColor": "white",
                "padding": "10px 0",
                "boxShadow": "0 -2px 5px rgba(0, 0, 0, 0.1)",  # Schatten über Inhalt
                "zIndex": "1000"  # Damit der Footer immer oben bleibt, falls Inhalte scrollen
            }
        )
    ]
)


@app.callback(
    Output("page-content", "children"),
    [Input("btn-to-page-1", "n_clicks"),
     Input("btn-to-page-2", "n_clicks"),
     Input("btn-to-page-3", "n_clicks"),
     Input("btn-to-page-4", "n_clicks"),
    ]
)
def navigate(button_1_clicks, button_2_clicks, button_3_clicks, button_4_clicks,):
    ctx = dash.callback_context
    if not ctx.triggered:
        return render_page_1()  # Standardseite: Seite 1
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "btn-to-page-1":
        return render_page_1()
    elif button_id == "btn-to-page-2":
        return render_page_2()
    elif button_id == "btn-to-page-3":
        return render_page_3_correlation_matrix()
    elif button_id == "btn-to-page-4":
        return render_page_4()

    return render_page_1()

# Inhalte der ersten Seite (KPIs & Trends)
def render_page_1():
    return html.Div(
        className="space-y-6",
        children=[

            # --- Dropdown zur Jahresauswahl ---
            html.Div(
                className="w-1/2 mx-auto",
                children=[
                    dcc.Dropdown(
                        id="year-dropdown",
                        options=[{"label": str(year), "value": year} for year in sorted(df["Jahr"].unique())],
                        placeholder="Jahr auswählen",
                        value=sorted(df["Jahr"].unique())[-1] if len(df["Jahr"].unique()) > 0 else None,
                        className="rounded border-gray-300"
                    )
                ],
            ),

            # --- Titelbereich ---
            html.H2("KPIs & Trends", className="text-center text-blue-600 font-bold text-2xl"),

            # --- KPI-Titel (z. B. Gehalt, Zufriedenheit, Krankheitstage) ---
            html.Div(
                className="grid grid-cols-3 gap-4",
                children=[
                    html.H3(id="kpi-title-salary", className="text-center text-lg text-gray-600"),
                    html.H3(id="kpi-title-satisfaction", className="text-center text-lg text-gray-600"),
                    html.H3(id="kpi-title-absence", className="text-center text-lg text-gray-600"),
                ]
            ),

            # --- KPI-Werte (durchschnittliches Gehalt, Zufriedenheit, Krankheitstage) ---
            html.Div(
                className="grid grid-cols-3 gap-4",
                children=[
                    html.H3(id="kpi-avg-salary", className="text-center text-lg text-gray-600"),
                    html.H3(id="kpi-avg-satisfaction", className="text-center text-lg text-gray-600"),
                    html.H3(id="kpi-avg-absence", className="text-center text-lg text-gray-600"),
                ]
            ),

            # --- Summen oben auf der Seite ---
            html.Div(
                className="grid grid-cols-3 gap-4",
                children=[
                    # Eingestellte
                    html.Div([
                        html.H3(id="sum-new-hires", style={"color": "#1f77b4"},  # Blau
                                className="text-center text-lg font-bold"),
                        html.P("Eingestellte", className="text-center text-gray-600")
                    ]),
                    # Ausgeschiedene
                    html.Div([
                        html.H3(id="sum-exited", style={"color": "#2ca02c"},  # Rot
                                className="text-center text-lg font-bold"),
                        html.P("Ausgeschiedene", className="text-center text-gray-600")
                    ]),
                    # Ruhestand
                    html.Div([
                        html.H3(id="sum-retired", style={"color": "#ff7f0e"},  # Orange
                                className="text-center text-lg font-bold"),
                        html.P("In Ruhestand", className="text-center text-gray-600")
                    ]),
                ],
            ),

            # --- Monatliche Trends ---
            html.Div(
                className="grid grid-cols-2 gap-6",
                children=[
                    html.Div([
                        html.H3("Monatlicher Trend: Aktiv", className="text-center text-gray-800"),
                        dcc.Graph(id="active-monthly-trend-line")
                    ]),
                    html.Div([
                        html.H3("Monatlicher Trend: Ausgeschieden und Ruhestand",
                                className="text-center text-gray-800"),
                        dcc.Graph(id="retired-exited-monthly-trend-line")
                    ]),
                ]
            ),

            # --- Jährliche Trends ---
            html.Div(
                className="grid grid-cols-2 gap-6",
                children=[
                    html.Div([
                        html.H3("Jährlicher Trend: Aktiv", className="text-center text-gray-800"),
                        dcc.Graph(id="active-trend-line")
                    ]),
                    html.Div([
                        html.H3("Jährlicher Trend: Ausgeschieden und Ruhestand",
                                className="text-center text-gray-800"),
                        dcc.Graph(id="retired-exited-trend-line")
                    ]),
                ]
            ),
        ]
    )

# Inhalte der zweiten Seite (Analysen & Details)
def render_page_2():
    scatter_fig = px.scatter(df, x="Gehalt", y="Zufriedenheit", color="Status",
                             title="Zufriedenheit vs. Gehalt nach Status", color_discrete_map=color_mapping
)
    return html.Div([
        html.H2("Analysen & Details", className="text-center text-blue-600 font-bold text-2xl mb-6"),

        # Streudiagramm: Zufriedenheit vs Gehalt
        dcc.Graph(figure=scatter_fig, className="rounded shadow-lg"),

        # Dropdown für interaktive Analyse
        html.Div(
            className="mt-6",
            children=[
                html.H3("Interaktive Analyse", className="text-center text-gray-700 text-lg"),
                dcc.Dropdown(
                    id="analysis-dropdown",
                    options=[
                        {"label": "Alter", "value": "Alter"},
                        {"label": "Gehalt", "value": "Gehalt"},
                    ],
                    value="Alter",
                ),
                dcc.Graph(id="interactive-plot"),
            ]
        )
    ])

def render_page_3_correlation_matrix():
    # Nur numerische Spalten beibehalten
    numeric_df = df.select_dtypes(include=["number"])  # Nur numerische Datentypen
    if numeric_df.empty:
        return html.Div([
            html.H2("Korrelationsmatrix", className="text-center text-blue-600 font-bold text-2xl mt-6 mb-6"),
            html.P("Der Datensatz enthält keine numerischen Spalten, die korreliert werden können.",
                   className="text-center text-gray-600 text-lg"),
        ])

    # Berechnung der Korrelationsmatrix
    correlation_matrix = numeric_df.corr()

    # Entferne die Hauptdiagonale durch Maskierung
    mask = np.eye(len(correlation_matrix), dtype=bool)  # Erstelle eine Maske für die Hauptdiagonale
    correlation_matrix_no_diag = correlation_matrix.mask(mask)  # Setze die Diagonale auf NaN

    # Hochauflösende Heatmap mit Seaborn
    plt.figure(figsize=(20, 20), dpi=300)  # Große Grafik und hohe Auflösung
    heatmap = sns.heatmap(
        correlation_matrix_no_diag,  # Verwendet die modifizierte Matrix
        annot=True,  # Werte im Diagramm anzeigen
        fmt=".2f",  # Zahlenformat
        cmap="coolwarm",  # Farbschema
        square=True,
        cbar_kws={"shrink": 0.6},  # Farbelement-Leiste verkleinern
        linewidths=0.5,  # Linien zwischen Zellen
        mask=mask  # Diagonale ausblenden
    )
    plt.title("Korrelationsmatrix ohne Hauptdiagonale", fontsize=20, pad=30)
    plt.xticks(rotation=45, fontsize=12, ha="right")
    plt.yticks(fontsize=12)

    # Matplotlib-Plot als Base64-Bild kodieren
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")  # Ränder abschneiden
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    # Matplotlib Bild in Dash einbetten
    return html.Div([
        html.H2("Korrelationsmatrix ohne Hauptdiagonale",
                className="text-center text-blue-600 font-bold text-2xl mt-4 mb-4"),
        html.Img(src=f"data:image/png;base64,{encoded_image}", style={"width": "100%", "height": "auto"}),
        # Bild skalieren
    ]
)

def render_page_4():
    return html.Div(
        children=[
            html.H2(
                "Datenupload & Aktualisierung",
                className="text-center text-blue-600 font-bold text-2xl mb-6",
            ),
            html.Div(
                className="mt-4 mb-6",
                children=[
                    html.H3("CSV-Datei hochladen:", className="text-lg text-gray-700"),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Datei per Drag & Drop hochladen oder ", html.A("Datei auswählen")]
                        ),
                        style={
                            "width": "100%",
                            "height": "60px",
                            "lineHeight": "60px",
                            "borderWidth": "1px",
                            "borderStyle": "dashed",
                            "borderRadius": "5px",
                            "textAlign": "center",
                            "margin": "10px",
                        },
                        multiple=False,
                    ),
                    html.Div(id="upload-feedbacks", className="mt-4 text-lg text-gray-700"),
                    html.H3("Modell auswählen:", className="mt-6 text-lg text-gray-700"),
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[
                            {"label": "XGBoost", "value": "XGBoost"},
                        ],
                        placeholder="Wähle ein Modell",
                        className="mt-2",
                    ),
                    html.H3("Schwellenwert für Fluktuationswahrscheinlichkeit (%):",
                            className="mt-6 text-lg text-gray-700"),
                    dcc.Input(
                        id="threshold-input",
                        type="number",
                        value=10,
                        placeholder="Schwellenwert eingeben",
                        className="mt-2",
                    ),
                    html.Div(id="upload-status", className="mt-4 text-lg text-gray-700"),
                    html.Div(id="critical-employees-table", className="mt-8", style={"marginBottom": "40px"}
                    ),
                    html.Div(
                        children=[
                            html.H3(
                                "Hinweis: Wenn keine Datei hochgeladen wird, wird die Standarddatei verwendet.",
                                className="text-md text-gray-500",
                            )
                        ]
                    ),
                ],
            ),
        ],
    )


@app.callback(
    Output("upload-feedback", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_dataset(content, filename):
    global df  # Zugriff auf die globale Variable für den DataFrame

    if content is not None:
        # Inhalte dekodieren und in einen Pandas-DataFrame umwandeln
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)

        try:
            # Prüfen, ob die hochgeladene Datei eine CSV-Datei ist
            if filename.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), low_memory=False)
                return f"Die Datei '{filename}' wurde erfolgreich hochgeladen!"
            else:
                return "Bitte nur CSV-Dateien hochladen."
        except Exception as e:
            return f"Fehler beim Einlesen der Datei: {str(e)}"
    else:
        return "Keine Datei hochgeladen."

@app.callback(
    [Output("kpi-avg-salary", "children"),
     Output("kpi-avg-satisfaction", "children"),
     Output("kpi-avg-absence", "children")],
    Input("year-dropdown", "value")
)
def update_kpis(selected_year):
    if selected_year is None or "Jahr" not in df.columns:
        return "Keine Daten", "Keine Daten", "Keine Daten"

    filtered_data = df[df["Jahr"] == selected_year]
    if filtered_data.empty:
        return "Keine Daten", "Keine Daten", "Keine Daten"

    avg_salary = filtered_data["Gehalt"].mean()
    avg_satisfaction = filtered_data["Zufriedenheit"].mean()
    avg_absence = filtered_data["Fehlzeiten_Krankheitstage"].mean()

    return (
        f"{avg_salary:.2f} €" if not pd.isna(avg_salary) else "Keine Daten",
        f"{avg_satisfaction:.2f} / 10" if not pd.isna(avg_satisfaction) else "Keine Daten",
        f"{avg_absence:.2f} Tage" if not pd.isna(avg_absence) else "Keine Daten"
    )

@app.callback(
    [Output("kpi-title-salary", "children"),
     Output("kpi-title-satisfaction", "children"),
     Output("kpi-title-absence", "children")],
    Input("year-dropdown", "value")
)
def update_kpi_titles(selected_year):
    if selected_year is None:
        return "Gehalt (Keine Daten)", "Zufriedenheit (Keine Daten)", "Krankheitstage (Keine Daten)"
    return (
        f"Gehalt ({selected_year})",
        f"Zufriedenheit ({selected_year})",
        f"Krankheitstage ({selected_year})"
    )

@app.callback(
    [Output("sum-new-hires", "children"),  # Neueinstellungen
     Output("sum-exited", "children"),  # Ausgeschiedene
     Output("sum-retired", "children")],  # Ruheständler
    Input("year-dropdown", "value")
)
def update_status_sums(selected_year):
    # Sicherstellen, dass alle erforderlichen Spalten vorhanden sind
    required_columns = ["Einstellungsdatum", "Austrittsdatum", "Status", "Monat", "Mitarbeiter_ID"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Die Spalte '{col}' fehlt im Datensatz.")

    # Konvertiere 'Einstellungsdatum' und 'Austrittsdatum' in datetime-Format
    df["Einstellungsdatum"] = pd.to_datetime(df["Einstellungsdatum"], errors="coerce", format="%Y-%m-%d")
    df["Austrittsdatum"] = pd.to_datetime(df["Austrittsdatum"], errors="coerce", format="%Y-%m-%d")

    # Wenn kein Jahr ausgewählt wurde, gebe Standardwerte zurück
    if not selected_year:
        return "kein Jahr gewählt", "kein Jahr gewählt", "kein Jahr gewählt"

    # FILTER: Neueinstellungen
    new_hires_df = df[
        (df["Einstellungsdatum"].notna()) &  # Einstellungsdatum darf nicht leer sein
        (df["Einstellungsdatum"].dt.year == selected_year)  # Einstellungsjahr muss dem ausgewählten Jahr entsprechen
        ]
    num_new_hires = new_hires_df["Mitarbeiter_ID"].nunique()  # Eindeutige IDs zählen

    # FILTER: Ausgeschiedene (entspricht Filterung in Code 2)
    exited_df = df[
        (df["Status"] == "Ausgeschieden") &  # Status = Ausgeschieden
        (df["Austrittsdatum"].notna()) &  # Austrittsdatum darf nicht leer sein
        (df["Austrittsdatum"].dt.year == selected_year)  # Austrittsjahr muss dem ausgewählten Jahr entsprechen
        ]
    num_exits = exited_df["Mitarbeiter_ID"].nunique()  # Eindeutige IDs zählen

    # FILTER: Ruhestand (entspricht Filterung in Code 2)
    retired_df = df[
        (df["Status"] == "Ruhestand") &  # Status = Ruhestand
        (df["Austrittsdatum"].notna()) &  # Austrittsdatum darf nicht leer sein
        (df["Austrittsdatum"].dt.year == selected_year)  # Austrittsjahr muss dem ausgewählten Jahr entsprechen
        ]
    num_retired = retired_df["Mitarbeiter_ID"].nunique()  # Eindeutige IDs zählen

    # Rückgabe der berechneten Ergebnisse
    return (
        f"{num_new_hires} Personen",  # Neueinstellungen
        f"{num_exits} Personen",  # Ausgeschiedene
        f"{num_retired} Personen"  # Ruhestand
    )

@app.callback(
    Output("active-monthly-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_active_monthly_trend(selected_year):
    filtered_data = df[(df["Status"] == "Aktiv") & (df["Jahr"] == selected_year)]
    filtered_data = filtered_data.groupby("Monat").size().reset_index(name="Anzahl")
    fig = px.line(
        filtered_data,
        x="Monat",
        y="Anzahl",
        title="Trend Aktiv",
        color_discrete_map=color_mapping,
        markers=True
    )
    return fig

@app.callback(
    Output("retired-exited-monthly-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_retired_exited_monthly_trend(selected_year):
    filtered_data = df[(df["Status"].isin(["Ausgeschieden", "Ruhestand"])) & (df["Jahr"] == selected_year)]
    filtered_data = filtered_data.groupby(["Monat", "Status"]).size().reset_index(name="Anzahl")
    fig = px.line(filtered_data, x="Monat", y="Anzahl", color="Status", title="Trend Ruhestand/Ausgeschieden", color_discrete_map=color_mapping,
    markers=True

)
    return fig

@app.callback(
    Output("active-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_active_trend(selected_year):
    # Filtern für aktive Mitarbeiter
    filtered_data = df[df["Status"] == "Aktiv"]

    # Nur eindeutige Kombinationen von Mitarbeiter_ID und Jahr
    unique_data = filtered_data.drop_duplicates(subset=["Mitarbeiter_ID", "Jahr"])

    # Gruppieren nach Jahr und zählen der eindeutigen Einträge
    trend_data = unique_data.groupby("Jahr").size().reset_index(name="Anzahl")

    # Visualisierung erstellen
    fig = px.line(trend_data, x="Jahr", y="Anzahl", title="Aktiv - Jährlicher Trend",
                  color_discrete_map={"Aktiv": "#1f77b4"},
        markers=True
)
    return fig

@app.callback(
    Output("retired-exited-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_retired_exited_trend(selected_year):
    filtered_data = df[df["Status"].isin(["Ausgeschieden", "Ruhestand"])]
    filtered_data = filtered_data.groupby(["Jahr", "Status"]).size().reset_index(name="Anzahl")
    fig = px.line(filtered_data, x="Jahr", y="Anzahl", color="Status", title="Jährliche Trends (ausgeschieden)", color_discrete_map=color_mapping,
    markers=True

)
    return fig

@app.callback(
    Output("interactive-plot", "figure"),  # Ausgabe für die interaktive Grafik
    Input("analysis-dropdown", "value")  # Eingabe vom Dropdown
)
def update_interactive_plot(selected_feature):
    # Sicherstellen, dass die Auswahl valide ist
    if selected_feature not in df.columns:
        return px.scatter(title="Keine gültigen Daten verfügbar")

    # Streudiagramm basierend auf der Auswahl im Dropdown
    fig = px.histogram(
        df,
        x=selected_feature,  # X-Achse basierend auf der ausgewählten Spalte
        title=f"Häufigkeitsverteilung: {selected_feature}",  # Titel der Grafik
        color="Status",  # Farbgebung nach 'Status'
        barmode="group",  # Gruppierte Darstellung der Balken
        histnorm="percent"  # Prozentuale Darstellung (optional)
    )
    return fig

# Callback für Datei-Upload und Verarbeitung
@app.callback(
    [
        Output("upload-feedbacks", "children"),
        Output("critical-employees-table", "children"),
        Output("upload-status", "children"),
    ],
    [
        Input("upload-data", "contents"),
        Input("model-dropdown", "value"),
        Input("threshold-input", "value"),
    ],
    [State("upload-data", "filename")],
)
def process_and_display_critical_employees(contents, selected_model, threshold, filename):
    if not contents:
        return "Bitte eine Datei hochladen.", None, "Keine Datei hochgeladen."

    if not selected_model:
        return "Bitte ein Modell auswählen.", None, "Kein Modell ausgewählt."

    try:
        # ** Schritt 1: Datei laden **
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        uploaded_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        # Pfad für temporäre Speicherung
        temp_path = "uploaded_temp.csv"
        uploaded_df.to_csv(temp_path, index=False)  # Datei temporär speichern

        file_path = temp_path
        critical_employees, top_15_employees = process_and_identify_critical_employees(
            file_path,  # Eingabedatei
            save_filtered_path= None,  # Speicherort für gefilterte Daten
            models_dir= "Models",  # Verzeichnis, in dem das Modell gespeichert ist
            model_file_name="xgboost_model.pkl",  # Name der Modell-Datei
            feature_names_file="Models/lightgbm_feature_names.pkl",  # Datei mit Feature-Namen
            threshold=0.0  # Schwellenwert für kritische Mitarbeiter
        )  # Schwellenwert für kritische Mitarbeiter

        if critical_employees is None or critical_employees.empty:
            return "Es wurden keine kritischen Mitarbeiter gefunden.", None, "Keine kritischen Mitarbeiter."

        # Columns für die Anzeige festlegen
        columns_to_display = [
            'Jahr', 'Monat', 'Mitarbeiter_ID', 'Name', 'Position',
            'Alter', 'Status', 'Fluktuationswahrscheinlichkeit'
        ]

        # DataFrame für die Anzeige filtern
        filtered_top_15_employees = top_15_employees[columns_to_display]

        # Erstellen der DataTable mit den gefilterten Spalten
        table = dash_table.DataTable(
            id="critical-employees-results",
            columns=[{"name": col, "id": col} for col in filtered_top_15_employees.columns],  # Nur gefilterte Spalten
            data=filtered_top_15_employees.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "5px"},
            style_header={"backgroundColor": "lightblue", "fontWeight": "bold"},
            page_size=15,  # Maximale Zeilenanzahl pro Seite
        )

        return f"Datei '{filename}' erfolgreich verarbeitet.", table, f"{len(critical_employees)} kritische Mitarbeiter gefunden."


    except Exception as e:
        return f"Ein Fehler ist aufgetreten: {str(e)}", None, "Fehler bei der Verarbeitung."

if __name__ == "__main__":
    app.run(debug=True)
