import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import joblib
from ML1_Fluctuation_best_model_5 import get_critical_employees_all_models
import xgboost as xgb
import os
import sys
import time
import select
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    log_loss,
    confusion_matrix
)
# Wichtige Standard-Imorts
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ML1_Fluctuation_best_model_5 import preprocess_data, split_and_scale, train_xgboost
import xgboost as xgb
import joblib  # Für das Laden des Scalers
import dash
from dash import Input, Output, State, ctx, dcc, html, dash_table
import plotly.express as px

# Schritt 1: Lade transformierte Daten
df_transformed = pd.read_csv("X_transformed.csv")

# Schritt 2: Scaler laden und Features validieren
scaler = joblib.load("Models/scaler.pkl")
expected_features = scaler.feature_names_in_

# Überprüfen, ob alle erwarteten Features vorhanden sind
for feature in expected_features:
    if feature not in df_transformed.columns:
        print(f"Feature '{feature}' fehlt. Ergänze es mit Dummy-Wert.")
        df_transformed[feature] = 0

# Sortiere die Features in der erwarteten Reihenfolge
df_transformed = df_transformed[expected_features]

# Sicherstellen, dass die Spalten numerisch sind
df_transformed = df_transformed.astype(float)

# Schritt 3: Transformierte Daten in eine DMatrix umwandeln
dmatrix = xgb.DMatrix(df_transformed)

# Debug-Ausgabe: Überprüfe die DMatrix
print(f"DMatrix erstellt mit {dmatrix.num_col()} Spalten und {dmatrix.num_row()} Zeilen.")

# Schritt 4: XGBoost-Modell laden und Vorhersage durchführen
xgb_model = xgb.Booster()
xgb_model.load_model("Models/xgboost_model.pkl")
print("XGBoost-Modell erfolgreich aus JSON geladen.")

# Vorhersagen durchführen
if dmatrix.num_col() == 0:
    raise ValueError("Keine gültigen Features in der DMatrix! Überprüfe die transformierten Daten.")
else:
    predictions = xgb_model.predict(dmatrix)
    print("Vorhersagen erfolgreich:", predictions)

# Optional: Transformierte Daten speichern (nur falls nötig)
output_path = "X_transformed_updated.csv"
df_transformed.to_csv(output_path, index=False)
print(f"Die aktualisierten transformierten Daten wurden unter '{output_path}' gespeichert.")

# Schritt 5: Modellvorhersagen durchführen
if dmatrix.num_col() == 0:
    raise ValueError("Keine gültigen Features in der DMatrix! Überprüfe die transformierten Daten.")
else:
    predictions = xgb_model.predict(dmatrix)
    print(f"Vorhersagen erfolgreich: {predictions}")


# Schritt 6: Modellvorhersage durchführen (falls relevant)
predictions = xgb_model.predict(xgb.DMatrix(df_transformed))
print("Vorhersagen:", predictions)

# Schritt 7: Features sortieren basierend auf den erwarteten Feature-Namen
df_transformed = df_transformed[expected_features]  # Kein erneutes "X_transformed"

# Schritt 8: Transformierte Daten speichern
output_path = "X_transformed.csv"
df_transformed.to_csv(output_path, index=False)
print(f"X_transformed.csv wurde erfolgreich unter '{output_path}' gespeichert.")


def get_critical_employees(model, X_transformed, df, scaler_file="Models/scaler.pkl", threshold=0.0):
    """
    Identifiziert kritische Mitarbeiter mit einer Fluktuationswahrscheinlichkeit höher
    als der angegebenen Schwelle unter Verwendung eines gespeicherten Scalers.
    Unterstützt XGBoost-Booster und andere sklearn-ähnliche Modelle.

    Args:
        model: Das trainierte Modell für die Vorhersage (z. B. ein XGBoost-Modell).
        X_transformed (np.ndarray): Die preprocessierten Features (unskaliert).
        df (pd.DataFrame): Der DataFrame mit zusätzlichen Informationen (z. B. Mitarbeiterdaten).
        scaler_file (str): Dateiname des gespeicherten Scalers.
        threshold (float): Der Schwellenwert für die Fluktuationswahrscheinlichkeit.

    Returns:
        dict: Ein Dictionary mit den folgenden Ergebnissen:
              - "critical_employees" (pd.DataFrame): Alle kritischen Mitarbeiter.
              - "top_15" (pd.DataFrame): Top 15 Mitarbeiter mit höchster Wahrscheinlichkeit.
              - "errors" (list(str)): Liste von Fehlermeldungen, falls aufgetreten.
    """

    errors = []

    # ** Debugging Input-Daten **
    try:
        print(f"Shape von X_transformed: {X_transformed.shape}")
        print(f"Shape von df: {df.shape}")
    except AttributeError as e:
        errors.append(f"Fehler bei Datenformaten: {e}")
        return {"critical_employees": None, "top_15": None, "errors": errors}

    # Dynamisch letzten Monat und Jahr bestimmen
    try:
        max_year = df["Jahr"].max()
        max_month_in_max_year = df[df["Jahr"] == max_year]["Monat"].max()
        print(f"Letztes Jahr: {max_year}, letzter Monat: {max_month_in_max_year}")
    except Exception as e:
        errors.append(f"Fehler beim Ermitteln von Monat/Jahr: {e}")
        return {"critical_employees": None, "top_15": None, "errors": errors}

    # Lade und nutze den Scaler
    try:
        scaler = joblib.load(scaler_file)
        X_transformed = scaler.transform(X_transformed)
        print("Scaler erfolgreich angewendet.")
    except Exception as e:
        errors.append(f"Fehler beim Laden des Scalers '{scaler_file}': {e}")
        return {"critical_employees": None, "top_15": None, "errors": errors}

    # Vorhersagen berechnen
    try:
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(X_transformed)
            y_proba = model.predict(dmatrix)
        elif hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_transformed)[:, 1]
        elif hasattr(model, "predict"):
            y_proba = model.predict(X_transformed).flatten()
        else:
            raise AttributeError(f"Modelltyp '{type(model)}' wird nicht unterstützt.")
    except Exception as e:
        errors.append(f"Fehler bei Modell-Vorhersagen: {e}")
        return {"critical_employees": None, "top_15": None, "errors": errors}

    # Ergebnisse verknüpfen
    try:
        df["Fluktuationswahrscheinlichkeit"] = y_proba * 100
    except ValueError as e:
        errors.append(f"Fehler beim Hinzufügen der Fluktuationswahrscheinlichkeit: {e}")
        return {"critical_employees": None, "top_15": None, "errors": errors}

    # Kritische Mitarbeiter identifizieren
    try:
        df_filtered = df[
            (df["Fluktuation"] == 0) &
            (df["Monat"] == max_month_in_max_year) &
            (df["Jahr"] == max_year)
            ]

        print(f"Shape nach Fluktuationsfilter: {df_filtered.shape}")

        critical_employees = df_filtered[df_filtered["Fluktuationswahrscheinlichkeit"] > threshold]
        critical_employees = critical_employees.sort_values(by="Fluktuationswahrscheinlichkeit", ascending=False)
        top_15 = critical_employees.head(15)

        if "Mitarbeiter_ID" in critical_employees.columns:
            critical_employees = critical_employees.drop_duplicates(subset=["Mitarbeiter_ID"])

    except Exception as e:
        errors.append(f"Fehler beim Filtern der Mitarbeiterdaten: {e}")
        return {"critical_employees": None, "top_15": None, "errors": errors}

    # Finaler Output
    return {
        "critical_employees": critical_employees,
        "top_15": top_15,
        "errors": errors
    }


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
                html.Button("Korrelationsmatrix", id="btn-to-page-4",  # Neuer Button für Korrelationsmatrix
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
                html.Button("Kritische Mitarbeiter", id="btn-to-page-5",
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
                html.Button("Datenupload", id="btn-to-page-3",
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
     Input("btn-to-page-5", "n_clicks")]  # Hinzugefügter Button für Seite 5
)
def navigate(button_1_clicks, button_2_clicks, button_3_clicks, button_4_clicks, button_5_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return render_page_1()  # Standardseite: Seite 1
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "btn-to-page-1":
        return render_page_1()
    elif button_id == "btn-to-page-2":
        return render_page_2()
    elif button_id == "btn-to-page-3":
        return render_page_3()
    elif button_id == "btn-to-page-4":
        return render_page_correlation_matrix()
    elif button_id == "btn-to-page-5":
        return render_page_critical_employees()

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

# Hinzufügen der dritten Seite (Render-Methode)
def render_page_3():
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
                        multiple=False,  # Keine Mehrfachuploads zulassen
                    ),
                    html.Div(
                        id="upload-feedback", className="mt-4 text-lg text-gray-700"
                    ),
                ],
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
    )

def render_page_critical_employees():
    """Seite für die Anzeige der kritischen Mitarbeiter."""
    return html.Div(
        className="space-y-6",
        children=[
            html.H2("Kritische Mitarbeiter", className="text-center text-blue-600 font-bold text-2xl mb-6"),

            # Dropdown-Menü zur Auswahl eines Modells
            html.Div(
                className="w-1/2 mx-auto",
                children=[
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[
                            {"label": "XGBoost", "value": "xgboost"},
                        ],
                        value="xgboost",  # Standardauswahl
                        placeholder="Wähle ein Modell",
                        className="rounded border-gray-300"
                    )
                ],
            ),

            # Eingabefeld für Schwellenwert (Threshold)
            html.Div(
                className="w-1/2 mx-auto mt-4",
                children=[
                    html.Label(
                        "Schwellenwert für Fluktuationswahrscheinlichkeit (%)",
                        className="text-gray-700 text-lg"
                    ),
                    dcc.Input(
                        id="threshold-input",
                        type="number",
                        value=0,  # Standardwert
                        placeholder="Schwellenwert",
                        className="rounded border-gray-300 w-full px-2 py-1",
                    )
                ],
            ),

            # Platzhalter für die Tabelle (wird dynamisch durch Callbacks aktualisiert)
            html.Div(id="critical-employees-table", className="mt-6"),
        ]
    )
# Callback zur Verarbeitung des Datei-Uploads und Aktualisierung des Datenrahmens
@app.callback(
    Output("upload-feedback", "children"),
    Input("upload-data", "contents"),
    [Input("upload-data", "filename"),
     Input("upload-data", "last_modified")]
)
def update_dataset(content, filename, last_modified):
    global df  # Greife auf den globalen DataFrame zu

    if content is not None:
        # Inhalte nach Base64 dekodieren und in Pandas-Datensatz umwandeln
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        try:
            if filename.endswith(".csv"):
                # Aktualisieren des globalen DataFrames mit der hochgeladenen Datei
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), low_memory=False)
                return f"Datei '{filename}' erfolgreich hochgeladen und Daten aktualisiert!"
            else:
                return "Fehler: Bitte laden Sie eine CSV-Datei hoch."
        except Exception as e:
            return f"Fehler beim Hochladen und Verarbeiten der Datei: {str(e)}"
    else:
        # Falls keine Datei hochgeladen wurde
        return "Keine Datei hochgeladen. Standarddatensatz wird verwendet."


# Aktualisierung des Datensatzes beim Starten der App
df = pd.read_csv("HR_cleaned.csv", low_memory=False)


@app.callback(
    [Output("kpi-avg-salary", "children"),
     Output("kpi-avg-satisfaction", "children"),
     Output("kpi-avg-absence", "children")],
    Input("year-dropdown", "value")
)
def update_kpis(selected_year):
    # Fallback, falls kein Jahr ausgewählt
    if selected_year is None:
        return "keine Daten verfügbar", "keine Daten verfügbar", "keine Daten verfügbar"

    # Sicherstellen, dass die Spalte "Jahr" numerisch ist
    if "Jahr" in df.columns and not pd.api.types.is_numeric_dtype(df["Jahr"]):
        df["Jahr"] = pd.to_numeric(df["Jahr"], errors="coerce")

    # Filtere die Daten für das ausgewählte Jahr
    df_filtered = df[df["Jahr"] == int(selected_year)].copy()

    # Fallback, falls die Filterung keine Daten ergibt
    if df_filtered.empty:
        return "keine Daten verfügbar", "keine Daten verfügbar", "keine Daten verfügbar"

    # Nur eindeutige Mitarbeiter berücksichtigen (basierend auf "Mitarbeiter_ID")
    if "Mitarbeiter_ID" not in df_filtered.columns:
        return "keine Daten verfügbar", "keine Daten verfügbar", "keine Daten verfügbar"

    df_unique = df_filtered.drop_duplicates(subset=["Mitarbeiter_ID"])

    # Durchschnittsgehalt berechnen
    avg_salary = df_unique["Gehalt"].mean() if "Gehalt" in df_unique.columns else None

    # Durchschnittliche Zufriedenheit berechnen
    avg_satisfaction = df_unique["Zufriedenheit"].mean() if "Zufriedenheit" in df_unique.columns else None

    # Nur Zeilen mit Abwesenheitsgrund "Krankheit" berücksichtigen
    if "Abwesenheitsgrund" in df_unique.columns and "Fehlzeiten_Krankheitstage" in df_unique.columns:
        absence_filtered = df_unique[df_unique["Abwesenheitsgrund"] == "Krankheit"]
        avg_absence = absence_filtered["Fehlzeiten_Krankheitstage"].mean()
    else:
        avg_absence = None

    # Rückgabe der berechneten Werte
    return (
        f"{avg_salary:,.2f} €" if avg_salary is not None else "keine Daten verfügbar",
        f"{avg_satisfaction:.2f} / 10" if avg_satisfaction is not None else "keine Daten verfügbar",
        f"{avg_absence:.2f} Tage" if avg_absence is not None else "keine Daten verfügbar"
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

def render_page_correlation_matrix():
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
@app.callback(
    Output("critical-employees-table", "children"),
    [Input("threshold-input", "value"),
     Input("model-dropdown", "value")]
)
def update_critical_employees(threshold, selected_model):
    """
    Ein zentraler Callback zur Verarbeitung von Schwellenwerteingabe oder Modellauswahl.
    """
    ctx = dash.callback_context

    # Prüfen, ob ein Trigger vorhanden ist
    if not ctx.triggered:
        return html.P("Kein Ereignis ausgelöst.", className="text-gray-500 text-center")

    # Herausfinden, welcher Input den Callback ausgelöst hat
    triggered_input = ctx.triggered[0]["prop_id"].split(".")[0]

    # Fall 1: Schwellenwertänderung
    if triggered_input == "threshold-input":
        try:
            # Lade die CSV-Datei
            input_file = "Outputs/Vergleich_Top_15.csv"
            critical_employees = pd.read_csv(input_file)
        except FileNotFoundError:
            return html.P(f"Die Datei '{input_file}' wurde nicht gefunden.", className="text-danger text-center")
        except Exception as e:
            return html.P(f"Fehler beim Laden der Datei: {str(e)}", className="text-danger text-center")

        # Anwenden der Schwellenwertlogik
        if threshold is not None:
            critical_employees = critical_employees[
                critical_employees["Fluktuationswahrscheinlichkeit"] >= threshold
                ]

        # Validierung der Daten
        if critical_employees.empty:
            return html.P("Keine kritischen Mitarbeiter gefunden.", className="text-center text-gray-500")

        # Rückgabe der Tabelle
        return dash_table.DataTable(
            id="critical-employees-table",
            columns=[{"name": col, "id": col} for col in critical_employees.columns],
            data=critical_employees.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "5px", "fontFamily": "Arial"},
            style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"},
        )

    # Fall 2: Modelländerung
    elif triggered_input == "model-dropdown":
        try:
            # Lade transformierte Daten und Originaldaten
            X_transformed = pd.read_csv("X_transformed.csv")
            df_original = pd.read_csv("HR_cleaned.csv")

            # Modell laden
            xgb_model = xgb.Booster()
            xgb_model.load_model("Models/xgboost_model.json")

            # Ergebnisse abrufen
            result = get_critical_employees(
                model=xgb_model,
                X_transformed=X_transformed,
                df=df_original,
                scaler_file="Models/scaler.pkl",
                threshold=threshold
            )

            # Extrahiere Resultate
            critical_employees = result["critical_employees"]
            errors = result.get("errors", [])

            # Fehleranzeige
            if errors:
                return html.P(f"Fehler: {'; '.join(errors)}", className="text-danger text-center")

            # Validieren der Ergebnisse
            if critical_employees.empty:
                return html.P("Keine kritischen Mitarbeiter gefunden.", className="text-center text-gray-500")

            # Rückgabe der Tabelle
            return dash_table.DataTable(
                id="critical-employees-table",
                columns=[{"name": col, "id": col} for col in critical_employees.columns],
                data=critical_employees.to_dict("records"),
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "5px", "fontFamily": "Arial"},
                style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"},
            )

        except Exception as e:
            return html.P(f"Fehler bei der Verarbeitung des Modells: {str(e)}", className="text-danger text-center")

    # Fallback-Fall (kein Ereignis ausgelöst)
    return html.P("Keine Aktion ausgeführt.", className="text-gray-500 text-center")

if __name__ == "__main__":
    app.run(debug=True)
