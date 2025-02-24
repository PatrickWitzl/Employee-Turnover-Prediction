import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from datetime import datetime


# Beispiel: CSV laden (ersetze dies durch deine tatsächlichen Daten)
df = pd.read_csv("../HR_cleaned.csv", low_memory=False)

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
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded")
            ],
        ),

        # Platzhalter für dynamische Inhalte (Seiteninhalt)
        html.Div(id="page-content", className="mt-4")
    ]
)

# Navigation zwischen Seiten
@app.callback(
    Output("page-content", "children"),
    [Input("btn-to-page-1", "n_clicks"),
     Input("btn-to-page-2", "n_clicks")]
)
def navigate(button_1_clicks, button_2_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return render_page_1()  # Standard: Seite 1
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == "btn-to-page-1":
        return render_page_1()
    elif button_id == "btn-to-page-2":
        return render_page_2()

    return render_page_1()  # Fallback

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
    fig = px.line(filtered_data, x="Monat", y="Anzahl", title="Trend Aktiv", color_discrete_map=color_mapping)
    return fig


@app.callback(
    Output("retired-exited-monthly-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_retired_exited_monthly_trend(selected_year):
    filtered_data = df[(df["Status"].isin(["Ausgeschieden", "Ruhestand"])) & (df["Jahr"] == selected_year)]
    filtered_data = filtered_data.groupby(["Monat", "Status"]).size().reset_index(name="Anzahl")
    fig = px.line(filtered_data, x="Monat", y="Anzahl", color="Status", title="Trend Ruhestand/Ausgeschieden", color_discrete_map=color_mapping
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
                  color_discrete_map={"Aktiv": "#1f77b4"})  # Blau für Aktiv
    return fig


@app.callback(
    Output("retired-exited-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_retired_exited_trend(selected_year):
    filtered_data = df[df["Status"].isin(["Ausgeschieden", "Ruhestand"])]
    filtered_data = filtered_data.groupby(["Jahr", "Status"]).size().reset_index(name="Anzahl")
    fig = px.line(filtered_data, x="Jahr", y="Anzahl", color="Status", title="Jährliche Trends (ausgeschieden)", color_discrete_map=color_mapping
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

if __name__ == "__main__":
    app.run(debug=True)
