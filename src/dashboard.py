import io
import os
import joblib
import seaborn as sns
import base64
import dash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from io import BytesIO
from dash import dcc, html, Input, Output, State, dash_table
from src.data_loading import load_dataset
from src.data_cleaning import clean_dataset
from ML1_churn_prediction_best_model import preprocess_data
from ML1_churn_prediction_best_model import get_critical_employees

# Beispielhafte Modellpfade
model_paths = {
    "Random Forest": "models/random_forest_model.pkl",
    "XGBoost": "models/xgboost_model.pkl",
    "LightGBM": "models/lightgbm_model.pkl"
}

try:
    df = pd.read_csv("../data/HR_cleaned.csv", low_memory=False)
except FileNotFoundError:
    df = pd.DataFrame()

df = df.copy()

color_mapping = {
    "Active": "#1f77b4",
    "Left": "#2ca02c",
    "Retired": "#ff7f0e",
}

# App-Konfiguration
# Tailwind CSS einbinden

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css",]

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
                html.Button("Analysis & Details", id="btn-to-page-2",
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
                html.Button("Correlation Matrix", id="btn-to-page-3",  # Neuer Button für Korrelationsmatrix
                            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"),
                html.Button("Critical Employees", id="btn-to-page-4",
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
                "boxShadow": "0 -2px 5px rgba(0, 0, 0, 0.1)",
                "zIndex": "1000"
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

def render_page_1():
    return html.Div(
        className="space-y-6",
        children=[

            # --- Dropdown for year selection ---
            html.Div(
                className="w-1/2 mx-auto",
                children=[
                    dcc.Dropdown(
                        id="year-dropdown",
                        options=[{"label": str(year), "value": year} for year in sorted(df["Year"].unique())],
                        placeholder="Select Year",
                        value=sorted(df["Year"].unique())[-1] if len(df["Year"].unique()) > 0 else None,
                        className="rounded border-gray-300"
                    )
                ],
            ),

            # --- Title section ---
            html.H2("KPIs & Trends", className="text-center text-blue-600 font-bold text-2xl"),

            # --- KPI Titles (e.g., Salary, Satisfaction, Sick Days) ---
            html.Div(
                className="grid grid-cols-3 gap-4",
                children=[
                    html.H3(id="kpi-title-salary", className="text-center text-lg text-gray-600"),
                    html.H3(id="kpi-title-satisfaction", className="text-center text-lg text-gray-600"),
                    html.H3(id="kpi-title-absence", className="text-center text-lg text-gray-600"),
                ]
            ),

            # --- KPI Values (average salary, satisfaction, sick days) ---
            html.Div(
                className="grid grid-cols-3 gap-4",
                children=[
                    html.H3(id="kpi-avg-salary", className="text-center text-lg text-gray-600"),
                    html.H3(id="kpi-avg-satisfaction", className="text-center text-lg text-gray-600"),
                    html.H3(id="kpi-avg-absence", className="text-center text-lg text-gray-600"),
                ]
            ),

            # --- Totals at the top of the page ---
            html.Div(
                className="grid grid-cols-3 gap-4",
                children=[
                    # New Hires
                    html.Div([
                        html.H3(id="sum-new-hires", style={"color": "#1f77b4"},  # Blue
                                className="text-center text-lg font-bold"),
                        html.P("New Hires", className="text-center text-gray-600")
                    ]),
                    # Left
                    html.Div([
                        html.H3(id="sum-left", style={"color": "#2ca02c"},  # Green
                                className="text-center text-lg font-bold"),
                        html.P("Left", className="text-center text-gray-600")
                    ]),
                    # Retired
                    html.Div([
                        html.H3(id="sum-retired", style={"color": "#ff7f0e"},  # Orange
                                className="text-center text-lg font-bold"),
                        html.P("In Retirement", className="text-center text-gray-600")
                    ]),
                ],
            ),

            # --- Monthly Trends ---
            html.Div(
                className="grid grid-cols-2 gap-6",
                children=[
                    html.Div([
                        html.H3("Monthly Trend: Active", className="text-center text-gray-800"),
                        dcc.Graph(id="active-monthly-trend-line")
                    ]),
                    html.Div([
                        html.H3("Monthly Trend: Left and Retired",
                                className="text-center text-gray-800"),
                        dcc.Graph(id="retired-left-monthly-trend-line")
                    ]),
                ]
            ),

            # --- Yearly Trends ---
            html.Div(
                className="grid grid-cols-2 gap-6",
                children=[
                    html.Div([
                        html.H3("Yearly Trend: Active", className="text-center text-gray-800"),
                        dcc.Graph(id="active-trend-line")
                    ]),
                    html.Div([
                        html.H3("Yearly Trend: Left and Retired",
                                className="text-center text-gray-800"),
                        dcc.Graph(id="retired-left-trend-line")
                    ]),
                ]
            ),
        ]
    )

@app.callback(
    [Output("kpi-avg-salary", "children"),
     Output("kpi-avg-satisfaction", "children"),
     Output("kpi-avg-absence", "children")],
    Input("year-dropdown", "value")
)
def update_kpis(selected_year):
    if selected_year is None or "Year" not in df.columns:
        return "No data", "No data", "No data"

    # Filter the data based on the selected year
    filtered_data = df[df["Year"] == selected_year]
    if filtered_data.empty:
        return "No data", "No data", "No data"

    # Calculate average salary and satisfaction
    avg_salary = filtered_data["Salary"].mean()
    avg_satisfaction = filtered_data["Satisfaction"].mean()

    # Calculate average sick days:
    # Filter rows where the Absence Reason is "Illness" and sum "Absence Days"
    sick_days_data = filtered_data[filtered_data["Absence Reason"] == "Illness"]
    avg_absence = sick_days_data["Absence Days"].mean()

    # Return the KPIs formatted
    return (
        f"{avg_salary:.2f} €" if not pd.isna(avg_salary) else "No data",
        f"{avg_satisfaction:.2f} / 10" if not pd.isna(avg_satisfaction) else "No data",
        f"{avg_absence:.2f} days" if not pd.isna(avg_absence) else "No data"
    )

@app.callback(
    [Output("kpi-title-salary", "children"),
     Output("kpi-title-satisfaction", "children"),
     Output("kpi-title-absence", "children")],
    Input("year-dropdown", "value")
)
def update_kpi_titles(selected_year):
    if selected_year is None:
        return "Salary (No Data)", "Satisfaction (No Data)", "Sick Days (No Data)"
    return (
        f"Salary ({selected_year})",
        f"Satisfaction ({selected_year})",
        f"Sick Days ({selected_year})"
    )
@app.callback(
    [Output("sum-new-hires", "children"),
     Output("sum-left", "children"),
     Output("sum-retired", "children")],
    Input("year-dropdown", "value")
)
def update_status_sums(selected_year):
    # Ensure all required columns are available
    required_columns = ["Hiring Date", "Exit Date", "Status", "Month", "Employee_ID"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The column '{col}' is missing in the dataset.")

    # Convert 'Hiring Date' and 'Exit Date' to datetime format
    df["Hiring Date"] = pd.to_datetime(df["Hiring Date"], errors="coerce", format="%Y-%m-%d")
    df["Exit Date"] = pd.to_datetime(df["Exit Date"], errors="coerce", format="%Y-%m-%d")

    # If no year is selected, return default values
    if not selected_year:
        return "No year selected", "No year selected", "No year selected"

    # FILTER: New Hires
    new_hires_df = df[
        (df["Hiring Date"].notna()) &
        (df["Hiring Date"].dt.year == selected_year)
        ]
    num_new_hires = new_hires_df["Employee_ID"].nunique()

    # FILTER: Exits
    left_df = df[
        (df["Status"] == "Left") &
        (df["Exit Date"].notna()) &
        (df["Exit Date"].dt.year == selected_year)
        ]
    num_exits = left_df["Employee_ID"].nunique()

    # FILTER: Retirements
    retired_df = df[
        (df["Status"] == "Retired") &
        (df["Exit Date"].notna()) &
        (df["Exit Date"].dt.year == selected_year)
        ]
    num_retired = retired_df["Employee_ID"].nunique()

    # Return the calculated results
    return (
        f"{num_new_hires} people",  # New Hires
        f"{num_exits} people",  # Exits
        f"{num_retired} people"  # Retirements
    )

@app.callback(
    Output("active-monthly-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_active_monthly_trend(selected_year):
    # Filter data for active employees in the selected year
    filtered_data = df[(df["Status"] == "Active") & (df["Year"] == selected_year)]
    filtered_data = filtered_data.groupby("Month").size().reset_index(name="Count")

    # Create a line plot
    fig = px.line(
        filtered_data,
        x="Month",
        y="Count",
        title="Active Trend",
        color_discrete_map=color_mapping,
        markers=True
    )
    return fig

@app.callback(
    Output("retired-left-monthly-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_retired_left_monthly_trend(selected_year):
    # Filter data for employees with status "Left" or "Retired" in the selected year
    filtered_data = df[(df["Status"].isin(["Left", "Retired"])) & (df["Year"] == selected_year)]
    filtered_data = filtered_data.groupby(["Month", "Status"]).size().reset_index(name="Count")

    # Create a line plot
    fig = px.line(
        filtered_data,
        x="Month",
        y="Count",
        color="Status",
        title="Retired/Left Trend",
        color_discrete_map=color_mapping,
        markers=True
    )
    return fig

@app.callback(
    Output("active-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_active_trend(selected_year):
    # Filter for active employees
    filtered_data = df[df["Status"] == "Active"]

    # Keep only unique combinations of Employee_ID and Year
    unique_data = filtered_data.drop_duplicates(subset=["Employee_ID", "Year"])

    # Group by Year and count unique entries
    trend_data = unique_data.groupby("Year").size().reset_index(name="Count")

    # Create the visualization
    fig = px.line(
        trend_data,
        x="Year",
        y="Count",
        title="Active - Yearly Trend",
        color_discrete_map={"Active": "#1f77b4"},
        markers=True
    )
    return fig
@app.callback(
    Output("retired-left-trend-line", "figure"),
    Input("year-dropdown", "value")
)
def update_retired_left_trend(selected_year):
    # Filter data for employees with status "Left" or "Retired"
    filtered_data = df[df["Status"].isin(["Left", "Retired"])]

    # Group by Year and Status, and count occurrences
    filtered_data = filtered_data.groupby(["Year", "Status"]).size().reset_index(name="Count")

    # Create a line plot
    fig = px.line(
        filtered_data,
        x="Year",
        y="Count",
        color="Status",
        title="Yearly Trends (Left/Retired)",
        color_discrete_map=color_mapping,
        markers=True
    )
    return fig



def render_page_2():
    scatter_fig = px.scatter(df, x="Salary", y="Satisfaction", color="Status",
                                 title="Satisfaction vs. Salary by Status", color_discrete_map=color_mapping
                                 )
    return html.Div([
        html.H2("Analytics & Details", className="text-center text-blue-600 font-bold text-2xl mb-6"),

        # Scatterplot: Satisfaction vs Salary
        dcc.Graph(figure=scatter_fig, className="rounded shadow-lg"),

        # Dropdown for interactive analysis
        html.Div(
            className="mt-6",
            children=[
                html.H3("Interactive Analysis", className="text-center text-gray-700 text-lg"),
                dcc.Dropdown(
                    id="analysis-dropdown",
                    options=[
                        {"label": "Age", "value": "Age"},
                        {"label": "Salary", "value": "Salary"},
                    ],
                    value="Age",
                ),
                dcc.Graph(id="interactive-plot"),
            ]
        )
    ])

@app.callback(
    Output("interactive-plot", "figure"),  # Output for the interactive graphic
    Input("analysis-dropdown", "value")  # Input from the dropdown
)
def update_interactive_plot(selected_feature):
    # Ensure the selection is valid
    if selected_feature not in df.columns:
        return px.scatter(title="No valid data available")

    # Create a histogram based on the selected feature from the dropdown
    fig = px.histogram(
        df,
        x=selected_feature,  # X-axis based on the selected column
        title=f"Frequency Distribution: {selected_feature}",  # Title of the plot
        color="Status",  # Color based on 'Status'
        barmode="group",  # Grouped display of bars
        histnorm="percent"  # Normalize to percent (optional)
    )
    return fig

def render_page_3_correlation_matrix():
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return html.Div([
            html.H2("Correlation Matrix", className="text-center text-blue-600 font-bold text-2xl mt-6 mb-6"),
            html.P("The dataset does not contain any numeric columns that can be correlated.",
                   className="text-center text-gray-600 text-lg"),
        ])

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Remove the main diagonal using masking
    mask = np.eye(len(correlation_matrix), dtype=bool)
    correlation_matrix_no_diag = correlation_matrix.mask(mask)

    # High-resolution heatmap using Seaborn
    plt.figure(figsize=(20, 20), dpi=300)
    heatmap = sns.heatmap(
        correlation_matrix_no_diag,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.6},
        linewidths=0.5,
        mask=mask
    )
    plt.title("Correlation Matrix Without Diagonal", fontsize=20, pad=30)
    plt.xticks(rotation=45, fontsize=12, ha="right")
    plt.yticks(fontsize=12)

    # Encode the Matplotlib plot as a Base64 image
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")  # Trim edges
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    # Embed Matplotlib image in Dash
    return html.Div([
        html.H2("Correlation Matrix Without Diagonal",
                className="text-center text-blue-600 font-bold text-2xl mt-4 mb-4"),
        html.Img(src=f"data:image/png;base64,{encoded_image}", style={"width": "100%", "height": "auto"}),
        # Scale image
    ])

def render_page_4():
    return html.Div(
        children=[
            html.H2(
                "Data Upload & Update",
                className="text-center text-blue-600 font-bold text-2xl mb-6",
            ),
            html.Div(
                className="mt-4 mb-6",
                children=[
                    html.H3("Upload CSV File:", className="text-lg text-gray-700"),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Drag and drop file here or ", html.A("Select File")]
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
                    html.H3("Select Model:", className="mt-6 text-lg text-gray-700"),
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[
                            {"label": "XGBoost", "value": "XGBoost"},
                        ],
                        placeholder="Choose a Model",
                        className="mt-2",
                    ),
                    html.H3("Threshold for Turnover Probability (%):",
                            className="mt-6 text-lg text-gray-700"),
                    dcc.Input(
                        id="threshold-input",
                        type="number",
                        value=10,
                        placeholder="Enter Threshold",
                        className="mt-2",
                    ),
                    html.Div(id="upload-status", className="mt-4 text-lg text-gray-700"),
                    html.Div(id="critical-employees-table", className="mt-8", style={"marginBottom": "40px"}
                             ),
                    html.Div(
                        children=[
                            html.H3(
                                "Note: If no file is uploaded, the default file will be used.",
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
    global df  # Access the global variable for the DataFrame

    if content is not None:
        # Decode content and convert it into a Pandas DataFrame
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)

        try:
            # Check if the uploaded file is a CSV file
            if filename.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), low_memory=False)
                return f"The file '{filename}' was successfully uploaded!"
            else:
                return "Please upload CSV files only."
        except Exception as e:
            return f"Error reading the file: {str(e)}"
    else:
        return "No file uploaded."




# Callback for file upload and processing
@app.callback(
    [
        Output("upload-feedbacks", "children"),  # Feedback to the user
        Output("critical-employees-table", "children"),  # Table showing critical employees
        Output("upload-status", "children"),  # Upload status message
    ],
    [
        Input("upload-data", "contents"),  # Uploaded file contents
        Input("model-dropdown", "value"),  # Selected model from the dropdown
        Input("threshold-input", "value"),  # Threshold value for critical employees
    ],
    [State("upload-data", "filename")],  # Filename of the uploaded file
)
def process_and_display_critical_employees(contents, selected_model, threshold, filename):
    # Check if any file was uploaded
    if not contents:
        return "Please upload a file.", None, "No file uploaded."

    # Check if a model was selected
    if not selected_model:
        return "Please select a model.", None, "No model selected."

    try:
        # ** Step 1: Load the file **
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        uploaded_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

        # Temporary storage path for the uploaded file
        temp_path = "../data/uploaded_temp.csv"
        uploaded_df.to_csv(temp_path, index=False)  # Save the file temporarily

        file_path = temp_path

        # Process and identify critical employees
        critical_employees, top_15_employees = process_and_identify_critical_employees(
            file_path,  # Input file path
            save_filtered_path=None,  # Path to save filtered data
            models_dir="models",  # Directory where the model is stored
            model_file_name="xgboost_model.pkl",  # Model file name
            feature_names_file="models/lightgbm_feature_names.pkl",  # Feature names file
            threshold=0.0  # Threshold for critical employees
        )

        # Check if any critical employees were found
        if critical_employees is None or critical_employees.empty:
            return "No critical employees found.", None, "No critical employees."

        # Define columns to display in the table
        columns_to_display = [
            'Year', 'Month', 'Employee_ID', 'Name', 'Position',
            'Age', 'Status', 'Turnover_Probability'
        ]

        # Filter the DataFrame to include only the selected columns
        filtered_top_15_employees = top_15_employees[columns_to_display]

        # Create a DataTable using Dash
        table = dash_table.DataTable(
            id="critical-employees-results",
            columns=[{"name": col, "id": col} for col in filtered_top_15_employees.columns],
            data=filtered_top_15_employees.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "5px"},
            style_header={"backgroundColor": "lightblue", "fontWeight": "bold"},
            page_size=15,  # Maximum rows per page
        )

        return f"File '{filename}' successfully processed.", table, f"{len(critical_employees)} critical employees found."

    except Exception as e:
        # Handle any errors that occur during file upload or processing
        return f"An error occurred: {str(e)}", None, "Error during processing."


def process_and_identify_critical_employees(
        file_path,  # Input file
        save_filtered_path=None,  # Path to save filtered data
        models_dir="Models",  # Directory where the model is stored
        model_file_name="xgboost_model.pkl",  # Name of the model file
        feature_names_file="Models/lightgbm_feature_names.pkl",  # File containing feature names
        threshold=0.0  # Threshold for identifying critical employees
):
    """
    This function handles data preprocessing, model loading, and identification
    of critical employees based on a predictive model.

    Args:
        file_path (str): Path to the input file (CSV).
        save_filtered_path (str): Path to save the filtered input file.
        models_dir (str): Directory where the trained model is stored.
        model_file_name (str): Name of the file in which the model is stored.
        feature_names_file (str): File containing feature names for interpretation.
        threshold (float): Threshold for identifying critical employees.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame with critical employees and the top 15 employees by risk.
    """
    try:
        # Load and clean data
        df = load_dataset(file_path, save_filtered_path)
        df_cleaned = clean_dataset(df)

        # Preprocess data
        df_processed, X_transformed, X_resampled, y_resampled, preprocessor = preprocess_data(df_cleaned)

        # Create model path and load the model
        model_file = os.path.join(models_dir, model_file_name)
        try:
            model = joblib.load(model_file)  # Load the model
            print("The model was successfully loaded.")
        except FileNotFoundError:
            print(f"Error: The model file '{model_file}' was not found.")
            return None, None
        except Exception as e:
            print(f"Error while loading the model: {e}")
            return None, None

        # Identify critical employees
        critical_employees, top_15_employees = get_critical_employees(
            model, X_transformed, df_processed, feature_names_file=feature_names_file, threshold=threshold
        )

        # Output results
        print(
            f"Top 15 Employees:\n{top_15_employees[['Employee_ID', 'Name', 'Turnover_Probability']]}"
            # Update column names
        )

        return critical_employees, top_15_employees
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


if __name__ == "__main__":
    app.run(debug=True)
