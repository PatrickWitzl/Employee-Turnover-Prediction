from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def prepare_data(df):
    """
    Categorize employees into three main categories:
    - Active
    - Retired
    - Left
    """
    if 'Status' in df.columns:
        df["Exit_Category"] = df["Status"].apply(
            lambda x: "Retired" if x == "Retired"
            else ("Left" if x == "Left"
                  else "Active")
        )
    return df


def filter_status_changes(df):
    """
    Filter status changes, including the following categories:
    - Active
    - Left
    - Retired (optional, depending on the analysis)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pd.DataFrame, received: {type(df)}")

    # Filter for specific analysis: Exclude "Retired" if not relevant.
    filtered_df = df[df['Status'].isin(['Active', 'Left', 'Retired'])]
    return filtered_df


def plot_active_and_left(df, save_path):
    """
    Add plots for 'Retired' status to the existing 'Active' and 'Left' statuses.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    required_columns = {'Year', 'Employee_ID', 'Exit_Category'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The DataFrame must contain the columns {required_columns}.")

    # Calculate yearly counts of Left, Retired, and Active employees
    left_per_year = (
        df[df['Exit_Category'] == 'Left']
        .groupby('Year')['Employee_ID']
        .nunique()
        .reset_index()
    )
    left_per_year.rename(columns={'Employee_ID': 'Left_Count'}, inplace=True)

    retired_per_year = (
        df[df['Exit_Category'] == 'Retired']
        .groupby('Year')['Employee_ID']
        .nunique()
        .reset_index()
    )
    retired_per_year.rename(columns={'Employee_ID': 'Retired_Count'}, inplace=True)

    active_per_year = (
        df[df['Exit_Category'] == 'Active']
        .groupby('Year')['Employee_ID']
        .nunique()
        .reset_index()
    )
    active_per_year.rename(columns={'Employee_ID': 'Active_Count'}, inplace=True)

    # Plot: Left
    plt.figure(figsize=(8, 6))
    plt.bar(left_per_year['Year'], left_per_year['Left_Count'], color='#FF5733', alpha=0.8,
            edgecolor='black')
    plt.title('Number of Employees Left per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Left Employees')
    plt.tight_layout()
    plt.savefig(save_path / "left_per_year.png", dpi=300)
    plt.close()

    # Plot: Retired
    plt.figure(figsize=(8, 6))
    plt.bar(retired_per_year['Year'], retired_per_year['Retired_Count'], color='#3498DB', alpha=0.8,
            edgecolor='black')
    plt.title('Number of Employees Retired per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Retired Employees')
    plt.tight_layout()
    plt.savefig(save_path / "retired_per_year.png", dpi=300)
    plt.close()

    # Plot: Active
    plt.figure(figsize=(8, 6))
    plt.bar(active_per_year['Year'], active_per_year['Active_Count'], color='#3CB371', alpha=0.8, edgecolor='black')
    plt.title('Number of Active Employees per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Active Employees')
    plt.tight_layout()
    plt.savefig(save_path / "active_per_year.png", dpi=300)
    plt.close()


def group_and_summarize_by_year(df):
    """
    Group data by year and category:
    - Active
    - Left
    - Retired
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pd.DataFrame, but received: {type(df)}")

    df = df.drop_duplicates(subset=['Year', 'Employee_ID'])

    grouped = df.groupby(['Year', 'Exit_Category'])['Employee_ID'].count().reset_index()
    grouped_pivot = grouped.pivot(index='Year', columns='Exit_Category', values='Employee_ID').fillna(0)
    grouped_pivot = grouped_pivot.reset_index()

    return grouped_pivot


def create_boxplot(df_filtered, variable, save_path):
    """
    Create a combined boxplot for multiple variables.

    Parameters:
        df_filtered (DataFrame): Filtered DataFrame for analysis.
        variable (List): List of variables to be visualized.
        save_path (Path): Path to save the combined plot.
    """
    # Check if necessary columns exist
    if 'Exit_Category' not in df_filtered.columns:
        print("Error: 'Exit_Category' column is missing in the DataFrame.")
        return

    # Transform DataFrame into long format
    df_long = df_filtered.melt(
        id_vars='Exit_Category',
        value_vars=variable,
        var_name='Variable',
        value_name='Value'
    )

    # Create plot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=df_long,
        x='Variable',
        y='Value',
        hue='Exit_Category',
        palette="coolwarm"
    )
    plt.title("Combined Boxplot: Variable Distribution by Exit Category")
    plt.xlabel("Variable")
    plt.ylabel("Value")
    plt.legend(title="Exit Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Prevent overlaps
    plt.savefig(save_path)
    plt.show()


def create_correlation_heatmap(df_filtered, save_path):
    """
    Create a correlation heatmap for numerical columns, masking only the diagonal.
    """
    # Select numerical columns
    numerical_columns = df_filtered.select_dtypes(include=['float64', 'int64']).columns
    correlations = df_filtered[numerical_columns].corr()

    # Create mask for the main diagonal
    mask = np.eye(correlations.shape[0], dtype=bool)  # Mask only the main diagonal

    # Draw the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlations,
        mask=mask,  # Apply mask to hide diagonal
        annot=True,  # Show correlation values
        cmap="coolwarm",  # Color palette
        fmt=".2f",  # Format values
        cbar=True  # Show color bar
    )
    plt.title("Correlations of Numerical Variables (Excluding Diagonal)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return correlations


def analyze_extreme_groups(df_filtered):
    """
    Analyze extreme groups (high/low salaries).
    """
    extreme_high = df_filtered[df_filtered['Salary'] > df_filtered['Salary'].quantile(0.95)]
    extreme_low = df_filtered[df_filtered['Salary'] < df_filtered['Salary'].quantile(0.05)]

    print("\nExtreme Groups: High Salaries:")
    print(extreme_high.describe())
    print("\nExtreme Groups: Low Salaries:")
    print(extreme_low.describe())

def enhanced_cluster_visualizations(df_filtered, cluster_features, save_dir):
    """
    Advanced visualizations for cluster analysis.

    Creates scatterplots, pair plots, 3D plots, heatmaps, silhouette analysis,
    and interactive charts based on cluster analysis results.
    """

    # Check for valid data for the analysis
    df = df_filtered.dropna(subset=cluster_features)
    if df.empty:
        print("Error: No suitable data available for cluster analysis.")
        return

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_features])

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Compute the cluster means
    cluster_means = df.groupby('Cluster')[cluster_features].mean()
    print("\nCluster Means:")
    print(cluster_means)

    # Pairplot
    sns.pairplot(df[cluster_features + ['Cluster']], hue='Cluster', palette="Set2", diag_kind="kde", height=2.5)
    plt.subplots_adjust(top=0.95)
    plt.suptitle("Pair Plot of Clusters", fontsize=16)  # Add title
    plt.savefig(save_dir / "pairplot_cluster.png", bbox_inches="tight")
    plt.show()

    # 3D Scatterplot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['Age'], df['Salary'], df['Tenure'], c=df['Cluster'], cmap='Set1', s=60)
    ax.set_title("3D Cluster Analysis", fontsize=16)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Salary", fontsize=12)
    ax.set_zlabel("Tenure", fontsize=12)
    legend_labels = [f"Cluster {i}" for i in range(kmeans.n_clusters)]
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Cluster", loc='upper left',
              fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "cluster_3d.png", bbox_inches="tight")
    plt.show()

    # Heatmap of Cluster Means
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Heatmap of Cluster Means", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Cluster", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / "cluster_heatmap.png", bbox_inches="tight")
    plt.show()

    # Silhouette Analysis
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
    plt.axvline(avg_silhouette, color="red", linestyle="--", label="Average Silhouette Value")
    plt.yticks(y_ticks, [f"Cluster {i}" for i in range(kmeans.n_clusters)], fontsize=12)
    plt.ylabel("Cluster", fontsize=14)
    plt.xlabel("Silhouette Coefficient", fontsize=14)
    plt.title("Silhouette Plot of Clusters", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "silhouette_plot.png", bbox_inches="tight")
    plt.show()


def yearly_active_and_left_analysis(df_filtered, save_path):
    """Analysis of the number of active and left employees per year."""
    if {'Year', 'Employee_ID', 'Status'}.issubset(df_filtered.columns):
        # Remove duplicate entries based on year and employee ID
        unique_df = df_filtered.drop_duplicates(subset=['Year', 'Employee_ID'])

        # Group and count the left ("Left") and active ("Active") employees per year
        yearly_data = (
            unique_df
            .groupby(['Year', 'Status'])['Employee_ID']
            .nunique()
            .reset_index(name='Count')
            .pivot(index='Year', columns='Status', values='Count')
        )
        yearly_data = yearly_data.fillna(0).reset_index()  # Fill missing values with 0

        # Display results
        print("Yearly Analysis (Count of Active and Left):")
        print(yearly_data)

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_data['Year'], yearly_data['Left'], marker='o', label="Left", color='red')
        plt.plot(yearly_data['Year'], yearly_data['Active'], marker='o', label="Active", color='green')
        plt.title("Number of Active and Left Employees Per Year")
        plt.xlabel("Year")
        plt.ylabel("Employee Count")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()
    else:
        print("The columns 'Year', 'Employee_ID', and 'Status' are missing in the DataFrame.")


def yearly_hired_analysis(df, save_path):
    """Analysis of the number of employees hired per year."""
    if 'Hire_Date' in df.columns:
        # Ensure 'Hire_Date' is in a datetime format
        df['Hire_Year'] = pd.to_datetime(df['Hire_Date']).dt.year

        # Group by year and count
        hires_per_year = df.groupby('Hire_Year')['Employee_ID'].nunique().reset_index()
        hires_per_year.columns = ['Year', 'Hired']

        # Display results
        print("Yearly Analysis (Count of Hired Employees):")
        print(hires_per_year)

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.bar(hires_per_year['Year'], hires_per_year['Hired'], color='blue', alpha=0.7, label="Hired")
        plt.title("Yearly Number of Hired Employees")
        plt.xlabel("Year")
        plt.ylabel("Number of Hired Employees")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(save_path)
        plt.show()
    else:
        print("The column 'Hire_Date' is missing in the DataFrame.")

def yearly_left_analysis(df, save_path):
    """Analysis of the number of left employees per year."""
    if {'Year', 'Employee_ID', 'Status'}.issubset(df.columns):
        # Filter left employees
        left_df = df[df['Status'] == 'Left']

        # Group and count employees per year
        left_per_year = left_df.groupby('Year')['Employee_ID'].nunique().reset_index()
        left_per_year.columns = ['Year', 'Left']

        # Display results
        print("Yearly Analysis (Number of Left Employees):")
        print(left_per_year)

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.bar(left_per_year['Year'], left_per_year['Left'], color='red', alpha=0.7,
                label="left")
        plt.title("Yearly Number of Left Employees")
        plt.xlabel("Year")
        plt.ylabel("Number of left Employees")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(save_path)
        plt.show()
    else:
        print("The columns 'Year', 'Employee_ID', and 'Status' are missing in the DataFrame.")


def yearly_active_trend(df, save_path):
    """Trend of active employees per year."""
    if {'Year', 'Employee_ID', 'Status'}.issubset(df.columns):
        # Filter active employees
        active_df = df[df['Status'] == 'Active']

        # Group and count employees per year
        active_per_year = active_df.groupby('Year')['Employee_ID'].nunique().reset_index()
        active_per_year.columns = ['Year', 'Active']

        # Display results
        print("Yearly Trend of Active Employees:")
        print(active_per_year)

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(active_per_year['Year'], active_per_year['Active'], marker='o', color='green', label="Active")
        plt.title("Trend of Active Employees Per Year")
        plt.xlabel("Year")
        plt.ylabel("Number of Active Employees")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(save_path)
        plt.show()
    else:
        print("The columns 'Year', 'Employee_ID', and 'Status' are missing in the DataFrame.")


def main():
    import time
    import pandas as pd
    from pathlib import Path

    # Start timer
    start_time = time.time()

    # Create directory for plots
    PLOTS_DIR = Path("plots")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load file
    file_path = "../data/HR_cleaned.csv"
    if not Path(file_path).exists():
        print(f"ERROR: File '{file_path}' does not exist.")
        return

    print(f"Loading file '{file_path}'...")
    df = pd.read_csv(file_path)

    # 1. Prepare data
    df = prepare_data(df)
    print("Data prepared.")

    # 2. Filter status changes (only keep relevant categories)
    df_filtered = filter_status_changes(df)
    print("Data filtered (only 'Active', 'Left', 'Retired').")

    # 3. Group and summarize data
    grouped_data = group_and_summarize_by_year(df_filtered)
    print("Summarized data by year and category:")
    print(grouped_data)

    # 4. Create plots

    # a) Trend of active/left employees
    print("Creating trend of active/left employees...")
    yearly_active_and_left_analysis(df_filtered, PLOTS_DIR / "active_and_left_trend.png")

    # b) Plots: Hired and left employees
    print("Creating plot for hired employees per year...")
    yearly_hired_analysis(df, PLOTS_DIR / "hired_per_year.png")

    print("Creating plot for left employees per year...")
    yearly_left_analysis(df, PLOTS_DIR / "left_per_year.png")

    print("Creating trend of active employees...")
    yearly_active_trend(df, PLOTS_DIR / "active_trend.png")

    # c) Combined plot (Active, Left, Retired)
    print("Creating combined plot for 'Active', 'Left', and 'Retired'...")
    plot_active_and_left(df_filtered, PLOTS_DIR)

    # 5. Create boxplots
    variables = ['Salary', 'Overtime', 'Willingness_to_Change', 'Satisfaction']  # Example variables
    for var in variables:
        if var in df_filtered.columns:
            print(f"Creating boxplot for {var}...")
            create_boxplot(df_filtered, var, PLOTS_DIR / f"boxplot_{var.lower()}.png")

    # 6. Correlation heatmap
    print("Creating correlation heatmap...")
    correlations = create_correlation_heatmap(df_filtered, PLOTS_DIR / "correlation_heatmap.png")
    print("Correlation matrix:")
    print(correlations)

    # 7. Analyze extreme groups
    print("Analyzing extreme groups (high/low salaries)...")
    analyze_extreme_groups(df_filtered)

    # 9. Advanced cluster analysis and visualizations
    cluster_features = ['Age', 'Salary', 'Overtime', 'Switching Readiness', 'Satisfaction']  # Example
    if all(feature in df_filtered.columns for feature in cluster_features):
        print("Performing advanced cluster visualizations...")
        enhanced_cluster_visualizations(df_filtered, cluster_features, PLOTS_DIR)

    # Stop timer
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()