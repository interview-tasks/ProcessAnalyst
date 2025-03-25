import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_validate_data(file_path):
    """
    Load data with validation and basic cleaning
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Pandas DataFrame with cleaned data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Basic validation
        required_columns = [
            'Proses ID', 'Proses Tipi', 'Proses Addımı', 'Emal Həcmi (ton)',
            'Temperatur (°C)', 'Təzyiq (bar)', 'Prosesin Müddəti (saat)', 
            'Emalın Səmərəliliyi (%)', 'Enerji İstifadəsi (kWh)',
            'Ətraf Mühitə Təsir (g CO2 ekvivalent)', 'Təhlükəsizlik Hadisələri',
            'Əməliyyat Xərcləri (AZN)'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check for null values in critical columns
        critical_columns = ['Proses Tipi', 'Proses Addımı', 'Emal Həcmi (ton)', 'Enerji İstifadəsi (kWh)']
        null_counts = {col: df[col].isnull().sum() for col in critical_columns if df[col].isnull().any()}
        
        if null_counts:
            print(f"Warning: Found null values in critical columns: {null_counts}")
        
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_derived_metrics(df):
    """
    Create comprehensive derived metrics for process analysis
    
    Args:
        df: Input dataframe with raw process data
        
    Returns:
        DataFrame with added derived metrics
    """
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Basic efficiency metrics
    processed_df['Energy_Efficiency'] = processed_df['Emal Həcmi (ton)'] / processed_df['Enerji İstifadəsi (kWh)']
    processed_df['Material_Productivity'] = processed_df['Emalın Səmərəliliyi (%)'] / 100  # Convert % to decimal
    processed_df['Cost_Efficiency'] = processed_df['Emal Həcmi (ton)'] / processed_df['Əməliyyat Xərcləri (AZN)']
    
    # Safety metrics
    processed_df['Safety_Risk'] = processed_df['Təhlükəsizlik Hadisələri'] / processed_df['Emal Həcmi (ton)'] * 1000  # Incidents per 1000 tons
    processed_df['Has_Incident'] = (processed_df['Təhlükəsizlik Hadisələri'] > 0).astype(int)  # Binary indicator
    
    # Environmental metrics
    processed_df['Environmental_Impact'] = processed_df['Ətraf Mühitə Təsir (g CO2 ekvivalent)'] / processed_df['Emal Həcmi (ton)']
    
    # Time efficiency metrics
    processed_df['Processing_Speed'] = processed_df['Emal Həcmi (ton)'] / processed_df['Prosesin Müddəti (saat)']  # Tons per hour
    
    # Operational metrics
    processed_df['Worker_Productivity'] = processed_df['Emal Həcmi (ton)'] / processed_df['İşçi Sayı']  # Tons per worker
    
    # Combined KPI score (weighted average of key metrics)
    # Weights can be adjusted based on business priorities
    processed_df['Process_KPI_Score'] = (
        0.3 * processed_df['Material_Productivity'] + 
        0.2 * (processed_df['Energy_Efficiency'] / processed_df['Energy_Efficiency'].max()) +
        0.2 * (1 - processed_df['Safety_Risk'] / processed_df['Safety_Risk'].max()) +  # Lower risk is better
        0.2 * (processed_df['Cost_Efficiency'] / processed_df['Cost_Efficiency'].max()) +
        0.1 * (1 - processed_df['Environmental_Impact'] / processed_df['Environmental_Impact'].max())  # Lower impact is better
    ) * 100  # Scale to 0-100
    
    # Process duration categories for analysis
    duration_bins = [0, 8, 12, 16, float('inf')]
    duration_labels = ['Short (<8h)', 'Medium (8-12h)', 'Long (12-16h)', 'Extended (>16h)']
    processed_df['Duration_Category'] = pd.cut(
        processed_df['Prosesin Müddəti (saat)'], 
        bins=duration_bins, 
        labels=duration_labels
    )
    
    # Temperature categories
    temp_bins = [0, 150, 300, 450, float('inf')]
    temp_labels = ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)']
    processed_df['Temperature_Category'] = pd.cut(
        processed_df['Temperatur (°C)'], 
        bins=temp_bins, 
        labels=temp_labels
    )
    
    # Pressure categories
    pressure_bins = [0, 10, 30, 50, float('inf')]
    pressure_labels = ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)']
    processed_df['Pressure_Category'] = pd.cut(
        processed_df['Təzyiq (bar)'], 
        bins=pressure_bins, 
        labels=pressure_labels
    )
    
    print(f"Created {len(processed_df.columns) - len(df.columns)} new derived metrics")
    return processed_df

def create_aggregated_datasets(df):
    """
    Create multiple aggregated datasets for different analysis views
    
    Args:
        df: Processed DataFrame with derived metrics
        
    Returns:
        Dictionary of DataFrames with various aggregations
    """
    datasets = {}
    
    # 1. Process Type aggregation
    datasets['process_types'] = df.groupby('Proses Tipi').agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Energy_per_ton': 'mean',
        'CO2_per_ton': 'mean',
        'Cost_per_ton': 'mean',
        'Təhlükəsizlik Hadisələri': 'sum',
        'Emal Həcmi (ton)': 'sum',
        'Has_Incident': 'mean',  # Gives proportion of processes with incidents
        'Process_KPI_Score': 'mean',
        'Safety_Risk': 'mean',
        'Energy_Efficiency': 'mean',
        'Processing_Speed': 'mean'
    }).reset_index()
    
    # 2. Process Step aggregation
    datasets['process_steps'] = df.groupby('Proses Addımı').agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Energy_per_ton': 'mean',
        'CO2_per_ton': 'mean',
        'Cost_per_ton': 'mean',
        'Təhlükəsizlik Hadisələri': 'sum',
        'Emal Həcmi (ton)': 'sum',
        'Has_Incident': 'mean',
        'Process_KPI_Score': 'mean',
        'Prosesin Müddəti (saat)': 'mean',
        'Temperatur (°C)': 'mean',
        'Təzyiq (bar)': 'mean'
    }).reset_index()
    
    # 3. Process Type + Step combined analysis
    datasets['process_type_step'] = df.groupby(['Proses Tipi', 'Proses Addımı']).agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Energy_per_ton': 'mean',
        'CO2_per_ton': 'mean',
        'Has_Incident': 'mean',
        'Process_KPI_Score': 'mean',
        'Emal Həcmi (ton)': 'sum'
    }).reset_index()
    
    # 4. Catalyst performance analysis
    datasets['catalyst_performance'] = df.groupby('İstifadə Edilən Katalizatorlar').agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Energy_per_ton': 'mean',
        'CO2_per_ton': 'mean',
        'Has_Incident': 'mean',
        'Process_KPI_Score': 'mean',
        'Emal Həcmi (ton)': 'sum'
    }).reset_index()
    
    # 5. Duration category analysis
    datasets['duration_analysis'] = df.groupby('Duration_Category').agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Has_Incident': 'mean',
        'Energy_per_ton': 'mean',
        'Process_KPI_Score': 'mean',
        'Emal Həcmi (ton)': 'sum'
    }).reset_index()
    
    # 6. Temperature category analysis
    datasets['temperature_analysis'] = df.groupby('Temperature_Category').agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Has_Incident': 'mean',
        'Energy_per_ton': 'mean',
        'Process_KPI_Score': 'mean',
        'Emal Həcmi (ton)': 'sum'
    }).reset_index()
    
    # 7. Pressure category analysis
    datasets['pressure_analysis'] = df.groupby('Pressure_Category').agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Has_Incident': 'mean',
        'Energy_per_ton': 'mean',
        'Process_KPI_Score': 'mean',
        'Emal Həcmi (ton)': 'sum'
    }).reset_index()
    
    # 8. Supplier performance analysis
    if 'Təchizatçı Adı' in df.columns:
        datasets['supplier_analysis'] = df.groupby('Təchizatçı Adı').agg({
            'Emalın Səmərəliliyi (%)': 'mean',
            'Has_Incident': 'mean',
            'Energy_per_ton': 'mean',
            'Process_KPI_Score': 'mean',
            'Emal Həcmi (ton)': 'sum'
        }).reset_index()
    
    # 9. Safety risk analysis by parameter combinations
    datasets['safety_parameters'] = df.groupby(['Temperature_Category', 'Pressure_Category']).agg({
        'Has_Incident': 'mean',
        'Təhlükəsizlik Hadisələri': 'sum',
        'Emal Həcmi (ton)': 'sum'
    }).reset_index()
    
    # 10. Optimal conditions for each process type
    optimal_conditions = []
    for process_type in df['Proses Tipi'].unique():
        process_data = df[df['Proses Tipi'] == process_type]
        
        # Get top 10% most efficient processes
        threshold = process_data['Process_KPI_Score'].quantile(0.9)
        top_processes = process_data[process_data['Process_KPI_Score'] >= threshold]
        
        # Extract optimal parameters
        optimal_conditions.append({
            'Proses Tipi': process_type,
            'Average_Efficiency': top_processes['Emalın Səmərəliliyi (%)'].mean(),
            'Optimal_Temperature': top_processes['Temperatur (°C)'].mean(),
            'Optimal_Pressure': top_processes['Təzyiq (bar)'].mean(),
            'Optimal_Duration': top_processes['Prosesin Müddəti (saat)'].mean(),
            'Incident_Rate': top_processes['Has_Incident'].mean(),
            'Energy_Usage': top_processes['Energy_per_ton'].mean(),
            'Process_Count': len(top_processes)
        })
    
    datasets['optimal_conditions'] = pd.DataFrame(optimal_conditions)
    
    # 11. ROI projection dataset
    # Calculate potential savings for each process
    df['Potential_Energy_Savings'] = 0.0
    df['Potential_Cost_Savings'] = 0.0
    
    for process_type in df['Proses Tipi'].unique():
        # Get best energy efficiency for this process type
        best_energy = df[df['Proses Tipi'] == process_type]['Energy_per_ton'].min()
        avg_energy = df[df['Proses Tipi'] == process_type]['Energy_per_ton'].mean()
        
        # Calculate potential savings
        df.loc[df['Proses Tipi'] == process_type, 'Potential_Energy_Savings'] = (
            (df.loc[df['Proses Tipi'] == process_type, 'Energy_per_ton'] - best_energy) / 
            df.loc[df['Proses Tipi'] == process_type, 'Energy_per_ton']
        ) * 100
        
        # Estimate cost savings (using a simplified model)
        df.loc[df['Proses Tipi'] == process_type, 'Potential_Cost_Savings'] = (
            df.loc[df['Proses Tipi'] == process_type, 'Potential_Energy_Savings'] * 0.3 +  # Energy component
            (1 - df.loc[df['Proses Tipi'] == process_type, 'Emalın Səmərəliliyi (%)'] / 100) * 100 * 0.4 +  # Efficiency component
            df.loc[df['Proses Tipi'] == process_type, 'Has_Incident'] * 10 * 0.3  # Safety component
        )
    
    # Create ROI projections
    datasets['roi_projections'] = df.groupby('Proses Tipi').agg({
        'Emal Həcmi (ton)': 'sum',
        'Əməliyyat Xərcləri (AZN)': 'sum',
        'Potential_Energy_Savings': 'mean',
        'Potential_Cost_Savings': 'mean'
    }).reset_index()
    
    # Calculate ROI metrics
    datasets['roi_projections']['Annual_Cost'] = datasets['roi_projections']['Əməliyyat Xərcləri (AZN)'] * 12  # Annualized
    datasets['roi_projections']['Potential_Annual_Savings'] = (
        datasets['roi_projections']['Annual_Cost'] * datasets['roi_projections']['Potential_Cost_Savings'] / 100
    )
    datasets['roi_projections']['Estimated_Investment'] = datasets['roi_projections']['Potential_Annual_Savings'] * 1.5  # Rough estimate
    datasets['roi_projections']['ROI_Months'] = datasets['roi_projections']['Estimated_Investment'] / (datasets['roi_projections']['Potential_Annual_Savings'] / 12)
    
    print(f"Created {len(datasets)} aggregated datasets")
    return datasets

def generate_exploratory_plots(df, output_dir):
    """
    Generate initial exploratory plots to understand the data
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set the style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Process efficiency by type
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Proses Tipi', y='Emalın Səmərəliliyi (%)', data=df, ci=None)
    plt.title('Average Process Efficiency by Type')
    plt.savefig(f"{output_dir}/process_efficiency_by_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Safety incidents by process type
    incident_by_type = df.groupby('Proses Tipi')['Has_Incident'].mean() * 100
    plt.figure(figsize=(10, 6))
    incident_by_type.plot(kind='bar')
    plt.title('Incident Rate by Process Type (%)')
    plt.ylabel('Incident Rate (%)')
    plt.savefig(f"{output_dir}/incident_rate_by_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Energy consumption vs efficiency
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Energy_per_ton', y='Emalın Səmərəliliyi (%)', 
                   hue='Proses Tipi', size='Emal Həcmi (ton)', 
                   data=df, sizes=(50, 200), alpha=0.7)
    plt.title('Energy Consumption vs. Process Efficiency')
    plt.xlabel('Energy per Ton (kWh/ton)')
    plt.ylabel('Efficiency (%)')
    plt.savefig(f"{output_dir}/energy_vs_efficiency.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Process KPI distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Proses Tipi', y='Process_KPI_Score', data=df)
    plt.title('Process KPI Score Distribution by Type')
    plt.savefig(f"{output_dir}/kpi_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Temperature vs Pressure heatmap for safety
    pivot = df.pivot_table(
        index='Temperature_Category', 
        columns='Pressure_Category', 
        values='Has_Incident',
        aggfunc='mean'
    ) * 100
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.title('Safety Incident Rate (%) by Temperature and Pressure')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temp_pressure_safety.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated exploratory plots in {output_dir}")

def save_datasets(datasets, output_dir):
    """
    Save all datasets to CSV files
    
    Args:
        datasets: Dictionary of DataFrames
        output_dir: Directory to save the files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save each dataset
    for name, dataset in datasets.items():
        output_path = f"{output_dir}/{name}.csv"
        dataset.to_csv(output_path, index=False)
    
    print(f"Saved {len(datasets)} datasets to {output_dir}")
    
def run_analysis(input_file, output_dir='socar-dashboard/data'):
    """
    Run the complete analysis pipeline
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save processed data
    """
    print(f"Starting SOCAR process analysis pipeline")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # 1. Load and validate data
    df = load_and_validate_data(input_file)
    
    # 2. Create derived metrics
    processed_df = create_derived_metrics(df)
    
    # 3. Create aggregated datasets
    datasets = create_aggregated_datasets(processed_df)
    
    # 4. Save processed data
    save_datasets(datasets, output_dir)
    processed_df.to_csv(f"{output_dir}/processed_data.csv", index=False)
    
    # 5. Generate exploratory plots
    generate_exploratory_plots(processed_df, f"{output_dir}/plots")
    
    print(f"Analysis completed successfully")
    return processed_df, datasets

if __name__ == "__main__":
    # Run the analysis
    run_analysis('data/data.csv', 'socar-dashboard/data')