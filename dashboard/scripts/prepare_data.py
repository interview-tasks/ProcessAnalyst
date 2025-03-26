import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import time
import yaml
import json
from functools import wraps
from typing import Dict, List, Tuple, Union, Optional
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_preparation")

def timer_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

class DataValidator:
    """Class to handle data validation and cleaning"""
    
    @staticmethod
    def check_missing_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
        """Check for missing required columns"""
        return [col for col in required_columns if col not in df.columns]
    
    @staticmethod
    def check_null_values(df: pd.DataFrame, critical_columns: List[str]) -> Dict[str, int]:
        """Check for null values in critical columns"""
        return {col: df[col].isnull().sum() for col in critical_columns if df[col].isnull().any()}
    
    @staticmethod
    def check_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
        """Check if columns have expected data types"""
        type_issues = {}
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = df[col].dtype.name
                if not actual_type.startswith(expected_type):
                    type_issues[col] = (actual_type, expected_type)
        return type_issues
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning"""
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values for numeric columns
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Replace with median for numeric columns if less than 5% are missing
            null_pct = df_clean[col].isnull().mean()
            if 0 < null_pct < 0.05:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Replace with mode for categorical columns if less than 5% are missing
            null_pct = df_clean[col].isnull().mean()
            if 0 < null_pct < 0.05:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Remove rows with too many missing values
        missing_threshold = 0.5  # If more than 50% of columns are missing, drop the row
        df_clean = df_clean.dropna(thresh=int(df_clean.shape[1] * (1 - missing_threshold)))
        
        # Check for and handle duplicates
        duplicate_count = df_clean.duplicated().sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows. Removing duplicates.")
            df_clean = df_clean.drop_duplicates()
        
        return df_clean

@timer_decorator
def load_and_validate_data(file_path: str, config: Dict) -> pd.DataFrame:
    """
    Load data with comprehensive validation and cleaning
    
    Args:
        file_path: Path to the CSV file
        config: Configuration dictionary with validation parameters
        
    Returns:
        Pandas DataFrame with cleaned data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        # Get file size for logging
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Loading data from {file_path} (Size: {file_size_mb:.2f} MB)")
        
        # Load data with optimized settings for larger files
        if file_size_mb > 100:  # If file is larger than 100MB
            # Use chunksize for large files to reduce memory usage
            chunk_size = 100000  # Adjust based on available memory
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded large file in chunks: {len(chunks)} chunks processed")
        else:
            # For smaller files, load directly
            df = pd.read_csv(file_path)
        
        # Basic validation
        required_columns = config.get('required_columns', [])
        critical_columns = config.get('critical_columns', [])
        expected_types = config.get('expected_column_types', {})
        
        validator = DataValidator()
        
        # Check for missing required columns
        missing_cols = validator.check_missing_columns(df, required_columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check for null values in critical columns
        null_counts = validator.check_null_values(df, critical_columns)
        if null_counts:
            logger.warning(f"Found null values in critical columns: {null_counts}")
        
        # Check data types
        type_issues = validator.check_data_types(df, expected_types)
        if type_issues:
            logger.warning(f"Column data type issues: {type_issues}")
            
            # Try to convert columns to expected types
            for col, (actual, expected) in type_issues.items():
                try:
                    if expected == 'int':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    elif expected == 'float':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Add more type conversions as needed
                except Exception as e:
                    logger.error(f"Failed to convert column {col} to {expected}: {e}")
        
        # Clean data
        df = validator.clean_data(df)
        
        # Basic statistics for logging
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
        # Log basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        stats = df[numeric_cols].describe().transpose()
        for col in numeric_cols[:5]:  # Log stats for first 5 numeric columns
            logger.info(f"Column {col}: min={stats.loc[col, 'min']:.2f}, mean={stats.loc[col, 'mean']:.2f}, max={stats.loc[col, 'max']:.2f}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise

@timer_decorator
def create_derived_metrics(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Create comprehensive derived metrics for process analysis
    
    Args:
        df: Input dataframe with raw process data
        config: Configuration dictionary with metrics parameters
        
    Returns:
        DataFrame with added derived metrics
    """
    # Create a copy to avoid modifying the original dataframe
    logger.info("Creating derived metrics...")
    processed_df = df.copy()
    
    # Get metric definitions from config
    metrics = config.get('derived_metrics', {})
    
    # Track created metrics
    created_metrics = []
    
    try:
        # Basic efficiency metrics
        if 'Energy_Efficiency' in metrics:
            processed_df['Energy_Efficiency'] = processed_df['Emal Həcmi (ton)'] / processed_df['Enerji İstifadəsi (kWh)']
            created_metrics.append('Energy_Efficiency')
        
        if 'Material_Productivity' in metrics:
            processed_df['Material_Productivity'] = processed_df['Emalın Səmərəliliyi (%)'] / 100  # Convert % to decimal
            created_metrics.append('Material_Productivity')
        
        if 'Cost_Efficiency' in metrics:
            processed_df['Cost_Efficiency'] = processed_df['Emal Həcmi (ton)'] / processed_df['Əməliyyat Xərcləri (AZN)']
            created_metrics.append('Cost_Efficiency')
        
        # Energy metrics
        if 'Energy_per_ton' in metrics:
            processed_df['Energy_per_ton'] = processed_df['Enerji İstifadəsi (kWh)'] / processed_df['Emal Həcmi (ton)']
            created_metrics.append('Energy_per_ton')
        
        # CO2 metrics
        if 'CO2_per_ton' in metrics:
            processed_df['CO2_per_ton'] = processed_df['Ətraf Mühitə Təsir (g CO2 ekvivalent)'] / processed_df['Emal Həcmi (ton)'] / 1000  # Convert to kg
            created_metrics.append('CO2_per_ton')
        
        # Cost metrics
        if 'Cost_per_ton' in metrics:
            processed_df['Cost_per_ton'] = processed_df['Əməliyyat Xərcləri (AZN)'] / processed_df['Emal Həcmi (ton)']
            created_metrics.append('Cost_per_ton')
        
        # Safety metrics
        if 'Safety_Risk' in metrics:
            processed_df['Safety_Risk'] = processed_df['Təhlükəsizlik Hadisələri'] / processed_df['Emal Həcmi (ton)'] * 1000  # Incidents per 1000 tons
            created_metrics.append('Safety_Risk')
        
        if 'Has_Incident' in metrics:
            processed_df['Has_Incident'] = (processed_df['Təhlükəsizlik Hadisələri'] > 0).astype(int)  # Binary indicator
            created_metrics.append('Has_Incident')
        
        # Environmental metrics
        if 'Environmental_Impact' in metrics:
            processed_df['Environmental_Impact'] = processed_df['Ətraf Mühitə Təsir (g CO2 ekvivalent)'] / processed_df['Emal Həcmi (ton)']
            created_metrics.append('Environmental_Impact')
        
        # Time efficiency metrics
        if 'Processing_Speed' in metrics:
            processed_df['Processing_Speed'] = processed_df['Emal Həcmi (ton)'] / processed_df['Prosesin Müddəti (saat)']  # Tons per hour
            created_metrics.append('Processing_Speed')
        
        # Resource efficiency 
        if 'Worker_Productivity' in metrics and 'İşçi Sayı' in processed_df.columns:
            processed_df['Worker_Productivity'] = processed_df['Emal Həcmi (ton)'] / processed_df['İşçi Sayı']  # Tons per worker
            created_metrics.append('Worker_Productivity')
        
        # Combined KPI score (weighted average of key metrics)
        if 'Process_KPI_Score' in metrics:
            # Get weights from config or use defaults
            weights = metrics.get('Process_KPI_Score', {}).get('weights', {
                'efficiency': 0.3,
                'energy': 0.2,
                'safety': 0.2,
                'cost': 0.2,
                'environmental': 0.1
            })
            
            # Normalize metrics to 0-1 scale
            if 'Material_Productivity' in processed_df.columns:
                efficiency_normalized = processed_df['Material_Productivity']
            else:
                efficiency_normalized = processed_df['Emalın Səmərəliliyi (%)'] / 100
                
            energy_normalized = 1 - (processed_df['Energy_per_ton'] / processed_df['Energy_per_ton'].max())
            safety_normalized = 1 - (processed_df['Safety_Risk'] / processed_df['Safety_Risk'].max() if 'Safety_Risk' in processed_df.columns 
                                     else processed_df['Təhlükəsizlik Hadisələri'] / processed_df['Təhlükəsizlik Hadisələri'].max())
            cost_normalized = 1 - (processed_df['Cost_per_ton'] / processed_df['Cost_per_ton'].max() if 'Cost_per_ton' in processed_df.columns
                                  else processed_df['Əməliyyat Xərcləri (AZN)'] / processed_df['Emal Həcmi (ton)'] / 
                                 (processed_df['Əməliyyat Xərcləri (AZN)'] / processed_df['Emal Həcmi (ton)']).max())
            env_normalized = 1 - (processed_df['CO2_per_ton'] / processed_df['CO2_per_ton'].max() if 'CO2_per_ton' in processed_df.columns
                                 else processed_df['Ətraf Mühitə Təsir (g CO2 ekvivalent)'] / processed_df['Emal Həcmi (ton)'] / 
                                (processed_df['Ətraf Mühitə Təsir (g CO2 ekvivalent)'] / processed_df['Emal Həcmi (ton)']).max())
            
            # Calculate weighted score
            processed_df['Process_KPI_Score'] = (
                weights['efficiency'] * efficiency_normalized + 
                weights['energy'] * energy_normalized +
                weights['safety'] * safety_normalized +
                weights['cost'] * cost_normalized +
                weights['environmental'] * env_normalized
            ) * 100  # Scale to 0-100
            
            created_metrics.append('Process_KPI_Score')
        
        # Process categorization
        # Duration categories
        if 'Duration_Category' in metrics:
            duration_bins = metrics.get('Duration_Category', {}).get('bins', [0, 8, 12, 16, float('inf')])
            duration_labels = metrics.get('Duration_Category', {}).get('labels', ['Short (<8h)', 'Medium (8-12h)', 'Long (12-16h)', 'Extended (>16h)'])
            
            processed_df['Duration_Category'] = pd.cut(
                processed_df['Prosesin Müddəti (saat)'], 
                bins=duration_bins, 
                labels=duration_labels
            )
            created_metrics.append('Duration_Category')
        
        # Temperature categories
        if 'Temperature_Category' in metrics:
            temp_bins = metrics.get('Temperature_Category', {}).get('bins', [0, 150, 300, 450, float('inf')])
            temp_labels = metrics.get('Temperature_Category', {}).get('labels', ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)'])
            
            processed_df['Temperature_Category'] = pd.cut(
                processed_df['Temperatur (°C)'], 
                bins=temp_bins, 
                labels=temp_labels
            )
            created_metrics.append('Temperature_Category')
        
        # Pressure categories
        if 'Pressure_Category' in metrics:
            pressure_bins = metrics.get('Pressure_Category', {}).get('bins', [0, 10, 30, 50, float('inf')])
            pressure_labels = metrics.get('Pressure_Category', {}).get('labels', ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)'])
            
            processed_df['Pressure_Category'] = pd.cut(
                processed_df['Təzyiq (bar)'], 
                bins=pressure_bins, 
                labels=pressure_labels
            )
            created_metrics.append('Pressure_Category')
        
        # Add any custom metrics from the configuration
        for metric_name, formula in metrics.get('custom_metrics', {}).items():
            try:
                # This is a simplified approach - in a real application,
                # you'd want a more robust way to evaluate formulas
                processed_df[metric_name] = eval(formula, {"df": processed_df, "np": np})
                created_metrics.append(metric_name)
            except Exception as e:
                logger.error(f"Failed to create custom metric {metric_name}: {e}")
        
        logger.info(f"Created {len(created_metrics)} new derived metrics: {', '.join(created_metrics)}")
        return processed_df
    
    except Exception as e:
        logger.error(f"Error creating derived metrics: {e}", exc_info=True)
        raise

@timer_decorator
def create_aggregated_datasets(df: pd.DataFrame, config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Create multiple aggregated datasets for different analysis views
    
    Args:
        df: Processed DataFrame with derived metrics
        config: Configuration dictionary with aggregation parameters
        
    Returns:
        Dictionary of DataFrames with various aggregations
    """
    logger.info("Creating aggregated datasets...")
    datasets = {}
    aggregations = config.get('aggregations', {})
    
    try:
        # Process each aggregation defined in the config
        for agg_name, agg_config in aggregations.items():
            group_by_cols = agg_config.get('group_by', [])
            if not group_by_cols or not all(col in df.columns for col in group_by_cols):
                logger.warning(f"Skipping aggregation {agg_name}: Missing group by columns")
                continue
                
            metrics = agg_config.get('metrics', {})
            if not metrics:
                logger.warning(f"Skipping aggregation {agg_name}: No metrics defined")
                continue
                
            # Build aggregation dictionary
            agg_dict = {}
            for col, agg_funcs in metrics.items():
                if col in df.columns:
                    agg_dict[col] = agg_funcs
            
            if not agg_dict:
                logger.warning(f"Skipping aggregation {agg_name}: No valid metrics found")
                continue
                
            # Perform aggregation
            try:
                grouped = df.groupby(group_by_cols).agg(agg_dict)
                
                # Flatten multi-level columns if needed
                if isinstance(grouped.columns, pd.MultiIndex):
                    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
                
                # Reset index for easier handling
                grouped = grouped.reset_index()
                
                # Add to datasets
                datasets[agg_name] = grouped
                logger.info(f"Created aggregation '{agg_name}': {grouped.shape[0]} rows, {grouped.shape[1]} columns")
            except Exception as e:
                logger.error(f"Failed to create aggregation {agg_name}: {e}")
        
        # Create some standard aggregations if not already defined
        
        # 1. Process Type aggregation
        if 'process_types' not in datasets:
            std_agg_cols = ['Emalın Səmərəliliyi (%)', 'Təhlükəsizlik Hadisələri', 'Emal Həcmi (ton)']
            for col in ['Energy_per_ton', 'CO2_per_ton', 'Cost_per_ton', 'Process_KPI_Score']:
                if col in df.columns:
                    std_agg_cols.append(col)
                    
            agg_dict = {col: ['mean'] if col != 'Emal Həcmi (ton)' and col != 'Təhlükəsizlik Hadisələri' else ['sum'] 
                        for col in std_agg_cols if col in df.columns}
            
            if 'Has_Incident' in df.columns:
                agg_dict['Has_Incident'] = ['mean']
                
            datasets['process_types'] = df.groupby('Proses Tipi').agg(agg_dict)
            
            # Flatten columns if needed
            if isinstance(datasets['process_types'].columns, pd.MultiIndex):
                datasets['process_types'].columns = ['_'.join(col).strip() for col in datasets['process_types'].columns.values]
                
            datasets['process_types'] = datasets['process_types'].reset_index()
            logger.info(f"Created standard aggregation 'process_types': {datasets['process_types'].shape[0]} rows")
        
        # 10. Optimal conditions for each process type
        if 'optimal_conditions' not in datasets and 'Process_KPI_Score' in df.columns:
            optimal_conditions = []
            for process_type in df['Proses Tipi'].unique():
                process_data = df[df['Proses Tipi'] == process_type]
                
                # Get top 10% most efficient processes
                threshold = process_data['Process_KPI_Score'].quantile(0.9)
                top_processes = process_data[process_data['Process_KPI_Score'] >= threshold]
                
                if not top_processes.empty:
                    # Extract optimal parameters
                    condition = {
                        'Proses Tipi': process_type,
                        'Average_Efficiency': top_processes['Emalın Səmərəliliyi (%)'].mean(),
                        'Optimal_Temperature': top_processes['Temperatur (°C)'].mean(),
                        'Optimal_Pressure': top_processes['Təzyiq (bar)'].mean(),
                        'Optimal_Duration': top_processes['Prosesin Müddəti (saat)'].mean(),
                        'Process_Count': len(top_processes)
                    }
                    
                    # Add optional metrics if available
                    if 'Has_Incident' in top_processes.columns:
                        condition['Incident_Rate'] = top_processes['Has_Incident'].mean()
                    
                    if 'Energy_per_ton' in top_processes.columns:
                        condition['Energy_Usage'] = top_processes['Energy_per_ton'].mean()
                        
                    optimal_conditions.append(condition)
            
            if optimal_conditions:
                datasets['optimal_conditions'] = pd.DataFrame(optimal_conditions)
                logger.info(f"Created 'optimal_conditions' dataset with {len(optimal_conditions)} process types")
        
        logger.info(f"Created {len(datasets)} aggregated datasets")
        return datasets
    
    except Exception as e:
        logger.error(f"Error creating aggregated datasets: {e}", exc_info=True)
        raise

@timer_decorator
def generate_exploratory_plots(df: pd.DataFrame, output_dir: str, config: Dict) -> None:
    """
    Generate initial exploratory plots to understand the data
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save plots
        config: Configuration dictionary with plot parameters
    """
    logger.info(f"Generating exploratory plots in {output_dir}...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set the style
    plot_style = config.get('plot_style', 'whitegrid')
    plot_context = config.get('plot_context', 'notebook')
    font_scale = config.get('font_scale', 1.2)
    
    sns.set_style(plot_style)
    sns.set_context(plot_context, font_scale=font_scale)
    
    # Set default figure size
    plt.rcParams["figure.figsize"] = (12, 8)
    
    # Get plots to generate
    plots = config.get('plots', {})
    generated_plots = []
    
    try:
        # 1. Process efficiency by type
        if 'process_efficiency_by_type' in plots:
            plt.figure()
            ax = sns.barplot(x='Proses Tipi', y='Emalın Səmərəliliyi (%)', data=df, errorbar=('ci', 95))
            plt.title('Average Process Efficiency by Type', fontsize=16)
            plt.xlabel('Process Type', fontsize=14)
            plt.ylabel('Efficiency (%)', fontsize=14)
            plt.ylim(85, 100)  # Focus on the relevant range
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.1f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha = 'center', va = 'bottom',
                            fontsize=12)
                
            plt.tight_layout()
            plt.savefig(f"{output_dir}/process_efficiency_by_type.png", dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append('process_efficiency_by_type')
        
        # 2. Safety incidents by process type
        if 'incident_rate_by_type' in plots:
            plt.figure()
            
            if 'Has_Incident' in df.columns:
                incident_by_type = df.groupby('Proses Tipi')['Has_Incident'].mean() * 100
                ax = incident_by_type.plot(kind='bar', color='crimson')
                plt.title('Incident Rate by Process Type (%)', fontsize=16)
                plt.ylabel('Incident Rate (%)', fontsize=14)
                plt.xlabel('Process Type', fontsize=14)
                
                # Add value labels
                for i, v in enumerate(incident_by_type):
                    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=12)
            else:
                incident_by_type = df.groupby('Proses Tipi')['Təhlükəsizlik Hadisələri'].sum()
                volume_by_type = df.groupby('Proses Tipi')['Emal Həcmi (ton)'].sum()
                incident_rate = (incident_by_type / volume_by_type * 1000).sort_values(ascending=False)
                
                ax = incident_rate.plot(kind='bar', color='crimson')
                plt.title('Incidents per 1000 Tons by Process Type', fontsize=16)
                plt.ylabel('Incidents per 1000 Tons', fontsize=14)
                plt.xlabel('Process Type', fontsize=14)
                
                # Add value labels
                for i, v in enumerate(incident_rate):
                    ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/incident_rate_by_type.png", dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append('incident_rate_by_type')
        
        # 3. Energy consumption vs efficiency
        if 'energy_vs_efficiency' in plots and 'Energy_per_ton' in df.columns:
            plt.figure()
            scatter = sns.scatterplot(x='Energy_per_ton', y='Emalın Səmərəliliyi (%)', 
                                     hue='Proses Tipi', size='Emal Həcmi (ton)', 
                                     sizes=(50, 500), alpha=0.7, data=df)
            
            # Add a regression line
            sns.regplot(x='Energy_per_ton', y='Emalın Səmərəliliyi (%)', 
                       data=df, scatter=False, ci=95, line_kws={"color": "red", "lw": 2})
            
            # Calculate correlation
            corr = df['Energy_per_ton'].corr(df['Emalın Səmərəliliyi (%)'])
            
            plt.title(f'Energy Consumption vs. Process Efficiency (r = {corr:.2f})', fontsize=16)
            plt.xlabel('Energy per Ton (kWh/ton)', fontsize=14)
            plt.ylabel('Efficiency (%)', fontsize=14)
            
            # Annotate areas of interest
            if df['Energy_per_ton'].max() > 2 and df['Emalın Səmərəliliyi (%)'].min() < 90:
                plt.annotate('High Energy, Low Efficiency\n(Improvement Target)', 
                             xy=(df['Energy_per_ton'].quantile(0.8), df['Emalın Səmərəliliyi (%)'].quantile(0.2)),
                             xytext=(df['Energy_per_ton'].quantile(0.9), df['Emalın Səmərəliliyi (%)'].quantile(0.1)),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                             fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/energy_vs_efficiency.png", dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append('energy_vs_efficiency')
        
        # 4. Process KPI distribution
        if 'kpi_distribution' in plots and 'Process_KPI_Score' in df.columns:
            plt.figure()
            
            # Create violin plot with box plot inside
            ax = sns.violinplot(x='Proses Tipi', y='Process_KPI_Score', data=df, inner='box', palette='viridis')
            
            plt.title('Process KPI Score Distribution by Type', fontsize=16)
            plt.xlabel('Process Type', fontsize=14)
            plt.ylabel('KPI Score (0-100)', fontsize=14)
            
            # Add mean markers
            means = df.groupby('Proses Tipi')['Process_KPI_Score'].mean()
            
            for i, mean_val in enumerate(means):
                ax.text(i, mean_val + 1, f'Mean: {mean_val:.1f}', ha='center', fontsize=10, color='darkred')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/kpi_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append('kpi_distribution')
        
        # 5. Temperature vs Pressure heatmap for safety
        if 'temp_pressure_safety' in plots and 'Temperature_Category' in df.columns and 'Pressure_Category' in df.columns:
            plt.figure()
            
            # Choose incident measure
            incident_col = 'Has_Incident' if 'Has_Incident' in df.columns else 'Təhlükəsizlik Hadisələri'
            
            # Create pivot table
            pivot = df.pivot_table(
                index='Temperature_Category', 
                columns='Pressure_Category', 
                values=incident_col,
                aggfunc='mean'
            )
            
            # Scale to percentage if using Has_Incident
            if incident_col == 'Has_Incident':
                pivot = pivot * 100
                title_suffix = 'Incident Rate (%)'
            else:
                title_suffix = 'Average Incidents'
            
            # Try to reorder categories if they're present
            try:
                temp_order = ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)']
                pressure_order = ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)']
                
                pivot = pivot.reindex(temp_order, axis=0)
                pivot = pivot.reindex(pressure_order, axis=1)
            except:
                logger.info("Could not reorder category levels for temperature/pressure heatmap")
            
            # Create heatmap with annotations
            sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.1f',
                       linewidths=0.5, cbar_kws={'label': title_suffix})
            
            plt.title(f'Safety {title_suffix} by Temperature and Pressure', fontsize=16)
            plt.xlabel('Pressure Category', fontsize=14)
            plt.ylabel('Temperature Category', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/temp_pressure_safety.png", dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append('temp_pressure_safety')
        
        # 6. Additional plots - Catalyst performance if available
        if 'catalyst_performance' in plots and 'İstifadə Edilən Katalizatorlar' in df.columns:
            plt.figure(figsize=(14, 8))
            
            # Calculate average efficiency by catalyst
            cat_efficiency = df.groupby('İstifadə Edilən Katalizatorlar')['Emalın Səmərəliliyi (%)'].mean().sort_values(ascending=False)
            
            # Create bar plot
            ax = sns.barplot(x=cat_efficiency.index, y=cat_efficiency.values, palette='viridis')
            
            plt.title('Catalyst Performance Comparison', fontsize=16)
            plt.xlabel('Catalyst Type', fontsize=14)
            plt.ylabel('Average Efficiency (%)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for i, v in enumerate(cat_efficiency):
                ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/catalyst_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append('catalyst_performance')
            
        # 7. Correlation matrix for key parameters
        if 'parameter_correlation' in plots:
            # Select relevant numeric columns
            numeric_cols = ['Temperatur (°C)', 'Təzyiq (bar)', 'Prosesin Müddəti (saat)',
                          'Emalın Səmərəliliyi (%)']
            
            # Add derived metrics if available
            for col in ['Energy_per_ton', 'CO2_per_ton', 'Cost_per_ton', 'Safety_Risk']:
                if col in df.columns:
                    numeric_cols.append(col)
            
            corr_matrix = df[numeric_cols].corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                       annot=True, square=True, linewidths=0.5, cbar_kws={"shrink": .5})
            
            plt.title('Parameter Correlation Matrix', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/parameter_correlation.png", dpi=300, bbox_inches='tight')
            plt.close()
            generated_plots.append('parameter_correlation')
        
        logger.info(f"Generated {len(generated_plots)} exploratory plots: {', '.join(generated_plots)}")
    
    except Exception as e:
        logger.error(f"Error generating exploratory plots: {e}", exc_info=True)
        raise

@timer_decorator
def save_datasets(datasets: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Save all datasets to CSV files with error handling
    
    Args:
        datasets: Dictionary of DataFrames
        output_dir: Directory to save the files
    """
    logger.info(f"Saving datasets to {output_dir}...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save each dataset
    for name, dataset in datasets.items():
        try:
            output_path = f"{output_dir}/{name}.csv"
            dataset.to_csv(output_path, index=False)
            logger.info(f"Saved dataset '{name}' ({dataset.shape[0]} rows, {dataset.shape[1]} columns) to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset '{name}': {e}")
    
    logger.info(f"Saved {len(datasets)} datasets to {output_dir}")
    
def load_config(config_file: str = 'data_config.yaml') -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to the YAML configuration file
        
    Returns:
        Configuration as a dictionary
    """
    # Default configuration
    default_config = {
        'required_columns': [
            'Proses ID', 'Proses Tipi', 'Proses Addımı', 'Emal Həcmi (ton)',
            'Temperatur (°C)', 'Təzyiq (bar)', 'Prosesin Müddəti (saat)', 
            'Emalın Səmərəliliyi (%)', 'Enerji İstifadəsi (kWh)',
            'Ətraf Mühitə Təsir (g CO2 ekvivalent)', 'Təhlükəsizlik Hadisələri',
            'Əməliyyat Xərcləri (AZN)'
        ],
        'critical_columns': [
            'Proses Tipi', 'Proses Addımı', 'Emal Həcmi (ton)', 'Enerji İstifadəsi (kWh)'
        ],
        'expected_column_types': {
            'Emal Həcmi (ton)': 'float',
            'Temperatur (°C)': 'float',
            'Təzyiq (bar)': 'float',
            'Prosesin Müddəti (saat)': 'float',
            'Emalın Səmərəliliyi (%)': 'float',
            'Enerji İstifadəsi (kWh)': 'float',
            'Ətraf Mühitə Təsir (g CO2 ekvivalent)': 'float',
            'Təhlükəsizlik Hadisələri': 'float',
            'Əməliyyat Xərcləri (AZN)': 'float'
        },
        'derived_metrics': {
            'Energy_Efficiency': {},
            'Material_Productivity': {},
            'Cost_Efficiency': {},
            'Energy_per_ton': {},
            'CO2_per_ton': {},
            'Cost_per_ton': {},
            'Safety_Risk': {},
            'Has_Incident': {},
            'Environmental_Impact': {},
            'Processing_Speed': {},
            'Worker_Productivity': {},
            'Process_KPI_Score': {
                'weights': {
                    'efficiency': 0.3,
                    'energy': 0.2,
                    'safety': 0.2,
                    'cost': 0.2,
                    'environmental': 0.1
                }
            },
            'Duration_Category': {
                'bins': [0, 8, 12, 16, float('inf')],
                'labels': ['Short (<8h)', 'Medium (8-12h)', 'Long (12-16h)', 'Extended (>16h)']
            },
            'Temperature_Category': {
                'bins': [0, 150, 300, 450, float('inf')],
                'labels': ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)']
            },
            'Pressure_Category': {
                'bins': [0, 10, 30, 50, float('inf')],
                'labels': ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)']
            }
        },
        'aggregations': {
            'process_types': {
                'group_by': ['Proses Tipi'],
                'metrics': {
                    'Emalın Səmərəliliyi (%)': ['mean'],
                    'Enerji İstifadəsi (kWh)': ['sum'],
                    'Təhlükəsizlik Hadisələri': ['sum'],
                    'Emal Həcmi (ton)': ['sum'],
                    'Energy_per_ton': ['mean'],
                    'Has_Incident': ['mean']
                }
            },
            'process_steps': {
                'group_by': ['Proses Addımı'],
                'metrics': {
                    'Emalın Səmərəliliyi (%)': ['mean'],
                    'Temperatur (°C)': ['mean'],
                    'Təzyiq (bar)': ['mean'],
                    'Prosesin Müddəti (saat)': ['mean'],
                    'Təhlükəsizlik Hadisələri': ['sum'],
                    'Emal Həcmi (ton)': ['sum'],
                    'Energy_per_ton': ['mean']
                }
            },
            'process_type_step': {
                'group_by': ['Proses Tipi', 'Proses Addımı'],
                'metrics': {
                    'Emalın Səmərəliliyi (%)': ['mean'],
                    'Energy_per_ton': ['mean'],
                    'Has_Incident': ['mean'],
                    'Emal Həcmi (ton)': ['sum']
                }
            },
            'safety_parameters': {
                'group_by': ['Temperature_Category', 'Pressure_Category'],
                'metrics': {
                    'Has_Incident': ['mean'],
                    'Təhlükəsizlik Hadisələri': ['sum'],
                    'Emal Həcmi (ton)': ['sum']
                }
            }
        },
        'plots': {
            'process_efficiency_by_type': True,
            'incident_rate_by_type': True,
            'energy_vs_efficiency': True,
            'kpi_distribution': True,
            'temp_pressure_safety': True,
            'catalyst_performance': True,
            'parameter_correlation': True
        },
        'plot_style': 'whitegrid',
        'plot_context': 'notebook',
        'font_scale': 1.2
    }
    
    # Try to load configuration from file
    config = default_config
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Update default config with file config
            if file_config:
                # Deep merge
                def deep_update(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                            deep_update(d[k], v)
                        else:
                            d[k] = v
                
                deep_update(config, file_config)
                logger.info(f"Loaded configuration from {config_file}")
            else:
                logger.warning(f"Configuration file {config_file} is empty, using defaults")
        else:
            logger.info(f"Configuration file {config_file} not found, using defaults")
            
            # Save default config as template
            # with open(f"{config_file}.template", 'w') as f:
            #     yaml.dump(default_config, f, default_flow_style=False)
            # logger.info(f"Saved default configuration template to {config_file}.template")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
    
    return config

@timer_decorator    
def run_analysis(input_file: str, output_dir: str = 'dashboard/data', config_file: str = 'data_config.yaml') -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run the complete analysis pipeline
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save processed data
        config_file: Path to the configuration file
        
    Returns:
        Tuple containing processed DataFrame and dictionary of aggregated datasets
    """
    start_time = time.time()
    logger.info(f"Starting SOCAR process analysis pipeline")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # 1. Load configuration
        config = load_config(config_file)
        
        # 2. Load and validate data
        df = load_and_validate_data(input_file, config)
        
        # 3. Create derived metrics
        processed_df = create_derived_metrics(df, config)
        
        # 4. Create aggregated datasets
        datasets = create_aggregated_datasets(processed_df, config)
        
        # 5. Save processed data
        processed_df.to_csv(f"{output_dir}/processed_data.csv", index=False)
        logger.info(f"Saved processed data ({processed_df.shape[0]} rows, {processed_df.shape[1]} columns) to {output_dir}/processed_data.csv")
        
        # 6. Save aggregated datasets
        save_datasets(datasets, output_dir)
        
        # 7. Generate exploratory plots
        plots_dir = f"{output_dir}/plots"
        generate_exploratory_plots(processed_df, plots_dir, config)
        
        # 8. Save analysis metadata
        metadata = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_file': input_file,
            'output_directory': output_dir,
            'data_rows': processed_df.shape[0],
            'data_columns': processed_df.shape[1],
            'aggregated_datasets': list(datasets.keys()),
            'derived_metrics': [col for col in processed_df.columns if col not in df.columns],
            'execution_time_seconds': round(time.time() - start_time, 2)
        }
        
        with open(f"{output_dir}/analysis_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Analysis completed successfully in {time.time() - start_time:.2f} seconds")
        return processed_df, datasets
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SOCAR Process Analysis Data Preparation')
    parser.add_argument('--input', '-i', type=str, default='dashboard/data/data.csv',
                       help='Path to the input CSV file')
    parser.add_argument('--output', '-o', type=str, default='dashboard/data',
                       help='Directory to save processed data')
    parser.add_argument('--config', '-c', type=str, default='data_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Run the analysis
    run_analysis(args.input, args.output, args.config)