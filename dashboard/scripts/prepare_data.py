#!/usr/bin/env python
"""
SOCAR Process Analysis Data Preparation

This script prepares raw data for the process analysis dashboard by:
1. Loading and validating the raw data
2. Creating derived metrics
3. Generating aggregated datasets
4. Saving processed data files

Usage:
    python prepare_data.py [options]
    
python dashboard/scripts/prepare_data.py --input dashboard/data/data.csv --output dashboard/data --no-plots
"""

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

def setup_logging(log_to_file=False, debug=False):
    """Set up logging with optional file output"""
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler("data_preparation.log"))
    
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger("data_preparation")

# Initialize logger with console-only output by default
logger = setup_logging()

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
        return {col: df[col].isnull().sum() for col in critical_columns if col in df.columns and df[col].isnull().any()}
    
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

def load_config(config_file: str = 'data_config.yaml', create_template: bool = False) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to the YAML configuration file
        create_template: Whether to create a template file if config file doesn't exist
        
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
            'Energy_per_ton': {},
            'CO2_per_ton': {},
            'Cost_per_ton': {},
            'Safety_Risk': {},
            'Has_Incident': {},
            'Process_KPI_Score': {
                'weights': {
                    'efficiency': 0.3,
                    'energy': 0.2,
                    'safety': 0.2,
                    'cost': 0.2,
                    'environmental': 0.1
                }
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
                    'Energy_per_ton': ['mean']
                }
            },
            'process_steps': {
                'group_by': ['Proses Addımı'],
                'metrics': {
                    'Emalın Səmərəliliyi (%)': ['mean'],
                    'Təhlükəsizlik Hadisələri': ['sum'],
                    'Emal Həcmi (ton)': ['sum']
                }
            },
            'safety_parameters': {
                'group_by': ['Temperature_Category', 'Pressure_Category'],
                'metrics': {
                    'Təhlükəsizlik Hadisələri': ['sum'],
                    'Emal Həcmi (ton)': ['sum']
                }
            }
        }
    }
    
    # Try to load configuration from file
    config = default_config.copy()
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
            
            # Only create template if explicitly requested
            if create_template:
                with open(f"{config_file}", 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created configuration template: {config_file}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
    
    return config

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
        
        try:
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
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file {file_path} is empty")
        except pd.errors.ParserError:
            raise ValueError(f"Could not parse {file_path} as CSV - check file format")
        
        # Basic validation
        required_columns = config.get('required_columns', [])
        critical_columns = config.get('critical_columns', [])
        expected_types = config.get('expected_column_types', {})
        
        validator = DataValidator()
        
        # Check for missing required columns
        missing_cols = validator.check_missing_columns(df, required_columns)
        if missing_cols:
            # Log what columns we found for debugging
            logger.debug(f"Found columns: {list(df.columns)}")
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
        
        # Log basic statistics
        logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load data from {file_path}: {str(e)}")

@timer_decorator
def create_derived_metrics(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Create derived metrics for process analysis
    
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
        # Energy metrics
        if 'Energy_per_ton' in metrics and 'Enerji İstifadəsi (kWh)' in df.columns and 'Emal Həcmi (ton)' in df.columns:
            processed_df['Energy_per_ton'] = processed_df['Enerji İstifadəsi (kWh)'] / processed_df['Emal Həcmi (ton)']
            created_metrics.append('Energy_per_ton')
        
        # CO2 metrics
        if 'CO2_per_ton' in metrics and 'Ətraf Mühitə Təsir (g CO2 ekvivalent)' in df.columns and 'Emal Həcmi (ton)' in df.columns:
            processed_df['CO2_per_ton'] = processed_df['Ətraf Mühitə Təsir (g CO2 ekvivalent)'] / processed_df['Emal Həcmi (ton)'] / 1000  # Convert to kg
            created_metrics.append('CO2_per_ton')
        
        # Cost metrics
        if 'Cost_per_ton' in metrics and 'Əməliyyat Xərcləri (AZN)' in df.columns and 'Emal Həcmi (ton)' in df.columns:
            processed_df['Cost_per_ton'] = processed_df['Əməliyyat Xərcləri (AZN)'] / processed_df['Emal Həcmi (ton)']
            created_metrics.append('Cost_per_ton')
        
        # Safety metrics
        if 'Safety_Risk' in metrics and 'Təhlükəsizlik Hadisələri' in df.columns and 'Emal Həcmi (ton)' in df.columns:
            processed_df['Safety_Risk'] = processed_df['Təhlükəsizlik Hadisələri'] / processed_df['Emal Həcmi (ton)'] * 1000  # Incidents per 1000 tons
            created_metrics.append('Safety_Risk')
        
        if 'Has_Incident' in metrics and 'Təhlükəsizlik Hadisələri' in df.columns:
            processed_df['Has_Incident'] = (processed_df['Təhlükəsizlik Hadisələri'] > 0).astype(int)  # Binary indicator
            created_metrics.append('Has_Incident')
        
        # Process categorization - Temperature categories
        if 'Temperature_Category' in metrics and 'Temperatur (°C)' in df.columns:
            temp_config = metrics['Temperature_Category']
            temp_bins = temp_config.get('bins', [0, 150, 300, 450, float('inf')])
            temp_labels = temp_config.get('labels', ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)'])
            
            processed_df['Temperature_Category'] = pd.cut(
                processed_df['Temperatur (°C)'], 
                bins=temp_bins, 
                labels=temp_labels
            )
            created_metrics.append('Temperature_Category')
        
        # Process categorization - Pressure categories
        if 'Pressure_Category' in metrics and 'Təzyiq (bar)' in df.columns:
            pressure_config = metrics['Pressure_Category']
            pressure_bins = pressure_config.get('bins', [0, 10, 30, 50, float('inf')])
            pressure_labels = pressure_config.get('labels', ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)'])
            
            processed_df['Pressure_Category'] = pd.cut(
                processed_df['Təzyiq (bar)'], 
                bins=pressure_bins, 
                labels=pressure_labels
            )
            created_metrics.append('Pressure_Category')
        
        # Combined KPI score
        if 'Process_KPI_Score' in metrics and 'Emalın Səmərəliliyi (%)' in df.columns:
            try:
                # Get weights
                weights = metrics.get('Process_KPI_Score', {}).get('weights', {
                    'efficiency': 0.3,
                    'energy': 0.2,
                    'safety': 0.2,
                    'cost': 0.2,
                    'environmental': 0.1
                })
                
                # Start with efficiency (always available)
                kpi_score = weights['efficiency'] * (processed_df['Emalın Səmərəliliyi (%)'] / 100)
                components = 1
                
                # Add energy component if available
                if 'Energy_per_ton' in processed_df.columns:
                    # Normalize: lower is better
                    max_energy = processed_df['Energy_per_ton'].max()
                    if max_energy > 0:
                        energy_score = 1 - (processed_df['Energy_per_ton'] / max_energy)
                        kpi_score += weights['energy'] * energy_score
                        components += 1
                
                # Add safety component if available
                if 'Safety_Risk' in processed_df.columns:
                    # Normalize: lower is better
                    max_risk = processed_df['Safety_Risk'].max()
                    if max_risk > 0:
                        safety_score = 1 - (processed_df['Safety_Risk'] / max_risk)
                        kpi_score += weights['safety'] * safety_score
                        components += 1
                
                # Add cost component if available
                if 'Cost_per_ton' in processed_df.columns:
                    # Normalize: lower is better
                    max_cost = processed_df['Cost_per_ton'].max()
                    if max_cost > 0:
                        cost_score = 1 - (processed_df['Cost_per_ton'] / max_cost)
                        kpi_score += weights['cost'] * cost_score
                        components += 1
                
                # Scale to account for missing components
                if components < 4:
                    kpi_score = kpi_score * (4 / components)
                
                # Scale to 0-100
                processed_df['Process_KPI_Score'] = kpi_score * 100
                created_metrics.append('Process_KPI_Score')
            except Exception as e:
                logger.warning(f"Could not calculate Process_KPI_Score: {e}")
        
        logger.info(f"Created {len(created_metrics)} derived metrics: {', '.join(created_metrics)}")
        return processed_df
    
    except Exception as e:
        logger.error(f"Error creating derived metrics: {e}", exc_info=True)
        # Return the original dataframe with any metrics that were successfully created
        logger.warning("Returning partially processed data with metrics that were successfully created")
        return processed_df

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
            try:
                group_by_cols = agg_config.get('group_by', [])
                if not group_by_cols or not all(col in df.columns for col in group_by_cols):
                    logger.warning(f"Skipping aggregation {agg_name}: Missing group by columns")
                    continue
                    
                metrics = agg_config.get('metrics', {})
                if not metrics:
                    logger.warning(f"Skipping aggregation {agg_name}: No metrics defined")
                    continue
                    
                # Build aggregation dictionary with only available columns
                agg_dict = {}
                for col, agg_funcs in metrics.items():
                    if col in df.columns:
                        agg_dict[col] = agg_funcs
                
                if not agg_dict:
                    logger.warning(f"Skipping aggregation {agg_name}: No valid metrics found")
                    continue
                    
                # Perform aggregation
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
        
        # Create standard process_types aggregation if not already defined
        if 'process_types' not in datasets and 'Proses Tipi' in df.columns:
            try:
                # Basic aggregation by process type
                std_agg_cols = ['Emalın Səmərəliliyi (%)', 'Emal Həcmi (ton)']
                
                # Add derived metrics if available
                for col in ['Energy_per_ton', 'Safety_Risk', 'Process_KPI_Score']:
                    if col in df.columns:
                        std_agg_cols.append(col)
                
                # Filter to available columns
                std_agg_cols = [col for col in std_agg_cols if col in df.columns]
                
                if std_agg_cols:
                    agg_dict = {col: ['mean'] if col != 'Emal Həcmi (ton)' else ['sum'] for col in std_agg_cols}
                    
                    if 'Təhlükəsizlik Hadisələri' in df.columns:
                        agg_dict['Təhlükəsizlik Hadisələri'] = ['sum']
                    
                    process_types = df.groupby('Proses Tipi').agg(agg_dict)
                    
                    # Flatten columns if needed
                    if isinstance(process_types.columns, pd.MultiIndex):
                        process_types.columns = ['_'.join(col).strip() for col in process_types.columns.values]
                    
                    process_types = process_types.reset_index()
                    datasets['process_types'] = process_types
                    logger.info(f"Created standard aggregation 'process_types': {process_types.shape[0]} rows")
            except Exception as e:
                logger.error(f"Failed to create standard process_types aggregation: {e}")
        
        logger.info(f"Created {len(datasets)} aggregated datasets")
        return datasets
    
    except Exception as e:
        logger.error(f"Error creating aggregated datasets: {e}", exc_info=True)
        return datasets  # Return any successfully created datasets

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
    
    # Track generated plots
    generated_plots = []
    
    try:
        # Process efficiency by type
        try:
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
        except Exception as e:
            logger.warning(f"Failed to generate process efficiency plot: {e}")
        
        # Energy vs efficiency scatter plot
        if 'Energy_per_ton' in df.columns:
            try:
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
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/energy_vs_efficiency.png", dpi=300, bbox_inches='tight')
                plt.close()
                generated_plots.append('energy_vs_efficiency')
            except Exception as e:
                logger.warning(f"Failed to generate energy vs efficiency plot: {e}")
        
        # Parameter correlation matrix
        try:
            # Select relevant numeric columns
            numeric_cols = ['Temperatur (°C)', 'Təzyiq (bar)', 'Prosesin Müddəti (saat)',
                        'Emalın Səmərəliliyi (%)']
            
            # Add derived metrics if available
            for col in ['Energy_per_ton', 'CO2_per_ton', 'Cost_per_ton', 'Safety_Risk']:
                if col in df.columns:
                    numeric_cols.append(col)
            
            # Filter to available columns
            numeric_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(numeric_cols) >= 3:
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
        except Exception as e:
            logger.warning(f"Failed to generate parameter correlation plot: {e}")
        
        # Temperature vs Pressure heatmap for safety (if categories available)
        if 'Temperature_Category' in df.columns and 'Pressure_Category' in df.columns:
            try:
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
                
                plt.figure()
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
            except Exception as e:
                logger.warning(f"Failed to generate temperature-pressure safety plot: {e}")
        
        logger.info(f"Generated {len(generated_plots)} exploratory plots: {', '.join(generated_plots)}")
    
    except Exception as e:
        logger.error(f"Error generating exploratory plots: {e}", exc_info=True)

@timer_decorator    
def run_analysis(input_file: str, output_dir: str = 'dashboard/data', config_file: str = 'data_config.yaml', 
                generate_plots: bool = True, log_to_file: bool = False, create_config: bool = False,
                debug: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run the complete analysis pipeline
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save processed data
        config_file: Path to the configuration file
        generate_plots: Whether to generate exploratory plots
        log_to_file: Whether to log to a file
        create_config: Whether to create a config template file if it doesn't exist
        debug: Whether to enable debug logging
        
    Returns:
        Tuple containing processed DataFrame and dictionary of aggregated datasets
    """
    # Configure logging based on parameters
    global logger
    logger = setup_logging(log_to_file=log_to_file, debug=debug)
    
    start_time = time.time()
    logger.info(f"Starting SOCAR process analysis pipeline")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # 1. Load configuration
        config = load_config(config_file, create_template=create_config)
        
        # 2. Load and validate data
        df = load_and_validate_data(input_file, config)
        
        # 3. Create derived metrics
        processed_df = create_derived_metrics(df, config)
        
        # 4. Create aggregated datasets
        datasets = create_aggregated_datasets(processed_df, config)
        
        # 5. Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 6. Save processed data
        processed_df.to_csv(f"{output_dir}/processed_data.csv", index=False)
        logger.info(f"Saved processed data ({processed_df.shape[0]} rows, {processed_df.shape[1]} columns) to {output_dir}/processed_data.csv")
        
        # 7. Save aggregated datasets
        save_datasets(datasets, output_dir)
        
        # 8. Generate exploratory plots (optional)
        if generate_plots:
            plots_dir = f"{output_dir}/plots"
            generate_exploratory_plots(processed_df, plots_dir, config)
        else:
            logger.info("Skipping exploratory plots generation")
        
        # 9. Save analysis metadata
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
        raise RuntimeError(f"Data analysis pipeline failed: {str(e)}")

def parse_arguments():
    """Parse command line arguments with improved options"""
    parser = argparse.ArgumentParser(description='SOCAR Process Analysis Data Preparation')
    parser.add_argument('--input', '-i', type=str, default='data/data.csv',
                       help='Path to the input CSV file')
    parser.add_argument('--output', '-o', type=str, default='dashboard/data',
                       help='Directory to save processed data')
    parser.add_argument('--config', '-c', type=str, default='data_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating exploratory plots')
    parser.add_argument('--log-to-file', action='store_true',
                       help='Enable logging to file')
    parser.add_argument('--create-config', action='store_true',
                       help='Create config template file if it doesn\'t exist')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the analysis
    run_analysis(
        args.input, 
        args.output, 
        args.config, 
        generate_plots=not args.no_plots,
        log_to_file=args.log_to_file,
        create_config=args.create_config,
        debug=args.debug
    )