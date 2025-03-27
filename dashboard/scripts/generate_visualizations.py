#!/usr/bin/env python
"""
SOCAR Process Analysis Visualization Generator

This script generates visualizations for the process analysis dashboard:
1. Loads processed data from the data preparation step
2. Creates various analytical visualizations
3. Saves visualizations in HTML and JSON formats for dashboard integration

Usage:

python dashboard/scripts/generate_visualizations.py --data-dir dashboard/data --output-dir dashboard/charts

"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import time
import logging
import yaml
import argparse
from pathlib import Path
from functools import wraps
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

def setup_logging(log_to_file=False, debug=False):
    """Set up logging with optional file output"""
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler("visualization_generation.log"))
    
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger("visualization_generation")

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

class VisualizationManager:
    """Class to manage visualization creation and configuration"""
    
    def __init__(self, output_dir: str, config: Dict, save_html: bool = True, save_json: bool = True):
        """
        Initialize visualization manager
        
        Args:
            output_dir: Directory to save visualizations
            config: Visualization configuration
            save_html: Whether to save HTML files
            save_json: Whether to save JSON files
        """
        self.output_dir = output_dir
        self.config = config
        self.save_html = save_html
        self.save_json = save_json
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set default visualization settings
        self.default_settings = config.get('default_settings', {})
        
        # Track created visualizations
        self.created_visualizations = {}
    
    def save_visualization(self, fig: go.Figure, name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Save visualization as HTML and/or JSON based on settings
        
        Args:
            fig: Plotly figure to save
            name: Name for the visualization files
            
        Returns:
            Tuple of (html_path, json_path)
        """
        html_path = None
        json_path = None
        
        try:
            # Save as HTML if enabled
            if self.save_html:
                html_path = f"{self.output_dir}/{name}.html"
                
                # Get HTML config from settings
                html_config = self.default_settings.get('html_config', {
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                })
                
                fig.write_html(html_path, config=html_config)
                logger.info(f"Saved HTML visualization: {html_path}")
            
            # Save as JSON if enabled
            if self.save_json:
                json_path = f"{self.output_dir}/{name}.json"
                with open(json_path, "w") as f:
                    f.write(fig.to_json())
                logger.info(f"Saved JSON visualization: {json_path}")
                
            # Record visualization metadata
            self.created_visualizations[name] = {
                'html_path': html_path,
                'json_path': json_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return html_path, json_path
            
        except Exception as e:
            logger.error(f"Error saving visualization {name}: {e}")
            # Continue execution despite error
            return None, None
    
    def apply_theme(self, fig: go.Figure, chart_type: str) -> go.Figure:
        """
        Apply theme settings to a figure
        
        Args:
            fig: Plotly figure to style
            chart_type: Type of chart for specific styling
            
        Returns:
            Styled Plotly figure
        """
        # Get theme settings
        theme = self.config.get('theme', {})
        
        # Apply general theme settings
        fig.update_layout(
            font=theme.get('font', {'family': 'Arial, sans-serif', 'size': 12}),
            paper_bgcolor=theme.get('paper_bgcolor', 'white'),
            plot_bgcolor=theme.get('plot_bgcolor', 'rgba(240, 240, 240, 0.5)'),
            margin=theme.get('margin', {'l': 60, 'r': 60, 't': 60, 'b': 50}),
            colorway=theme.get('colorway', px.colors.qualitative.G10),
            legend=theme.get('legend', {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1})
        )
        
        # Apply chart-specific themes
        if chart_type == 'bar':
            fig.update_traces(
                marker_line_width=theme.get('bar', {}).get('marker_line_width', 1),
                marker_line_color=theme.get('bar', {}).get('marker_line_color', 'rgb(8, 48, 107)'),
                opacity=theme.get('bar', {}).get('opacity', 0.8)
            )
        elif chart_type == 'scatter':
            fig.update_traces(
                marker=theme.get('scatter', {}).get('marker', {'size': 10, 'line': {'width': 1, 'color': 'DarkSlateGrey'}}),
                opacity=theme.get('scatter', {}).get('opacity', 0.7)
            )
        elif chart_type == 'heatmap':
            fig.update_traces(
                colorscale=theme.get('heatmap', {}).get('colorscale', 'RdBu_r'),
                showscale=theme.get('heatmap', {}).get('showscale', True)
            )
        
        return fig

def load_config(config_file: str = 'visualization_config.yaml', create_template: bool = False) -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to the YAML configuration file
        create_template: Whether to create a template file if config doesn't exist
        
    Returns:
        Configuration as a dictionary
    """
    # Default configuration
    default_config = {
        'theme': {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'rgba(240, 240, 240, 0.5)',
            'margin': {'l': 60, 'r': 60, 't': 60, 'b': 50},
            'colorway': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 
                      'xanchor': 'right', 'x': 1},
            'bar': {'marker_line_width': 1, 'marker_line_color': 'rgb(8, 48, 107)', 'opacity': 0.8},
            'scatter': {'marker': {'size': 10, 'line': {'width': 1, 'color': 'DarkSlateGrey'}}, 'opacity': 0.7},
            'heatmap': {'colorscale': 'RdBu_r', 'showscale': True}
        },
        'default_settings': {
            'html_config': {
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            }
        },
        'visualizations': {
            'process_efficiency': True,
            'energy_safety': True, 
            'process_hierarchy': True,
            'catalyst_analysis': True,
            'parameter_correlation': True,
            'roi_projection': True,
            'safety_optimization': True,
            'kpi_dashboard': True,
            'energy_by_process': True,
            "energy_parameters": True,
            "temp_pressure": True,
            "duration_impact": True,
            "efficiency_correlation": True,
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
            
            # Create template only if explicitly requested
            if create_template:
                with open(config_file, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                logger.info(f"Created configuration template: {config_file}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
    
    return config

@timer_decorator
def load_processed_data(data_dir: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load and validate all processed datasets
    
    Args:
        data_dir: Directory containing the processed data files
        
    Returns:
        Dictionary of DataFrames
    """
    datasets = {}
    required_files = ["processed_data.csv"]
    
    try:
        # Check if directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Get list of available files
        available_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(available_files)} CSV files in {data_dir}")
        
        # Check for required files
        missing_files = [f for f in required_files if f not in available_files]
        if missing_files:
            logger.warning(f"Missing required files: {', '.join(missing_files)}")
            if "processed_data.csv" in missing_files:
                raise FileNotFoundError("Required file processed_data.csv not found. Run data preparation first.")
        
        # Load main dataset
        main_file = os.path.join(data_dir, "processed_data.csv")
        try:
            # Get file size for optimization
            file_size_mb = os.path.getsize(main_file) / (1024 * 1024)
            
            # Use chunking for large files
            if file_size_mb > 100:
                logger.info(f"Loading large main dataset ({file_size_mb:.2f} MB) in chunks")
                chunks = []
                for chunk in pd.read_csv(main_file, chunksize=100000):
                    chunks.append(chunk)
                datasets["df"] = pd.concat(chunks, ignore_index=True)
            else:
                datasets["df"] = pd.read_csv(main_file)
            
            logger.info(f"Loaded main dataset: {datasets['df'].shape[0]} rows, {datasets['df'].shape[1]} columns")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The main dataset file {main_file} is empty")
        except pd.errors.ParserError:
            raise ValueError(f"Could not parse {main_file} as CSV - check file format")
        
        # Load all available aggregated datasets
        for file in available_files:
            if file == "processed_data.csv":
                continue
                
            name = file.replace(".csv", "")
            file_path = os.path.join(data_dir, file)
            
            try:
                datasets[name] = pd.read_csv(file_path)
                logger.info(f"Loaded {name}: {datasets[name].shape[0]} rows, {datasets[name].shape[1]} columns")
            except Exception as e:
                logger.warning(f"Error loading {name}: {e}")
                # Continue execution despite error
        
        return datasets
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise RuntimeError(f"Failed to load processed data: {str(e)}")

@timer_decorator
def create_process_efficiency_comparison(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create process efficiency comparison visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating process efficiency comparison visualization")
    
    try:
        # Determine the dataset to use
        if "process_types" in datasets:
            process_data = datasets["process_types"]
            logger.info("Using process_types dataset for efficiency visualization")
        else:
            # Calculate from main dataset
            logger.info("Calculating process efficiency from main dataset")
            process_data = datasets["df"].groupby("Proses Tipi").agg({
                "Emalın Səmərəliliyi (%)": ["mean", "std"],
                "Emal Həcmi (ton)": "sum",
                "Təhlükəsizlik Hadisələri": "sum"
            }).reset_index()
            
            # Flatten columns
            process_data.columns = ["Proses Tipi", "Efficiency_Mean", "Efficiency_Std", "Volume", "Incidents"]
            process_data["Emalın Səmərəliliyi (%)"] = process_data["Efficiency_Mean"]
        
        # Check if we have the minimum required columns
        required_cols = ["Proses Tipi", "Emalın Səmərəliliyi (%)"]
        missing_cols = [col for col in required_cols if col not in process_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for process efficiency visualization: {missing_cols}")
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart for efficiency
        fig.add_trace(go.Bar(
            x=process_data["Proses Tipi"],
            y=process_data["Emalın Səmərəliliyi (%)"],
            name="Efficiency (%)",
            text=process_data["Emalın Səmərəliliyi (%)"].round(1).astype(str) + "%",
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Efficiency: %{y:.1f}%<extra></extra>"
        ))
        
        # Add safety incidents as a line on secondary axis if available
        incident_column = next((col for col in ["Təhlükəsizlik Hadisələri", "Incidents", "Safety_Risk"] 
                              if col in process_data.columns), None)
        
        if incident_column:
            fig.add_trace(go.Scatter(
                x=process_data["Proses Tipi"],
                y=process_data[incident_column],
                name="Safety Incidents",
                yaxis="y2",
                line=dict(color="red", width=3),
                marker=dict(size=10),
                hovertemplate="<b>%{x}</b><br>Incidents: %{y}<extra></extra>"
            ))
        
        # Update layout with second y-axis
        layout_updates = {
            "title": "Process Efficiency vs. Safety by Type",
            "xaxis": {"title": "Process Type"},
            "yaxis": {
                "title": "Efficiency (%)",
                "range": [85, 100]  # Start from 85% to emphasize differences
            },
            "height": 500,
            "margin": dict(l=50, r=50, t=60, b=50),
            "hovermode": "x unified",
        }
        
        if incident_column:
            layout_updates["yaxis2"] = {
                "title": "Safety Incidents",
                "overlaying": "y",
                "side": "right",
                "rangemode": "nonnegative"
            }
        
        fig.update_layout(**layout_updates)
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'bar')
        
        # Save visualizations
        viz_manager.save_visualization(fig, "process_efficiency")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating process efficiency comparison: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Try to save this error visualization
        viz_manager.save_visualization(fig, "process_efficiency")
        
        return fig

@timer_decorator
def create_energy_safety_relationship(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create energy consumption vs safety incidents visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating energy vs safety relationship visualization")
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if required columns exist
        required_cols = ["Proses Tipi", "Təhlükəsizlik Hadisələri", "Emal Həcmi (ton)"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        energy_col = "Energy_per_ton"
        if energy_col not in df.columns:
            # Try to create Energy_per_ton if missing
            if "Enerji İstifadəsi (kWh)" in df.columns and "Emal Həcmi (ton)" in df.columns:
                df[energy_col] = df["Enerji İstifadəsi (kWh)"] / df["Emal Həcmi (ton)"]
                logger.info("Created Energy_per_ton column from raw data")
            else:
                missing_cols.append(energy_col)
        
        if missing_cols:
            raise ValueError(f"Cannot create energy safety visualization due to missing columns: {missing_cols}")
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter points grouped by process type
        for process_type in df["Proses Tipi"].unique():
            process_df = df[df["Proses Tipi"] == process_type]
            
            # Calculate marker sizes based on volume
            max_volume = process_df["Emal Həcmi (ton)"].max()
            sizes = process_df["Emal Həcmi (ton)"] / max_volume * 20 + 5 if max_volume > 0 else 10
            
            # Add scatter trace
            fig.add_trace(go.Scatter(
                x=process_df[energy_col],
                y=process_df["Təhlükəsizlik Hadisələri"],
                mode="markers",
                name=process_type,
                marker=dict(
                    size=sizes,
                    sizemode="diameter",
                    sizeref=2.0 * df["Emal Həcmi (ton)"].max() / (40.**2) if df["Emal Həcmi (ton)"].max() > 0 else 0.1,
                    sizemin=4
                ),
                text=process_df["Proses Addımı"] if "Proses Addımı" in process_df.columns else None,
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    "Energy: %{x:.2f} kWh/ton<br>" +
                    "Incidents: %{y}<br>" +
                    "Process Type: " + process_type +
                    "<extra></extra>"
                )
            ))
        
        # Add title and axis labels
        fig.update_layout(
            title="Energy Consumption vs. Safety Incidents",
            xaxis_title="Energy per Ton (kWh/ton)",
            yaxis_title="Safety Incidents",
            height=500,
            margin=dict(l=50, r=50, t=60, b=50),
            hovermode="closest"
        )
        
        # Add regions for risk zones
        fig.add_shape(
            type="rect",
            x0=0, y0=3,
            x1=3, y1=100,
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0,
        )
        
        fig.add_annotation(
            x=1.5, y=3.5,
            text="High Risk Zone",
            showarrow=False,
            font=dict(size=12, color="red")
        )
        
        # Add safe zone
        fig.add_shape(
            type="rect",
            x0=0, y0=0,
            x1=1.5, y1=1,
            fillcolor="green",
            opacity=0.1,
            layer="below",
            line_width=0,
        )
        
        fig.add_annotation(
            x=0.75, y=0.5,
            text="Safe Zone",
            showarrow=False,
            font=dict(size=12, color="green")
        )
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'scatter')
        
        # Save visualizations
        viz_manager.save_visualization(fig, "energy_safety")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating energy safety relationship: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating energy-safety visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Try to save this error visualization
        viz_manager.save_visualization(fig, "energy_safety")
        
        return fig

@timer_decorator
def create_process_hierarchy(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create process hierarchy visualization with sunburst chart
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating process hierarchy visualization")
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if required columns exist
        required_cols = ["Proses Tipi", "Proses Addımı", "Emalın Səmərəliliyi (%)", "Emal Həcmi (ton)"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Cannot create process hierarchy visualization due to missing columns: {missing_cols}")
        
        # Group data by process type and step
        param_impact = df.groupby(['Proses Tipi', 'Proses Addımı']).agg({
            'Emalın Səmərəliliyi (%)': 'mean',
            'Emal Həcmi (ton)': 'sum'
        })
        
        # Add energy and safety metrics if available
        for col in ['Energy_per_ton', 'Təhlükəsizlik Hadisələri']:
            if col in df.columns:
                if col == 'Təhlükəsizlik Hadisələri':
                    param_impact[col] = df.groupby(['Proses Tipi', 'Proses Addımı'])[col].sum()
                else:
                    param_impact[col] = df.groupby(['Proses Tipi', 'Proses Addımı'])[col].mean()
        
        # Reset index for sunburst processing
        param_impact = param_impact.reset_index()
        
        # Prepare hover data
        hover_data = ['Emalın Səmərəliliyi (%)']
        for col in ['Energy_per_ton', 'Təhlükəsizlik Hadisələri']:
            if col in param_impact.columns:
                hover_data.append(col)
        
        # Create sunburst chart
        fig = px.sunburst(
            param_impact, 
            path=['Proses Tipi', 'Proses Addımı'], 
            values='Emal Həcmi (ton)',
            color='Emalın Səmərəliliyi (%)',
            color_continuous_scale='RdBu',
            range_color=[85, 100],  # Set color range to emphasize differences
            hover_data=hover_data,
            title='Process Hierarchy, Efficiency and Volume'
        )
        
        # Improve layout
        fig.update_layout(
            height=650,
            margin=dict(l=50, r=50, t=60, b=50),
            coloraxis_colorbar=dict(
                title="Efficiency (%)"
            )
        )
        
        # Apply theme customizations (limited for sunburst charts)
        fig.update_layout(
            font=viz_manager.config.get('theme', {}).get('font', {'family': 'Arial, sans-serif', 'size': 12}),
            paper_bgcolor=viz_manager.config.get('theme', {}).get('paper_bgcolor', 'white')
        )
        
        # Save visualizations
        viz_manager.save_visualization(fig, "process_hierarchy")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating process hierarchy: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating process hierarchy visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Try to save this error visualization
        viz_manager.save_visualization(fig, "process_hierarchy")
        
        return fig

@timer_decorator
def create_safety_heatmap(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create safety heatmap visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating safety heatmap visualization")
    
    try:
        # Try to use safety_parameters aggregation if available
        if "safety_parameters" in datasets:
            safety_data = datasets["safety_parameters"]
            logger.info("Using safety_parameters dataset for heatmap")
        else:
            # Get the main dataset
            df = datasets["df"]
            
            # Check if required columns exist
            if 'Temperature_Category' not in df.columns or 'Pressure_Category' not in df.columns:
                if 'Temperatur (°C)' in df.columns and 'Təzyiq (bar)' in df.columns:
                    # Create categories
                    logger.info("Creating temperature and pressure categories")
                    
                    # Temperature categories
                    temp_bins = [0, 150, 300, 450, float('inf')]
                    temp_labels = ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)']
                    df['Temperature_Category'] = pd.cut(df['Temperatur (°C)'], bins=temp_bins, labels=temp_labels)
                    
                    # Pressure categories
                    pressure_bins = [0, 10, 30, 50, float('inf')]
                    pressure_labels = ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)']
                    df['Pressure_Category'] = pd.cut(df['Təzyiq (bar)'], bins=pressure_bins, labels=pressure_labels)
                else:
                    raise ValueError("Cannot create safety heatmap: missing temperature and pressure data")
            
            # Group by temperature and pressure categories
            safety_data = df.groupby(['Temperature_Category', 'Pressure_Category']).agg({
                'Təhlükəsizlik Hadisələri': 'sum',
                'Emal Həcmi (ton)': 'sum'
            }).reset_index()
            
            # Calculate incident rate
            safety_data['Incident_Rate'] = safety_data['Təhlükəsizlik Hadisələri'] / safety_data['Emal Həcmi (ton)'] * 100
        
        # Check if we have the data we need for the heatmap
        required_cols = ['Temperature_Category', 'Pressure_Category']
        missing_cols = [col for col in required_cols if col not in safety_data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns for safety heatmap: {missing_cols}")
        
        # Determine the incident measure to use
        incident_cols = ['Incident_Rate', 'Has_Incident_mean', 'Təhlükəsizlik Hadisələri']
        incident_col = next((col for col in incident_cols if col in safety_data.columns), None)
        
        if not incident_col:
            raise ValueError("No suitable incident measure found for safety heatmap")
        
        # Create pivot table
        pivot_data = safety_data.pivot_table(
            index='Temperature_Category', 
            columns='Pressure_Category', 
            values=incident_col,
            aggfunc='mean'
        )
        
        # Ensure all combinations have values
        temp_categories = safety_data['Temperature_Category'].unique()
        pressure_categories = safety_data['Pressure_Category'].unique()
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Reds',
            colorbar=dict(title="Incident Rate" if "Rate" in incident_col else "Incidents"),
            text=np.round(pivot_data.values, 1),
            texttemplate="%{text}",
            hovertemplate="Temperature: %{y}<br>Pressure: %{x}<br>Value: %{z:.1f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Safety Incident Rate by Temperature and Pressure",
            xaxis_title="Pressure Category",
            yaxis_title="Temperature Category",
            height=600,
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'heatmap')
        
        # Save visualization
        viz_manager.save_visualization(fig, "safety_heatmap")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating safety heatmap: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating safety heatmap visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "safety_heatmap")
        
        return fig

@timer_decorator
def create_process_step_safety(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create process step safety visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating process step safety visualization")
    
    try:
        # Try to use process_steps aggregation if available
        if "process_steps" in datasets:
            step_data = datasets["process_steps"]
            logger.info("Using process_steps dataset for safety analysis")
        else:
            # Get the main dataset
            df = datasets["df"]
            
            # Check if required columns exist
            required_cols = ["Proses Addımı", "Təhlükəsizlik Hadisələri", "Emal Həcmi (ton)"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Cannot create process step safety visualization due to missing columns: {missing_cols}")
            
            # Group by process step
            step_data = df.groupby("Proses Addımı").agg({
                "Təhlükəsizlik Hadisələri": "sum",
                "Emal Həcmi (ton)": "sum",
                "Emalın Səmərəliliyi (%)": "mean"
            }).reset_index()
            
            # Calculate incident rate
            step_data["Safety_Risk"] = step_data["Təhlükəsizlik Hadisələri"] / step_data["Emal Həcmi (ton)"] * 1000
        
        # Determine incident rate column to use
        incident_cols = ["Safety_Risk", "Təhlükəsizlik Hadisələri_sum", "Təhlükəsizlik Hadisələri"]
        incident_col = next((col for col in incident_cols if col in step_data.columns), None)
        
        if not incident_col:
            raise ValueError("No suitable incident measure found for process step safety visualization")
        
        # Sort by incident rate
        step_data = step_data.sort_values(by=incident_col, ascending=False)
        
        # Create horizontal bar chart
        fig = go.Figure(data=go.Bar(
            y=step_data["Proses Addımı"],
            x=step_data[incident_col],
            orientation="h",
            marker_color=step_data[incident_col],
            marker_colorscale="Reds",
            text=np.round(step_data[incident_col], 2),
            textposition="auto",
            hovertemplate="Process Step: %{y}<br>Value: %{x:.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Safety Incident Rate by Process Step",
            xaxis_title="Incident Rate" if "Risk" in incident_col else "Incidents",
            yaxis_title="Process Step",
            height=600,
            margin=dict(l=150, r=50, t=60, b=50),
            yaxis=dict(automargin=True)
        )
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'bar')
        
        # Save visualization
        viz_manager.save_visualization(fig, "process_step_safety")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating process step safety visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating process step safety visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "process_step_safety")
        
        return fig

@timer_decorator
def create_catalyst_analysis(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> Tuple[go.Figure, go.Figure]:
    """
    Create catalyst analysis visualizations
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Tuple of (catalyst_matrix, catalyst_parallel) figures
    """
    logger.info("Creating catalyst analysis visualizations")
    
    # Initialize with error figures in case we need to return early
    matrix_fig = go.Figure()
    parallel_fig = go.Figure()
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if catalyst column exists
        if 'İstifadə Edilən Katalizatorlar' not in df.columns:
            raise ValueError("Cannot create catalyst analysis due to missing catalyst column")
        
        # Group by catalyst
        agg_dict = {
            'Emalın Səmərəliliyi (%)': 'mean',
            'Emal Həcmi (ton)': 'sum',
            'Təhlükəsizlik Hadisələri': 'sum'
        }
        
        # Add energy metrics if available
        if 'Energy_per_ton' in df.columns:
            agg_dict['Energy_per_ton'] = 'mean'
        
        catalyst_data = df.groupby('İstifadə Edilən Katalizatorlar').agg(agg_dict).reset_index()
        
        # Calculate incident rate
        if 'Təhlükəsizlik Hadisələri' in catalyst_data.columns and 'Emal Həcmi (ton)' in catalyst_data.columns:
            catalyst_data['Incident_Rate'] = catalyst_data['Təhlükəsizlik Hadisələri'] / catalyst_data['Emal Həcmi (ton)'] * 1000
        
        # VISUALIZATION 1: Catalyst Matrix Chart
        if 'Energy_per_ton' in catalyst_data.columns:
            # Create scatter plot
            matrix_fig = px.scatter(
                catalyst_data,
                x='Energy_per_ton',
                y='Emalın Səmərəliliyi (%)',
                size='Emal Həcmi (ton)',
                color='Incident_Rate' if 'Incident_Rate' in catalyst_data.columns else None,
                hover_name='İstifadə Edilən Katalizatorlar',
                labels={
                    'Energy_per_ton': 'Energy Consumption (kWh/ton)',
                    'Emalın Səmərəliliyi (%)': 'Process Efficiency (%)',
                    'Emal Həcmi (ton)': 'Processing Volume (tons)',
                    'Incident_Rate': 'Incident Rate (per 1000 tons)'
                },
                title='Catalyst Performance Matrix',
                color_continuous_scale='RdYlGn_r'  # Red for high incident rate, green for low
            )
            
            # Improve layout
            matrix_fig.update_layout(
                height=600,
                margin=dict(l=50, r=50, t=60, b=50),
                xaxis=dict(title='Energy Consumption (kWh/ton)'),
                yaxis=dict(title='Process Efficiency (%)', range=[catalyst_data['Emalın Səmərəliliyi (%)'].min() * 0.95, 100])
            )
        else:
            # Create simple bar chart if energy data not available
            matrix_fig = px.bar(
                catalyst_data.sort_values('Emalın Səmərəliliyi (%)', ascending=False),
                x='İstifadə Edilən Katalizatorlar',
                y='Emalın Səmərəliliyi (%)',
                title='Catalyst Efficiency Comparison',
                labels={'Emalın Səmərəliliyi (%)': 'Process Efficiency (%)'}
            )
            
            # Add text labels
            matrix_fig.update_traces(
                text=catalyst_data.sort_values('Emalın Səmərəliliyi (%)', ascending=False)['Emalın Səmərəliliyi (%)'].round(1),
                textposition='auto'
            )
            
            matrix_fig.update_layout(
                height=500,
                margin=dict(l=50, r=50, t=60, b=50),
                xaxis=dict(title='Catalyst Type'),
                yaxis=dict(title='Process Efficiency (%)', range=[catalyst_data['Emalın Səmərəliliyi (%)'].min() * 0.95, 100])
            )
        
        # VISUALIZATION 2: Catalyst Parallel Coordinates
        # If we have at least 3 metrics
        metrics = ['Emalın Səmərəliliyi (%)', 'Energy_per_ton', 'Incident_Rate']
        available_metrics = [m for m in metrics if m in catalyst_data.columns]
        
        if len(available_metrics) >= 2:
            # Create dimensions
            dimensions = []
            
            for metric in available_metrics:
                if metric == 'Emalın Səmərəliliyi (%)':
                    dimensions.append(dict(
                        range=[catalyst_data[metric].min() * 0.95, 100],
                        label='Efficiency (%)',
                        values=catalyst_data[metric]
                    ))
                elif metric == 'Energy_per_ton':
                    dimensions.append(dict(
                        range=[0, catalyst_data[metric].max() * 1.05],
                        label='Energy (kWh/ton)',
                        values=catalyst_data[metric]
                    ))
                elif metric == 'Incident_Rate':
                    dimensions.append(dict(
                        range=[0, catalyst_data[metric].max() * 1.05],
                        label='Incident Rate (per 1000 tons)',
                        values=catalyst_data[metric]
                    ))
            
            # Add volume dimension
            dimensions.append(dict(
                range=[0, catalyst_data['Emal Həcmi (ton)'].max() * 1.05],
                label='Volume (tons)',
                values=catalyst_data['Emal Həcmi (ton)']
            ))
            
            # Create the parallel coordinates plot
            parallel_fig = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=catalyst_data['Emalın Səmərəliliyi (%)'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title='Efficiency (%)')
                    ),
                    dimensions=dimensions,
                    labelfont=dict(size=12),
                    tickfont=dict(size=10)
                )
            )
            
            # Update layout
            parallel_fig.update_layout(
                title='Catalyst Performance Comparison',
                height=600,
                margin=dict(l=80, r=80, t=60, b=50)
            )
        else:
            # Create a simple fallback
            parallel_fig = go.Figure()
            parallel_fig.add_annotation(
                text="Insufficient metrics available for parallel coordinates plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            
            parallel_fig.update_layout(
                title='Catalyst Performance Comparison',
                height=400,
                margin=dict(l=50, r=50, t=60, b=50)
            )
        
        # Apply theme (limited for these complex chart types)
        matrix_fig.update_layout(
            font=viz_manager.config.get('theme', {}).get('font', {'family': 'Arial, sans-serif', 'size': 12}),
            paper_bgcolor=viz_manager.config.get('theme', {}).get('paper_bgcolor', 'white')
        )
        
        parallel_fig.update_layout(
            font=viz_manager.config.get('theme', {}).get('font', {'family': 'Arial, sans-serif', 'size': 12}),
            paper_bgcolor=viz_manager.config.get('theme', {}).get('paper_bgcolor', 'white')
        )
        
        # Save visualizations
        viz_manager.save_visualization(matrix_fig, "catalyst_matrix")
        viz_manager.save_visualization(parallel_fig, "catalyst_parallel")
        
        return matrix_fig, parallel_fig
    
    except Exception as e:
        logger.error(f"Error creating catalyst analysis visualizations: {e}")
        
        # Create minimal versions to avoid breaking the dashboard
        if matrix_fig.data == ():  # If we haven't created it yet
            matrix_fig = go.Figure()
            matrix_fig.add_annotation(
                text=f"Error creating catalyst matrix visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="red", size=14)
            )
            viz_manager.save_visualization(matrix_fig, "catalyst_matrix")
        
        if parallel_fig.data == ():  # If we haven't created it yet
            parallel_fig = go.Figure()
            parallel_fig.add_annotation(
                text=f"Error creating catalyst parallel visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="red", size=14)
            )
            viz_manager.save_visualization(parallel_fig, "catalyst_parallel")
        
        return matrix_fig, parallel_fig

@timer_decorator
def create_roi_projection(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create ROI projection visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        ROI by process visualization
    """
    logger.info("Creating ROI projection visualization")
    
    try:
        # Get the process types data
        if "process_types" in datasets:
            process_data = datasets["process_types"]
        else:
            # Calculate from main dataset
            df = datasets["df"]
            process_data = df.groupby("Proses Tipi").agg({
                "Emalın Səmərəliliyi (%)": "mean",
                "Emal Həcmi (ton)": "sum",
                "Əməliyyat Xərcləri (AZN)": "sum"
            }).reset_index()
        
        # Check if we have the necessary columns
        if "Əməliyyat Xərcləri (AZN)" not in process_data.columns and "Əməliyyat Xərcləri (AZN)_sum" not in process_data.columns:
            # Try to get from main dataset
            df = datasets["df"]
            if "Əməliyyat Xərcləri (AZN)" in df.columns:
                cost_by_type = df.groupby("Proses Tipi")["Əməliyyat Xərcləri (AZN)"].sum().reset_index()
                process_data = process_data.merge(cost_by_type, on="Proses Tipi", how="left")
            else:
                raise ValueError("Cannot create ROI projection: missing cost data")
        
        # Determine column names based on what's available
        cost_col = next((col for col in ["Əməliyyat Xərcləri (AZN)", "Əməliyyat Xərcləri (AZN)_sum"] 
                        if col in process_data.columns), None)
        
        if not cost_col:
            raise ValueError("Cannot create ROI projection: missing cost data")
        
        # Calculate ROI metrics
        annual_factor = 12  # Annualization factor - adjust if needed
        
        # Annualized costs
        process_data["Annual_Cost"] = process_data[cost_col] * annual_factor
        
        # Set improvement percentages (could be made configurable)
        process_data["Efficiency_Improvement"] = 5  # 5% efficiency improvement
        process_data["Energy_Reduction"] = 10  # 10% energy reduction
        process_data["Safety_Improvement"] = 15  # 15% safety improvement
        
        # Calculate savings components
        process_data["Efficiency_Savings"] = process_data["Annual_Cost"] * 0.4 * (process_data["Efficiency_Improvement"] / 100)
        process_data["Energy_Savings"] = process_data["Annual_Cost"] * 0.3 * (process_data["Energy_Reduction"] / 100)
        process_data["Safety_Savings"] = process_data["Annual_Cost"] * 0.2 * (process_data["Safety_Improvement"] / 100)
        process_data["Other_Savings"] = process_data["Annual_Cost"] * 0.1 * 0.05  # 5% on remaining 10% of costs
        
        # Total savings and ROI
        process_data["Total_Savings"] = (
            process_data["Efficiency_Savings"] + 
            process_data["Energy_Savings"] + 
            process_data["Safety_Savings"] +
            process_data["Other_Savings"]
        )
        
        # Investment (typically 1.5-2x annual savings)
        process_data["Investment"] = process_data["Total_Savings"] * 1.8
        
        # ROI in months
        process_data["ROI_Months"] = process_data["Investment"] / (process_data["Total_Savings"] / 12)
        
        # Create a subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add savings bars
        fig.add_trace(
            go.Bar(
                x=process_data["Proses Tipi"],
                y=process_data["Total_Savings"],
                name="Annual Savings",
                marker_color="#2ca02c",
                text=[f"{val:,.0f} AZN" for val in process_data["Total_Savings"]],
                textposition="auto"
            ),
            secondary_y=False
        )
        
        # Add ROI months line
        fig.add_trace(
            go.Scatter(
                x=process_data["Proses Tipi"],
                y=process_data["ROI_Months"],
                name="Payback Period",
                line=dict(color="#ff7f0e", width=3),
                mode="lines+markers+text",
                text=[f"{val:.1f} months" for val in process_data["ROI_Months"]],
                textposition="top center"
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="ROI Analysis by Process Type",
            height=500,
            margin=dict(l=50, r=50, t=60, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_yaxes(title_text="Annual Savings (AZN)", secondary_y=False)
        fig.update_yaxes(title_text="Payback Period (months)", secondary_y=True)
        
        # Apply theme customizations 
        fig.update_layout(
            font=viz_manager.config.get('theme', {}).get('font', {'family': 'Arial, sans-serif', 'size': 12}),
            paper_bgcolor=viz_manager.config.get('theme', {}).get('paper_bgcolor', 'white'),
            plot_bgcolor=viz_manager.config.get('theme', {}).get('plot_bgcolor', 'rgba(240, 240, 240, 0.5)')
        )
        
        # Save visualization
        viz_manager.save_visualization(fig, "roi_by_process")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating ROI projection visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating ROI projection visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "roi_by_process")
        
        return fig

@timer_decorator
def create_energy_by_process(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create energy consumption by process type visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating energy by process visualization")
    
    try:
        # Determine the dataset to use
        if "process_types" in datasets:
            process_data = datasets["process_types"]
            logger.info("Using process_types dataset for energy by process visualization")
        else:
            # Calculate from main dataset
            df = datasets["df"]
            
            # Check if required columns exist
            required_cols = ["Proses Tipi", "Emal Həcmi (ton)"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            # Check for energy columns
            energy_col = None
            if "Energy_per_ton" in df.columns:
                energy_col = "Energy_per_ton"
            elif "Enerji İstifadəsi (kWh)" in df.columns and "Emal Həcmi (ton)" in df.columns:
                energy_col = "Enerji İstifadəsi (kWh)"
                logger.info("Will calculate Energy_per_ton from raw energy consumption")
            else:
                missing_cols.append("Energy metric")
            
            if missing_cols:
                raise ValueError(f"Cannot create energy by process visualization due to missing columns: {missing_cols}")
            
            # Calculate aggregated data
            agg_dict = {"Emal Həcmi (ton)": "sum"}
            if energy_col == "Energy_per_ton":
                agg_dict[energy_col] = "mean"
            else:  # energy_col == "Enerji İstifadəsi (kWh)"
                agg_dict[energy_col] = "sum"
            
            process_data = df.groupby("Proses Tipi").agg(agg_dict).reset_index()
            
            # Calculate Energy_per_ton if needed
            if energy_col == "Enerji İstifadəsi (kWh)":
                process_data["Energy_per_ton"] = process_data[energy_col] / process_data["Emal Həcmi (ton)"]
        
        # Ensure we have Energy_per_ton
        if "Energy_per_ton" not in process_data.columns:
            # Try to find column with similar name
            energy_cols = [col for col in process_data.columns if "energy" in col.lower() or "enerji" in col.lower()]
            if energy_cols:
                # Use the first match
                process_data["Energy_per_ton"] = process_data[energy_cols[0]]
                logger.info(f"Using {energy_cols[0]} as Energy_per_ton")
            else:
                raise ValueError("Energy_per_ton column not available for visualization")
        
        # Sort by energy consumption
        process_data = process_data.sort_values(by="Energy_per_ton", ascending=False)
        
        # Create figure with bar chart
        fig = go.Figure()
        
        # Add bar chart for energy consumption
        fig.add_trace(go.Bar(
            x=process_data["Proses Tipi"],
            y=process_data["Energy_per_ton"],
            name="Energy per Ton",
            marker_color="#1f77b4",
            text=process_data["Energy_per_ton"].round(2),
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Energy: %{y:.2f} kWh/ton<extra></extra>"
        ))
        
        # Add volume as a line on secondary axis if available
        if "Emal Həcmi (ton)" in process_data.columns:
            fig.add_trace(go.Scatter(
                x=process_data["Proses Tipi"],
                y=process_data["Emal Həcmi (ton)"],
                name="Processing Volume",
                yaxis="y2",
                line=dict(color="#ff7f0e", width=3),
                marker=dict(size=10),
                hovertemplate="<b>%{x}</b><br>Volume: %{y:.0f} tons<extra></extra>"
            ))
        
        # Update layout
        layout = {
            "title": "Energy Consumption by Process Type",
            "xaxis": {"title": "Process Type"},
            "yaxis": {
                "title": "Energy (kWh/ton)",
                "rangemode": "nonnegative"
            },
            "height": 500,
            "margin": dict(l=50, r=50, t=60, b=50),
            "hovermode": "x unified"
        }
        
        if "Emal Həcmi (ton)" in process_data.columns:
            layout["yaxis2"] = {
                "title": "Volume (tons)",
                "overlaying": "y",
                "side": "right",
                "rangemode": "nonnegative"
            }
        
        fig.update_layout(**layout)
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'bar')
        
        # Save visualization
        viz_manager.save_visualization(fig, "energy_by_process")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating energy by process visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating energy by process visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "energy_by_process")
        
        return fig

@timer_decorator
def create_energy_parameters(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create energy consumption vs temperature and pressure visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating energy vs parameters visualization")
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if required columns exist
        required_cols = ["Temperatur (°C)", "Təzyiq (bar)"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        # Check for energy column
        if "Energy_per_ton" not in df.columns:
            if "Enerji İstifadəsi (kWh)" in df.columns and "Emal Həcmi (ton)" in df.columns:
                df["Energy_per_ton"] = df["Enerji İstifadəsi (kWh)"] / df["Emal Həcmi (ton)"]
                logger.info("Created Energy_per_ton column from raw energy consumption")
            else:
                missing_cols.append("Energy_per_ton")
        
        if missing_cols:
            raise ValueError(f"Cannot create energy vs parameters visualization due to missing columns: {missing_cols}")
        
        # Create figure with 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=df["Temperatur (°C)"],
            y=df["Təzyiq (bar)"],
            z=df["Energy_per_ton"],
            mode='markers',
            marker=dict(
                size=5,
                color=df["Energy_per_ton"],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Energy (kWh/ton)")
            ),
            text=df["Proses Addımı"] if "Proses Addımı" in df.columns else None,
            hovertemplate="<b>%{text}</b><br>" +
                        "Temperature: %{x}°C<br>" +
                        "Pressure: %{y} bar<br>" +
                        "Energy: %{z:.2f} kWh/ton<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Energy Consumption vs Temperature and Pressure",
            scene=dict(
                xaxis_title="Temperature (°C)",
                yaxis_title="Pressure (bar)",
                zaxis_title="Energy Consumption (kWh/ton)",
                aspectmode='cube'
            ),
            height=600,
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        # Apply basic theme elements (limited for 3D plots)
        fig.update_layout(
            font=viz_manager.config.get('theme', {}).get('font', {'family': 'Arial, sans-serif', 'size': 12}),
            paper_bgcolor=viz_manager.config.get('theme', {}).get('paper_bgcolor', 'white')
        )
        
        # Save visualization
        viz_manager.save_visualization(fig, "energy_parameters")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating energy vs parameters visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating energy vs parameters visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "energy_parameters")
        
        return fig

@timer_decorator
def create_parameter_correlation(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create parameter correlation matrix visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating parameter correlation matrix visualization")
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Select numeric columns for correlation
        numeric_cols = [
            'Temperatur (°C)', 'Təzyiq (bar)', 'Prosesin Müddəti (saat)',
            'Emalın Səmərəliliyi (%)', 'Təhlükəsizlik Hadisələri'
        ]
        
        # Add energy metrics if available
        for col in ['Energy_per_ton', 'Enerji İstifadəsi (kWh)', 'CO2_per_ton', 'Cost_per_ton']:
            if col in df.columns:
                numeric_cols.append(col)
        
        # Filter to columns that exist in the dataset
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 3:
            raise ValueError("Not enough numeric columns available for correlation matrix")
        
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Parameter Correlation Matrix",
            height=600,
            margin=dict(l=50, r=50, t=60, b=100),
            xaxis=dict(tickangle=45)
        )
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'heatmap')
        
        # Save visualization
        viz_manager.save_visualization(fig, "parameter_correlation")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating parameter correlation matrix visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating parameter correlation matrix visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "parameter_correlation")
        
        return fig

@timer_decorator
def create_temp_pressure(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create temperature-pressure relationship visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating temperature-pressure relationship visualization")
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if required columns exist
        required_cols = ["Temperatur (°C)", "Təzyiq (bar)", "Proses Tipi"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Cannot create temperature-pressure visualization due to missing columns: {missing_cols}")
        
        # Check for efficiency column
        # efficiency_col = None
        # if "Emalın Səmərəliliyi (%)" in df.columns:
        #     efficiency_col = "Emalın Səmərəliliyi (%)"
        # elif "Process_KPI_Score" in df.columns:
        #     efficiency_col = "Process_KPI_Score"
        efficiency_col = next((col for col in ["Emalın Səmərəliliyi (%)", "Process_KPI_Score"]
                               if col in df.columns), None)
        
        # Create scatter plot with process types
        fig = go.Figure()
        
        # Add traces for each process type
        for process_type in df["Proses Tipi"].unique():
            process_df = df[df["Proses Tipi"] == process_type]
            
            # Create hover text
            hover_text = []
            for idx, row in process_df.iterrows():
                text = f"Process: {process_type}<br>Temperature: {row['Temperatur (°C)']}°C<br>Pressure: {row['Təzyiq (bar)']} bar"
                if "Proses Addımı" in process_df.columns:
                    text += f"<br>Step: {row['Proses Addımı']}"
                if efficiency_col and efficiency_col in process_df.columns:
                    text += f"<br>Efficiency: {row[efficiency_col]:.1f}%"
                hover_text.append(text)
            
            # Add trace with color based on efficiency if available
            marker_args = {}
            if efficiency_col and efficiency_col in process_df.columns:
                marker_args = {
                    "color": process_df[efficiency_col],
                    "colorscale": "Viridis",
                    "colorbar": {
                        "title": "Efficiency (%)",
                        "title": {"side": "right"}
                        # "titleside": "right"
                    },
                    "showscale": process_type == df["Proses Tipi"].unique()[0]  # Show colorbar only for first process
                }
            
            fig.add_trace(go.Scatter(
                x=process_df["Temperatur (°C)"],
                y=process_df["Təzyiq (bar)"],
                mode="markers",
                name=process_type,
                text=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=10,
                    opacity=0.7,
                    **marker_args
                )
            ))
        
        # Update layout
        fig.update_layout(
            title="Temperature-Pressure Relationship by Process Type",
            xaxis_title="Temperature (°C)",
            yaxis_title="Pressure (bar)",
            height=500,
            margin=dict(l=50, r=50, t=60, b=50),
            hovermode="closest"
        )
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'scatter')
        
        # Save visualization
        viz_manager.save_visualization(fig, "temp_pressure")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating temperature-pressure visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating temperature-pressure visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "temp_pressure")
        
        return fig

@timer_decorator
def create_duration_impact(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create process duration impact visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating process duration impact visualization")
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if required columns exist
        required_cols = ["Prosesin Müddəti (saat)", "Emalın Səmərəliliyi (%)", "Proses Tipi"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Cannot create duration impact visualization due to missing columns: {missing_cols}")
        
        # Create a scatter plot with trend lines for each process type
        fig = go.Figure()
        
        # Add scatter points and trend lines for each process type
        for process_type in df["Proses Tipi"].unique():
            process_df = df[df["Proses Tipi"] == process_type]
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=process_df["Prosesin Müddəti (saat)"],
                y=process_df["Emalın Səmərəliliyi (%)"],
                mode="markers",
                name=f"{process_type} (Data)",
                marker=dict(
                    size=10,
                    opacity=0.7
                ),
                hovertemplate="<b>" + process_type + "</b><br>" +
                            "Duration: %{x} hours<br>" +
                            "Efficiency: %{y:.1f}%<extra></extra>"
            ))
            
            # Add trend line using linear regression if there are enough points
            if len(process_df) >= 3:
                # Calculate regression using numpy
                x = process_df["Prosesin Müddəti (saat)"].values
                y = process_df["Emalın Səmərəliliyi (%)"].values
                
                # Add a constant term to x for the intercept
                x_with_const = np.vstack([x, np.ones(len(x))]).T
                
                # Calculate the regression coefficients (slope, intercept)
                try:
                    # Use numpy's least squares solution
                    m, c = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                    
                    # Create x values for the trend line
                    x_trend = np.array([min(x), max(x)])
                    y_trend = m * x_trend + c
                    
                    # Add trend line
                    fig.add_trace(go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode="lines",
                        name=f"{process_type} (Trend)",
                        line=dict(
                            dash="dash",
                            width=2
                        ),
                        hoverinfo="skip"
                    ))
                except Exception as e:
                    logger.warning(f"Could not calculate trend line for {process_type}: {e}")
        
        # Update layout
        fig.update_layout(
            title="Process Duration Impact on Efficiency",
            xaxis_title="Process Duration (hours)",
            yaxis_title="Process Efficiency (%)",
            height=500,
            margin=dict(l=50, r=50, t=60, b=50),
            hovermode="closest"
        )
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'scatter')
        
        # Save visualization
        viz_manager.save_visualization(fig, "duration_impact")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating duration impact visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating duration impact visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "duration_impact")
        
        return fig

@timer_decorator
def create_efficiency_correlation(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create efficiency correlation visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating efficiency correlation visualization")
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if required columns exist
        required_cols = ["Emalın Səmərəliliyi (%)"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Cannot create efficiency correlation visualization due to missing columns: {missing_cols}")
        
        # Potential parameter columns
        parameter_cols = [
            "Temperatur (°C)", "Təzyiq (bar)", "Prosesin Müddəti (saat)"
        ]
        
        # Add additional metrics if available
        for col in ["Energy_per_ton", "Enerji İstifadəsi (kWh)"]:
            if col in df.columns:
                parameter_cols.append(col)
        
        # Filter to available columns
        available_params = [col for col in parameter_cols if col in df.columns]
        
        if len(available_params) < 1:
            raise ValueError("No parameter columns available for efficiency correlation")
        
        # Create subplot with multiple parameters
        nrows = (len(available_params) + 1) // 2  # Ceiling division
        fig = make_subplots(rows=nrows, cols=2, 
                           subplot_titles=[f"{param} vs Efficiency" for param in available_params],
                           vertical_spacing=0.1,
                           horizontal_spacing=0.1)
        
        # Loop through parameters and create scatter plots
        for i, param in enumerate(available_params):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Calculate correlation
            corr = df[param].corr(df["Emalın Səmərəliliyi (%)"])
            corr_text = f"Correlation: {corr:.2f}"
            
            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=df[param],
                    y=df["Emalın Səmərəliliyi (%)"],
                    mode="markers",
                    name=param,
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        color=df["Emalın Səmərəliliyi (%)"],
                        colorscale="Viridis",
                        showscale=(i == 0)  # Show colorbar only for first parameter
                    ),
                    hovertemplate=f"{param}: %{{x}}<br>Efficiency: %{{y:.1f}}%<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Add trend line
            x = df[param].values
            y = df["Emalın Səmərəliliyi (%)"].values
            
            # Add a constant term to x for the intercept
            x_with_const = np.vstack([x, np.ones(len(x))]).T
            
            try:
                # Calculate regression
                m, c = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                
                # Create x values for the trend line
                x_trend = np.array([min(x), max(x)])
                y_trend = m * x_trend + c
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode="lines",
                        name=f"{param} Trend",
                        line=dict(
                            color="red",
                            dash="dash",
                            width=2
                        ),
                        showlegend=False,
                        hoverinfo="skip"
                    ),
                    row=row, col=col
                )
                
                # Add correlation annotation
                fig.add_annotation(
                    x=0.95,
                    y=0.05,
                    xref=f"x{i+1 if i > 0 else ''} domain",
                    yref=f"y{i+1 if i > 0 else ''} domain",
                    text=corr_text,
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255, 255, 255, 0.7)",
                    borderpad=4
                )
            except Exception as e:
                logger.warning(f"Could not calculate trend line for {param}: {e}")
        
        # Update layout for empty subplots if odd number of parameters
        if len(available_params) % 2 == 1:
            # Hide unused subplot
            fig.update_xaxes(visible=False, row=nrows, col=2)
            fig.update_yaxes(visible=False, row=nrows, col=2)
        
        # Update overall layout
        fig.update_layout(
            title="Parameter Correlation with Process Efficiency",
            height=300 * nrows,
            margin=dict(l=50, r=50, t=80, b=50),
            showlegend=False
        )
        
        # Update all xaxes and yaxes
        for i in range(len(available_params)):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.update_xaxes(title_text=available_params[i], row=row, col=col)
            fig.update_yaxes(title_text="Efficiency (%)" if col == 1 else None, row=row, col=col)
        
        # Apply theme elements (limited for subplots)
        fig.update_layout(
            font=viz_manager.config.get('theme', {}).get('font', {'family': 'Arial, sans-serif', 'size': 12}),
            paper_bgcolor=viz_manager.config.get('theme', {}).get('paper_bgcolor', 'white'),
            plot_bgcolor=viz_manager.config.get('theme', {}).get('plot_bgcolor', 'rgba(240, 240, 240, 0.5)')
        )
        
        # Save visualization
        viz_manager.save_visualization(fig, "efficiency_correlation")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating efficiency correlation visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating efficiency correlation visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "efficiency_correlation")
        
        return fig

@timer_decorator
def create_kpi_dashboard(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager
) -> go.Figure:
    """
    Create KPI dashboard visualization with summary metrics
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        
    Returns:
        Plotly figure
    """
    logger.info("Creating KPI dashboard visualization")
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Initialize KPI data
        kpi_data = {}
        
        # Get overall efficiency
        if "Emalın Səmərəliliyi (%)" in df.columns:
            kpi_data["efficiency"] = {
                "value": df["Emalın Səmərəliliyi (%)"].mean(),
                "target": 95.0,
                "unit": "%",
                "title": "Average Process Efficiency",
                "color": "#1f77b4"  # Blue
            }
        
        # Get energy metrics
        energy_col = None
        if "Energy_per_ton" in df.columns:
            energy_col = "Energy_per_ton"
        elif "Enerji İstifadəsi (kWh)" in df.columns and "Emal Həcmi (ton)" in df.columns:
            # Calculate energy per ton
            total_energy = df["Enerji İstifadəsi (kWh)"].sum()
            total_volume = df["Emal Həcmi (ton)"].sum()
            
            if total_volume > 0:
                energy_value = total_energy / total_volume
                kpi_data["energy"] = {
                    "value": energy_value,
                    "target": 1.6,
                    "unit": "kWh/ton",
                    "title": "Energy Consumption",
                    "color": "#2ca02c"  # Green
                }
        elif energy_col:
            kpi_data["energy"] = {
                "value": df[energy_col].mean(),
                "target": 1.6,
                "unit": "kWh/ton",
                "title": "Energy Consumption",
                "color": "#2ca02c"  # Green
            }
        
        # Get safety metrics
        if "Təhlükəsizlik Hadisələri" in df.columns and "Emal Həcmi (ton)" in df.columns:
            total_incidents = df["Təhlükəsizlik Hadisələri"].sum()
            total_volume = df["Emal Həcmi (ton)"].sum()
            
            if total_volume > 0:
                incidents_per_1000t = (total_incidents / total_volume) * 1000
                kpi_data["safety"] = {
                    "value": incidents_per_1000t,
                    "target": 2.0,
                    "unit": "per 1000t",
                    "title": "Safety Incidents",
                    "color": "#d62728"  # Red
                }
        
        # Get cost metrics
        if "Əməliyyat Xərcləri (AZN)" in df.columns and "Emal Həcmi (ton)" in df.columns:
            total_cost = df["Əməliyyat Xərcləri (AZN)"].sum()
            total_volume = df["Emal Həcmi (ton)"].sum()
            
            if total_volume > 0:
                cost_per_ton = total_cost / total_volume
                kpi_data["cost"] = {
                    "value": cost_per_ton,
                    "target": cost_per_ton * 0.9,  # Target 10% lower than current
                    "unit": "AZN/ton",
                    "title": "Operating Cost",
                    "color": "#ff7f0e"  # Orange
                }
        
        # Check if we have any KPIs
        if not kpi_data:
            raise ValueError("No KPI metrics available for dashboard visualization")
        
        # Create figure with gauge charts for KPIs
        # fig = make_subplots(
        #     rows=1, cols=len(kpi_data),
        #     subplot_titles=[kpi["title"] for kpi in kpi_data.values()]
        # )
        fig = make_subplots(
            rows=1, cols=len(kpi_data),
            specs=[[{"type": "indicator"} for _ in range(len(kpi_data))]]
        )
        
        # Add gauge charts for each KPI
        for i, (kpi_key, kpi) in enumerate(kpi_data.items()):
            # Determine if higher is better or lower is better
            higher_is_better = kpi_key == "efficiency"
            
            # Calculate performance percentage (0-100)
            if higher_is_better:
                # For efficiency, higher is better
                performance = min(100, (kpi["value"] / kpi["target"]) * 100)
            else:
                # For energy, safety, cost, lower is better
                performance = max(0, (1 - (kpi["value"] / kpi["target"])) * 100)
            
            # Define gauge chart ranges
            if higher_is_better:
                steps = [
                    {'range': [0, 70], 'color': 'rgba(255, 0, 0, 0.3)'},
                    {'range': [70, 90], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [90, 100], 'color': 'rgba(0, 255, 0, 0.3)'}
                ]
            else:
                steps = [
                    {'range': [0, 70], 'color': 'rgba(255, 0, 0, 0.3)'},
                    {'range': [70, 90], 'color': 'rgba(255, 255, 0, 0.3)'},
                    {'range': [90, 100], 'color': 'rgba(0, 255, 0, 0.3)'}
                ]
            
            # Create gauge chart
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=kpi["value"],
                    delta={'reference': kpi["target"], 'relative': True, 'valueformat': '.1%'},
                    number={'suffix': kpi["unit"], 'valueformat': '.2f'},
                    title={'text': kpi["title"]},
                    gauge={
                        'axis': {'range': [0, kpi["target"] * 1.5], 'ticksuffix': kpi["unit"]},
                        'bar': {'color': kpi["color"]},
                        'steps': steps,
                        'threshold': {
                            'line': {'color': "black", 'width': 3},
                            'thickness': 0.75,
                            'value': kpi["target"]
                        }
                    }
                ),
                row=1, col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title="Process Performance KPI Dashboard",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Apply theme (limited for indicator charts)
        fig.update_layout(
            font=viz_manager.config.get('theme', {}).get('font', {'family': 'Arial, sans-serif', 'size': 12}),
            paper_bgcolor=viz_manager.config.get('theme', {}).get('paper_bgcolor', 'white')
        )
        
        # Save visualization
        viz_manager.save_visualization(fig, "kpi_dashboard")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating KPI dashboard visualization: {e}")
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating KPI dashboard visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="red", size=14)
        )
        
        # Save this error visualization
        viz_manager.save_visualization(fig, "kpi_dashboard")
        
        return fig
    

@timer_decorator
def generate_all_visualizations(
    data_dir: str = "dashboard/data", 
    output_dir: str = "dashboard/charts",
    config_file: str = "visualization_config.yaml",
    save_html: bool = True,
    save_json: bool = True,
    create_config: bool = False,
    log_to_file: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Generate all visualizations for the dashboard
    
    Args:
        data_dir: Directory containing the processed data files
        output_dir: Directory to save visualizations
        config_file: Path to the configuration file
        save_html: Whether to save HTML files
        save_json: Whether to save JSON files
        create_config: Whether to create a config template file if it doesn't exist
        log_to_file: Whether to log to a file
        debug: Whether to enable debug logging
        
    Returns:
        Dictionary of visualization results
    """
    # Configure logging based on parameters
    global logger
    logger = setup_logging(log_to_file=log_to_file, debug=debug)
    
    start_time = time.time()
    logger.info(f"Starting visualization generation...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # 1. Load configuration
        config = load_config(config_file, create_template=create_config)
        
        # 2. Create visualization manager
        viz_manager = VisualizationManager(output_dir, config, save_html=save_html, save_json=save_json)
        
        # 3. Load processed data
        datasets = load_processed_data(data_dir)
        
        # 4. Get visualizations to generate
        visualizations_config = config.get('visualizations', {})
        
        # 5. Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 6. Track created visualizations
        result = {}
        
        # 7. Create visualizations with robust error handling
        # Process efficiency comparison
        if visualizations_config.get('process_efficiency', True):
            try:
                result["process_efficiency"] = create_process_efficiency_comparison(datasets, viz_manager)
                logger.info("Successfully created process efficiency visualization")
            except Exception as e:
                logger.error(f"Failed to create process efficiency visualization: {e}")
        
        # Energy safety relationship
        if visualizations_config.get('energy_safety', True):
            try:
                result["energy_safety"] = create_energy_safety_relationship(datasets, viz_manager)
                logger.info("Successfully created energy safety visualization")
            except Exception as e:
                logger.error(f"Failed to create energy safety visualization: {e}")
        
        # Process hierarchy
        if visualizations_config.get('process_hierarchy', True):
            try:
                result["process_hierarchy"] = create_process_hierarchy(datasets, viz_manager)
                logger.info("Successfully created process hierarchy visualization")
            except Exception as e:
                logger.error(f"Failed to create process hierarchy visualization: {e}")
        
        # Safety optimization
        if visualizations_config.get('safety_optimization', True):
            try:
                result["safety_heatmap"] = create_safety_heatmap(datasets, viz_manager)
                result["process_step_safety"] = create_process_step_safety(datasets, viz_manager)
                logger.info("Successfully created safety optimization visualizations")
            except Exception as e:
                logger.error(f"Failed to create safety optimization visualizations: {e}")
        
        # Catalyst analysis
        if visualizations_config.get('catalyst_analysis', True):
            try:
                result["catalyst_matrix"], result["catalyst_parallel"] = create_catalyst_analysis(datasets, viz_manager)
                logger.info("Successfully created catalyst analysis visualizations")
            except Exception as e:
                logger.error(f"Failed to create catalyst analysis visualizations: {e}")
        
        # ROI projection
        if visualizations_config.get('roi_projection', True):
            try:
                result["roi_by_process"] = create_roi_projection(datasets, viz_manager)
                logger.info("Successfully created ROI projection visualization")
            except Exception as e:
                logger.error(f"Failed to create ROI projection visualization: {e}")
        
        # Energy by process
        if visualizations_config.get('energy_by_process', True):
            try:
                result["energy_by_process"] = create_energy_by_process(datasets, viz_manager)
                logger.info("Successfully created energy by process visualization")
            except Exception as e:
                logger.error(f"Failed to create energy by process visualization: {e}")

        # Energy parameters
        if visualizations_config.get('energy_parameters', True):
            try:
                result["energy_parameters"] = create_energy_parameters(datasets, viz_manager)
                logger.info("Successfully created energy parameters visualization")
            except Exception as e:
                logger.error(f"Failed to create energy parameters visualization: {e}")

        # Parameter correlation
        if visualizations_config.get('parameter_correlation', True):
            try:
                result["parameter_correlation"] = create_parameter_correlation(datasets, viz_manager)
                logger.info("Successfully created parameter correlation visualization")
            except Exception as e:
                logger.error(f"Failed to create parameter correlation visualization: {e}")

        # Temperature pressure
        if visualizations_config.get('temp_pressure', True):
            try:
                result["temp_pressure"] = create_temp_pressure(datasets, viz_manager)
                logger.info("Successfully created temperature-pressure visualization")
            except Exception as e:
                logger.error(f"Failed to create temperature-pressure visualization: {e}")

        # Duration impact
        if visualizations_config.get('duration_impact', True):
            try:
                result["duration_impact"] = create_duration_impact(datasets, viz_manager)
                logger.info("Successfully created duration impact visualization")
            except Exception as e:
                logger.error(f"Failed to create duration impact visualization: {e}")

        # KPI dashboard
        if visualizations_config.get('kpi_dashboard', True):
            try:
                result["kpi_dashboard"] = create_kpi_dashboard(datasets, viz_manager)
                logger.info("Successfully created KPI dashboard visualization")
            except Exception as e:
                logger.error(f"Failed to create KPI dashboard visualization: {e}")

        # Efficiency correlation
        if visualizations_config.get('efficiency_correlation', True):
            try:
                result["efficiency_correlation"] = create_efficiency_correlation(datasets, viz_manager)
                logger.info("Successfully created efficiency correlation visualization")
            except Exception as e:
                logger.error(f"Failed to create efficiency correlation visualization: {e}")
        
        # 8. Save visualization metadata
        metadata = {
            'visualization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_directory': data_dir,
            'output_directory': output_dir,
            'config_file': config_file,
            'visualizations_created': list(viz_manager.created_visualizations.keys()),
            'execution_time_seconds': round(time.time() - start_time, 2)
        }
        
        with open(f"{output_dir}/visualization_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Generated {len(viz_manager.created_visualizations)} visualizations in {time.time() - start_time:.2f} seconds")
        logger.info(f"All visualizations saved to {output_dir}")
        
        return result
    
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise RuntimeError(f"Visualization generation failed: {str(e)}")

def parse_arguments():
    """Parse command line arguments with improved options"""
    parser = argparse.ArgumentParser(description='SOCAR Process Analysis Visualization Generator')
    parser.add_argument('--data-dir', '-d', type=str, default='dashboard/data',
                       help='Directory containing the processed data files')
    parser.add_argument('--output-dir', '-o', type=str, default='dashboard/charts',
                       help='Directory to save visualizations')
    parser.add_argument('--config', '-c', type=str, default='visualization_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--no-html', action='store_true',
                       help='Skip generating HTML files')
    parser.add_argument('--no-json', action='store_true',
                       help='Skip generating JSON files')
    parser.add_argument('--create-config', action='store_true',
                       help='Create config template file if it doesn\'t exist')
    parser.add_argument('--log-to-file', action='store_true',
                       help='Enable logging to file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Generate all visualizations
    generate_all_visualizations(
        args.data_dir, 
        args.output_dir, 
        args.config,
        save_html=not args.no_html,
        save_json=not args.no_json,
        create_config=args.create_config,
        log_to_file=args.log_to_file,
        debug=args.debug
    )