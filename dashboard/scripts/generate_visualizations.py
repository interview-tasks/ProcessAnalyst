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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("visualization_generation")

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
    
    def __init__(self, output_dir: str, config: Dict):
        """
        Initialize visualization manager
        
        Args:
            output_dir: Directory to save visualizations
            config: Visualization configuration
        """
        self.output_dir = output_dir
        self.config = config
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set default visualization settings
        self.default_settings = config.get('default_settings', {})
        
        # Track created visualizations
        self.created_visualizations = {}
    
    def save_visualization(self, fig: go.Figure, name: str) -> Tuple[str, str]:
        """
        Save visualization as HTML and JSON
        
        Args:
            fig: Plotly figure to save
            name: Name for the visualization files
            
        Returns:
            Tuple of (html_path, json_path)
        """
        html_path = f"{self.output_dir}/{name}.html"
        json_path = f"{self.output_dir}/{name}.json"
        
        try:
            # Save as HTML with config from settings
            html_config = self.default_settings.get('html_config', {
                'responsive': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            })
            
            fig.write_html(html_path, config=html_config)
            
            # Save as JSON
            with open(json_path, "w") as f:
                f.write(fig.to_json())
                
            self.created_visualizations[name] = {
                'html_path': html_path,
                'json_path': json_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return html_path, json_path
        
        except Exception as e:
            logger.error(f"Error saving visualization {name}: {e}")
            raise
    
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
            margin=theme.get('margin', {'l': 60, 'r': 60, 't': 60, 'b': 60}),
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

def load_config(config_file: str = 'visualization_config.yaml') -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to the YAML configuration file
        
    Returns:
        Configuration as a dictionary
    """
    # Default configuration
    default_config = {
        'theme': {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'rgba(240, 240, 240, 0.5)',
            'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60},
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
            'kpi_dashboard': True
        },
        'process_efficiency': {
            'height': 500,
            'efficiency_range': [85, 100],
            'title': 'Process Efficiency vs. Safety by Type',
            'annotations': [
                {'x': 0.5, 'y': -0.15, 'text': 'Higher efficiency doesn\'t always mean safer processes'}
            ]
        },
        'energy_safety': {
            'height': 600,
            'title': 'Energy Consumption vs. Safety Incidents',
            'trendline': 'ols',
            'risk_zones': True
        }
        # Other visualization-specific configs would go here
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
            with open(f"{config_file}.template", 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Saved default configuration template to {config_file}.template")
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
    required_files = ["processed_data.csv", "process_types.csv"]
    
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
        
        # Load main dataset
        main_file = os.path.join(data_dir, "processed_data.csv")
        if os.path.exists(main_file):
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
            except Exception as e:
                logger.error(f"Error loading main dataset: {e}")
                raise
        
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
                logger.error(f"Error loading {name}: {e}")
        
        # Validate integrity between datasets
        if "df" in datasets and "process_types" in datasets:
            main_process_types = set(datasets["df"]["Proses Tipi"].unique())
            agg_process_types = set(datasets["process_types"]["Proses Tipi"].unique())
            
            if not main_process_types == agg_process_types:
                logger.warning(f"Process type mismatch between main and aggregated datasets")
                logger.warning(f"Main dataset has {len(main_process_types)} types, aggregated has {len(agg_process_types)}")
        
        return datasets
    
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise

@timer_decorator
def create_process_efficiency_comparison(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager,
    config: Dict
) -> go.Figure:
    """
    Create process efficiency comparison visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        config: Configuration dictionary
        
    Returns:
        Plotly figure
    """
    logger.info("Creating process efficiency comparison visualization")
    
    # Get visualization-specific config
    viz_config = config.get('process_efficiency', {})
    
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
            "title": viz_config.get("title", "Process Efficiency vs. Safety by Type"),
            "xaxis": {"title": "Process Type"},
            "yaxis": {
                "title": "Efficiency (%)",
                "range": viz_config.get("efficiency_range", [85, 100])  # Start from 85% to emphasize differences
            },
            "height": viz_config.get("height", 500),
            "margin": viz_config.get("margin", dict(l=50, r=50, t=60, b=50)),
            "hovermode": "x unified",
        }
        
        if incident_column:
            layout_updates["yaxis2"] = {
                "title": "Safety Incidents",
                "overlaying": "y",
                "side": "right",
                "rangemode": "nonnegative"
            }
        
        # Add annotations if specified
        if "annotations" in viz_config:
            layout_updates["annotations"] = viz_config["annotations"]
        
        fig.update_layout(**layout_updates)
        
        # Apply theme
        fig = viz_manager.apply_theme(fig, 'bar')
        
        # Save visualizations
        viz_manager.save_visualization(fig, "process_efficiency")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating process efficiency comparison: {e}", exc_info=True)
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating visualization. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@timer_decorator
def create_energy_safety_relationship(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager,
    config: Dict
) -> go.Figure:
    """
    Create energy consumption vs safety incidents visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        config: Configuration dictionary
        
    Returns:
        Plotly figure
    """
    logger.info("Creating energy vs safety relationship visualization")
    
    # Get visualization-specific config
    viz_config = config.get('energy_safety', {})
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if required columns exist
        required_cols = ["Energy_per_ton", "Təhlükəsizlik Hadisələri", "Proses Tipi", "Emal Həcmi (ton)"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for energy safety visualization: {missing_cols}")
            # Try to create Energy_per_ton if missing
            if "Energy_per_ton" in missing_cols and "Enerji İstifadəsi (kWh)" in df.columns and "Emal Həcmi (ton)" in df.columns:
                df["Energy_per_ton"] = df["Enerji İstifadəsi (kWh)"] / df["Emal Həcmi (ton)"]
                missing_cols.remove("Energy_per_ton")
            
            if missing_cols:
                raise ValueError(f"Cannot create energy safety visualization due to missing columns: {missing_cols}")
        
        # Create a more sophisticated scatter plot with trendline
        hover_data = {
            "Emalın Səmərəliliyi (%)": True,
            "Prosesin Müddəti (saat)": True
        }
        
        # Add optional hover data if available
        for col in ["Təzyiq (bar)", "Temperatur (°C)"]:
            if col in df.columns:
                hover_data[col] = True
        
        # Create figure using Plotly Express for the main scatter plot
        if viz_config.get("trendline"):
            fig = px.scatter(
                df, 
                x="Energy_per_ton", 
                y="Təhlükəsizlik Hadisələri",
                color="Proses Tipi",
                size="Emal Həcmi (ton)",
                hover_name="Proses Addımı" if "Proses Addımı" in df.columns else None,
                hover_data=hover_data,
                title=viz_config.get("title", "Energy Consumption vs. Safety Incidents"),
                labels={
                    "Energy_per_ton": "Energy per Ton (kWh/ton)", 
                    "Təhlükəsizlik Hadisələri": "Safety Incidents",
                    "Proses Tipi": "Process Type",
                    "Emal Həcmi (ton)": "Processing Volume (tons)"
                },
                trendline=viz_config.get("trendline"),
                trendline_scope=viz_config.get("trendline_scope", "overall"),
                trendline_color_override="red"
            )
        else:
            # Create using go.Figure for more customization if no trendline
            fig = go.Figure()
            
            # Add scatter points grouped by process type
            for process_type in df["Proses Tipi"].unique():
                process_df = df[df["Proses Tipi"] == process_type]
                
                fig.add_trace(go.Scatter(
                    x=process_df["Energy_per_ton"],
                    y=process_df["Təhlükəsizlik Hadisələri"],
                    mode="markers",
                    name=process_type,
                    marker=dict(
                        size=process_df["Emal Həcmi (ton)"] / process_df["Emal Həcmi (ton)"].max() * 20 + 5,
                        sizemode="diameter",
                        sizeref=2.0 * df["Emal Həcmi (ton)"].max() / (40.**2),
                        sizemin=4
                    ),
                    text=process_df["Proses Addımı"] if "Proses Addımı" in process_df.columns else None,
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Energy: %{x:.2f} kWh/ton<br>" +
                        "Incidents: %{y}<br>" +
                        "Volume: %{marker.size:.0f} tons" +
                        "<extra></extra>"
                    )
                ))
            
            # Add title and axis labels
            fig.update_layout(
                title=viz_config.get("title", "Energy Consumption vs. Safety Incidents"),
                xaxis_title="Energy per Ton (kWh/ton)",
                yaxis_title="Safety Incidents"
            )
        
        # Improve layout
        fig.update_layout(
            height=viz_config.get("height", 600),
            margin=viz_config.get("margin", dict(l=50, r=50, t=60, b=50)),
            hovermode="closest"
        )
        
        # Add regions for risk zones if specified in config
        if viz_config.get("risk_zones", False):
            # Add high risk zone
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
        logger.error(f"Error creating energy safety relationship: {e}", exc_info=True)
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating visualization. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@timer_decorator
def create_process_hierarchy(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager,
    config: Dict
) -> go.Figure:
    """
    Create process hierarchy visualization with sunburst chart
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        config: Configuration dictionary
        
    Returns:
        Plotly figure
    """
    logger.info("Creating process hierarchy visualization")
    
    # Get visualization-specific config
    viz_config = config.get('process_hierarchy', {})
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if required columns exist
        required_cols = ["Proses Tipi", "Proses Addımı", "Emalın Səmərəliliyi (%)", "Emal Həcmi (ton)"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for process hierarchy visualization: {missing_cols}")
            raise ValueError(f"Cannot create process hierarchy visualization due to missing columns: {missing_cols}")
        
        # Group data by process type and step
        param_impact = df.groupby(['Proses Tipi', 'Proses Addımı']).agg({
            'Emalın Səmərəliliyi (%)': 'mean',
            'Emal Həcmi (ton)': 'sum'
        })
        
        # Add energy and safety metrics if available
        if 'Energy_per_ton' in df.columns:
            param_impact['Energy_per_ton'] = df.groupby(['Proses Tipi', 'Proses Addımı'])['Energy_per_ton'].mean()
            
        if 'Təhlükəsizlik Hadisələri' in df.columns:
            param_impact['Təhlükəsizlik Hadisələri'] = df.groupby(['Proses Tipi', 'Proses Addımı'])['Təhlükəsizlik Hadisələri'].sum()
        
        # Reset index for sunburst processing
        param_impact = param_impact.reset_index()
        
        # Prepare hover data
        hover_data = ['Emalın Səmərəliliyi (%)']
        for col in ['Energy_per_ton', 'Təhlükəsizlik Hadisələri']:
            if col in param_impact.columns:
                hover_data.append(col)
        
        # Create a more informative sunburst chart
        fig = px.sunburst(
            param_impact, 
            path=['Proses Tipi', 'Proses Addımı'], 
            values='Emal Həcmi (ton)',
            color='Emalın Səmərəliliyi (%)',
            color_continuous_scale=viz_config.get('colorscale', 'RdBu'),
            range_color=viz_config.get('color_range', [85, 100]),  # Set color range to emphasize differences
            hover_data=hover_data,
            title=viz_config.get('title', 'Process Hierarchy, Efficiency and Volume')
        )
        
        # Improve layout
        fig.update_layout(
            height=viz_config.get('height', 650),
            margin=viz_config.get('margin', dict(l=50, r=50, t=60, b=50)),
            coloraxis_colorbar=dict(
                title=viz_config.get('colorbar_title', "Efficiency (%)")
            )
        )
        
        # Apply any additional layout updates from config
        if 'layout_updates' in viz_config:
            fig.update_layout(**viz_config['layout_updates'])
        
        # Save visualizations
        viz_manager.save_visualization(fig, "process_hierarchy")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating process hierarchy: {e}", exc_info=True)
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating visualization. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@timer_decorator
def create_catalyst_analysis(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager,
    config: Dict
) -> Tuple[go.Figure, go.Figure]:
    """
    Create catalyst performance analysis visualizations
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        config: Configuration dictionary
        
    Returns:
        Tuple of Plotly figures (parallel coords, matrix)
    """
    logger.info("Creating catalyst analysis visualizations")
    
    # Get visualization-specific config
    viz_config = config.get('catalyst_analysis', {})
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if catalyst column exists
        if 'İstifadə Edilən Katalizatorlar' not in df.columns:
            logger.warning("Missing catalyst column for catalyst analysis")
            raise ValueError("Cannot create catalyst analysis visualizations due to missing catalyst column")
        
        # Aggregate catalyst performance
        agg_dict = {
            'Emalın Səmərəliliyi (%)': 'mean',
            'Emal Həcmi (ton)': 'sum'
        }
        
        # Add additional metrics if available
        for col in ['Energy_per_ton', 'CO2_per_ton', 'Təhlükəsizlik Hadisələri']:
            if col in df.columns:
                agg_dict[col] = ['sum', 'mean'] if col == 'Təhlükəsizlik Hadisələri' else 'mean'
        
        # Group by catalyst
        catalyst_data = df.groupby('İstifadə Edilən Katalizatorlar').agg(agg_dict)
        
        # Flatten column names if needed
        if isinstance(catalyst_data.columns, pd.MultiIndex):
            catalyst_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in catalyst_data.columns]
        
        # Reset index for easier processing
        catalyst_data = catalyst_data.reset_index()
        
        # Identify the incident rate column
        incident_rate_col = next((col for col in catalyst_data.columns if col.endswith('mean') and 'Təhlükəsizlik' in col), None)
        
        # If we have incident rate, scale it to percentage for better visualization
        if incident_rate_col:
            incident_scaled_col = 'Incident_Rate'
            catalyst_data[incident_scaled_col] = catalyst_data[incident_rate_col] * 100
        else:
            incident_scaled_col = None
        
        # Get column names for dimensions
        efficiency_col = 'Emalın Səmərəliliyi (%)' if 'Emalın Səmərəliliyi (%)' in catalyst_data.columns else next((col for col in catalyst_data.columns if 'Efficiency' in col), None)
        energy_col = 'Energy_per_ton' if 'Energy_per_ton' in catalyst_data.columns else None
        co2_col = 'CO2_per_ton' if 'CO2_per_ton' in catalyst_data.columns else None
        
        # VISUALIZATION 1: Parallel coordinates chart
        dimensions = []
        
        # Add efficiency dimension
        if efficiency_col:
            dimensions.append(dict(
                range=[catalyst_data[efficiency_col].min() * 0.95, catalyst_data[efficiency_col].max() * 1.05], 
                label='Efficiency (%)', 
                values=catalyst_data[efficiency_col]
            ))
        
        # Add energy dimension
        if energy_col:
            dimensions.append(dict(
                range=[catalyst_data[energy_col].min() * 0.95, catalyst_data[energy_col].max() * 1.05], 
                label='Energy (kWh/ton)', 
                values=catalyst_data[energy_col]
            ))
        
        # Add CO2 dimension
        if co2_col:
            dimensions.append(dict(
                range=[catalyst_data[co2_col].min() * 0.95, catalyst_data[co2_col].max() * 1.05], 
                label='CO₂ (kg/ton)', 
                values=catalyst_data[co2_col]
            ))
        
        # Add incident rate dimension
        if incident_scaled_col:
            dimensions.append(dict(
                range=[0, max(100, catalyst_data[incident_scaled_col].max() * 1.1)], 
                label='Incident Rate (%)', 
                values=catalyst_data[incident_scaled_col]
            ))
        
        # Create parallel coordinates plot if we have enough dimensions
        if len(dimensions) >= 2:
            fig1 = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=catalyst_data[efficiency_col] if efficiency_col else catalyst_data.index,
                        colorscale=viz_config.get('colorscale', 'Viridis'),
                        showscale=True,
                        colorbar=dict(title='Efficiency (%)' if efficiency_col else 'Index')
                    ),
                    dimensions=dimensions,
                    labelfont=dict(size=12),
                    tickfont=dict(size=10)
                )
            )
            
            # Add a title
            fig1.update_layout(
                title=viz_config.get('parallel_title', "Catalyst Performance Comparison"),
                height=viz_config.get('height', 600),
                margin=viz_config.get('margin', dict(l=100, r=50, t=60, b=50))
            )
        else:
            # Create a fallback figure
            fig1 = go.Figure()
            fig1.add_annotation(
                text="Insufficient metrics for parallel coordinates visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # VISUALIZATION 2: Scatter plot matrix
        # Check if we have the necessary columns
        if energy_col and efficiency_col and 'Emal Həcmi (ton)' in catalyst_data.columns:
            fig2 = px.scatter(
                catalyst_data,
                x=energy_col,
                y=efficiency_col,
                size='Emal Həcmi (ton)',
                color=incident_scaled_col if incident_scaled_col else None,
                hover_name='İstifadə Edilən Katalizatorlar',
                labels={
                    energy_col: 'Energy Consumption (kWh/ton)',
                    efficiency_col: 'Process Efficiency (%)',
                    'Emal Həcmi (ton)': 'Processing Volume (tons)',
                    incident_scaled_col: 'Incident Rate (%)' if incident_scaled_col else None
                },
                title=viz_config.get('matrix_title', 'Catalyst Performance Matrix'),
                color_continuous_scale=viz_config.get('color_continuous_scale', 'RdYlGn_r')  # Red for high incident rate, green for low
            )
            
            # Improve layout
            fig2.update_layout(
                height=viz_config.get('height', 600),
                margin=viz_config.get('margin', dict(l=50, r=50, t=60, b=50)),
                xaxis=dict(title='Energy Consumption (kWh/ton)'),
                yaxis=dict(title='Process Efficiency (%)', range=[catalyst_data[efficiency_col].min() * 0.95, 100])
            )
        else:
            # Create a fallback figure
            fig2 = go.Figure()
            fig2.add_annotation(
                text="Insufficient metrics for catalyst matrix visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Apply theme to both figures
        fig1 = viz_manager.apply_theme(fig1, 'scatter')
        fig2 = viz_manager.apply_theme(fig2, 'scatter')
        
        # Save visualizations
        viz_manager.save_visualization(fig1, "catalyst_parallel")
        viz_manager.save_visualization(fig2, "catalyst_matrix")
        
        return fig1, fig2
    
    except Exception as e:
        logger.error(f"Error creating catalyst analysis: {e}", exc_info=True)
        # Create minimal versions to avoid breaking the dashboard
        fig1 = go.Figure()
        fig1.add_annotation(
            text="Error creating catalyst parallel visualization. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        fig2 = go.Figure()
        fig2.add_annotation(
            text="Error creating catalyst matrix visualization. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        return fig1, fig2

@timer_decorator
def create_parameter_correlation(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager,
    config: Dict
) -> Tuple[go.Figure, go.Figure]:
    """
    Create parameter correlation visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        config: Configuration dictionary
        
    Returns:
        Tuple of Plotly figures (correlation matrix, efficiency correlation)
    """
    logger.info("Creating parameter correlation visualizations")
    
    # Get visualization-specific config
    viz_config = config.get('parameter_correlation', {})
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Select relevant numeric columns for correlation
        base_cols = [
            'Temperatur (°C)', 'Təzyiq (bar)', 'Prosesin Müddəti (saat)',
            'Emalın Səmərəliliyi (%)'
        ]
        
        # Filter to columns that exist in the dataset
        numeric_cols = [col for col in base_cols if col in df.columns]
        
        # Add derived metrics if available
        for col in ['Energy_per_ton', 'CO2_per_ton', 'Təhlükəsizlik Hadisələri']:
            if col in df.columns:
                numeric_cols.append(col)
        
        if len(numeric_cols) < 3:
            logger.warning(f"Not enough numeric columns for correlation. Found only: {numeric_cols}")
            raise ValueError("Cannot create correlation matrix with fewer than 3 columns")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # VISUALIZATION 1: Correlation matrix heatmap
        fig1 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=viz_config.get('colorscale', 'RdBu_r'),
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": viz_config.get('textfont_size', 10)}
        ))
        
        # Update layout
        fig1.update_layout(
            title=viz_config.get('matrix_title', "Parameter Correlation Matrix"),
            height=viz_config.get('height', 700),
            margin=viz_config.get('margin', dict(l=50, r=50, t=60, b=50)),
            xaxis=dict(tickangle=45)
        )
        
        # VISUALIZATION 2: Top correlations with efficiency
        if 'Emalın Səmərəliliyi (%)' in numeric_cols:
            efficiency_corr = corr_matrix['Emalın Səmərəliliyi (%)'].drop('Emalın Səmərəliliyi (%)').sort_values(ascending=False)
            
            # Create horizontal bar chart
            fig2 = go.Figure(data=go.Bar(
                x=efficiency_corr.values,
                y=efficiency_corr.index,
                orientation='h',
                marker=dict(
                    color=efficiency_corr.values,
                    colorscale=viz_config.get('colorscale2', 'RdBu'),
                    cmin=-1, cmax=1
                ),
                text=np.round(efficiency_corr.values, 2),
                textposition='auto'
            ))
            
            # Update layout
            fig2.update_layout(
                title=viz_config.get('efficiency_title', "Parameter Correlation with Process Efficiency"),
                height=viz_config.get('height2', 500),
                margin=viz_config.get('margin2', dict(l=150, r=50, t=60, b=50)),
                xaxis=dict(
                    title="Correlation Coefficient",
                    range=[-1, 1]
                ),
                yaxis=dict(title="Parameter")
            )
        else:
            # Create a fallback figure
            fig2 = go.Figure()
            fig2.add_annotation(
                text="Efficiency column not available for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Apply theme
        fig1 = viz_manager.apply_theme(fig1, 'heatmap')
        fig2 = viz_manager.apply_theme(fig2, 'bar')
        
        # Save visualizations
        viz_manager.save_visualization(fig1, "parameter_correlation")
        viz_manager.save_visualization(fig2, "efficiency_correlation")
        
        return fig1, fig2
    
    except Exception as e:
        logger.error(f"Error creating parameter correlation: {e}", exc_info=True)
        # Create minimal versions to avoid breaking the dashboard
        fig1 = go.Figure()
        fig1.add_annotation(
            text="Error creating parameter correlation matrix. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        fig2 = go.Figure()
        fig2.add_annotation(
            text="Error creating efficiency correlation chart. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        return fig1, fig2

@timer_decorator
def create_roi_projection(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager,
    config: Dict
) -> Tuple[go.Figure, go.Figure]:
    """
    Create ROI projection visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        config: Configuration dictionary
        
    Returns:
        Tuple of Plotly figures (waterfall, by process)
    """
    logger.info("Creating ROI projection visualizations")
    
    # Get visualization-specific config
    viz_config = config.get('roi_projection', {})
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Check if we have ROI dataset or need to calculate it
        if "roi_projections" in datasets:
            roi_data = datasets["roi_projections"]
            logger.info("Using precomputed ROI projections")
        else:
            logger.info("Computing ROI projections from main dataset")
            
            # Check if required columns exist
            required_cols = ['Proses Tipi', 'Emal Həcmi (ton)', 'Əməliyyat Xərcləri (AZN)']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for ROI calculation: {missing_cols}")
                raise ValueError(f"Cannot calculate ROI due to missing columns: {missing_cols}")
            
            # Calculate potential savings for each process type
            process_savings = df.groupby('Proses Tipi').agg({
                'Emal Həcmi (ton)': 'sum',
                'Əməliyyat Xərcləri (AZN)': 'sum'
            })
            
            # Add efficiency and energy metrics if available
            if 'Energy_per_ton' in df.columns:
                energy_agg = df.groupby('Proses Tipi')['Energy_per_ton'].agg(['min', 'mean'])
                process_savings = process_savings.join(energy_agg)
            
            if 'Emalın Səmərəliliyi (%)' in df.columns:
                efficiency_agg = df.groupby('Proses Tipi')['Emalın Səmərəliliyi (%)'].agg(['max', 'mean'])
                process_savings = process_savings.join(efficiency_agg)
            
            # Reset index
            process_savings = process_savings.reset_index()
            
            # Calculate improvement potential
            if 'min' in process_savings.columns and 'mean' in process_savings.columns:
                process_savings['Energy_Improvement'] = (
                    (process_savings['mean'] - process_savings['min']) / 
                    process_savings['mean'] * 100
                )
            else:
                # Default to 15% improvement if we can't calculate
                process_savings['Energy_Improvement'] = 15
            
            if 'max' in process_savings.columns and 'mean' in process_savings.columns:
                process_savings['Efficiency_Improvement'] = (
                    (process_savings['max'] - process_savings['mean']) / 
                    process_savings['mean'] * 100
                )
            else:
                # Default to 8% improvement if we can't calculate
                process_savings['Efficiency_Improvement'] = 8
            
            # Calculate annual costs
            process_savings['Annual_Cost'] = process_savings['Əməliyyat Xərcləri (AZN)'] * 12  # annualized
            
            # Energy savings: 30% of costs are energy-related
            process_savings['Energy_Savings'] = (
                process_savings['Annual_Cost'] * 0.3 * process_savings['Energy_Improvement'] / 100
            )
            
            # Efficiency savings: 40% of costs are affected by efficiency
            process_savings['Efficiency_Savings'] = (
                process_savings['Annual_Cost'] * 0.4 * process_savings['Efficiency_Improvement'] / 100
            )
            
            # Safety savings: 10% of costs for every 10% reduction in incidents
            # Assuming 20% incident reduction across all processes for simplicity
            process_savings['Safety_Savings'] = process_savings['Annual_Cost'] * 0.1 * 0.2
            
            # Total potential savings
            process_savings['Total_Savings'] = (
                process_savings['Energy_Savings'] + 
                process_savings['Efficiency_Savings'] + 
                process_savings['Safety_Savings']
            )
            
            # Investment estimate (1.5x annual savings)
            process_savings['Investment'] = process_savings['Total_Savings'] * 1.5
            
            # Calculate ROI (months to recoup investment)
            process_savings['ROI_Months'] = (
                process_savings['Investment'] / (process_savings['Total_Savings'] / 12)
            )
            
            roi_data = process_savings
        
        # Create waterfall chart showing savings breakdown
        savings_data = []
        
        # Start with operating costs
        total_opex = roi_data['Annual_Cost'].sum() if 'Annual_Cost' in roi_data.columns else roi_data['Əməliyyat Xərcləri (AZN)'].sum() * 12
        
        if 'Total_Savings' in roi_data.columns:
            total_savings = roi_data['Total_Savings'].sum() 
        else:
            # Estimate if not directly available
            total_savings = total_opex * 0.15  # Assume 15% savings
        
        if 'Energy_Savings' in roi_data.columns and 'Efficiency_Savings' in roi_data.columns and 'Safety_Savings' in roi_data.columns:
            energy_savings = roi_data['Energy_Savings'].sum()
            efficiency_savings = roi_data['Efficiency_Savings'].sum()
            safety_savings = roi_data['Safety_Savings'].sum()
        else:
            # Estimate if not available
            energy_savings = total_savings * 0.4
            efficiency_savings = total_savings * 0.4
            safety_savings = total_savings * 0.2
        
        # Build waterfall data
        savings_data = [
            {"category": "Current Annual Costs", "value": total_opex},
            {"category": "Energy Optimization", "value": -energy_savings},
            {"category": "Efficiency Improvement", "value": -efficiency_savings},
            {"category": "Safety Enhancement", "value": -safety_savings},
            {"category": "Optimized Annual Costs", "value": total_opex - total_savings}
        ]
        
        # Convert to dataframe
        savings_df = pd.DataFrame(savings_data)
        
        # VISUALIZATION 1: Waterfall chart
        fig1 = go.Figure(go.Waterfall(
            name="Cost Waterfall", 
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=savings_df['category'],
            y=savings_df['value'],
            textposition="outside",
            text=[f"{val:,.0f} AZN" for val in savings_df['value']],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#2ca02c"}},  # Green for savings
            increasing={"marker": {"color": "#d62728"}},  # Red for costs
            totals={"marker": {"color": "#1f77b4"}}      # Blue for totals
        ))
        
        # Update layout
        fig1.update_layout(
            title=viz_config.get('waterfall_title', "Annual Cost Savings Potential"),
            showlegend=False,
            height=viz_config.get('height', 600),
            margin=viz_config.get('margin', dict(l=50, r=50, t=60, b=50)),
            yaxis=dict(title="Annual Cost (AZN)")
        )
        
        # Add annotations
        fig1.add_annotation(
            x=4, y=savings_df['value'].iloc[-1] * 1.1,
            text=f"{total_savings:,.0f} AZN Potential Annual Savings ({total_savings/total_opex*100:.1f}%)",
            showarrow=False,
            font=dict(size=14, color="#2ca02c")
        )
        
        # VISUALIZATION 2: ROI comparison chart by process type
        # Check if we have the necessary columns
        if 'ROI_Months' in roi_data.columns and 'Proses Tipi' in roi_data.columns and 'Total_Savings' in roi_data.columns:
            roi_by_process = roi_data[['Proses Tipi', 'Total_Savings', 'ROI_Months']]
            
            # Create subplot with two y-axes
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add savings bars
            fig2.add_trace(
                go.Bar(
                    x=roi_by_process['Proses Tipi'],
                    y=roi_by_process['Total_Savings'],
                    name="Annual Savings",
                    marker_color="#2ca02c",
                    text=[f"{val:,.0f} AZN" for val in roi_by_process['Total_Savings']],
                    textposition="auto"
                ),
                secondary_y=False
            )
            
            # Add ROI months line
            fig2.add_trace(
                go.Scatter(
                    x=roi_by_process['Proses Tipi'],
                    y=roi_by_process['ROI_Months'],
                    name="Payback Period",
                    line=dict(color="#ff7f0e", width=3),
                    mode='lines+markers+text',
                    text=[f"{val:.1f} months" for val in roi_by_process['ROI_Months']],
                    textposition="top center"
                ),
                secondary_y=True
            )
            
            # Update layout
            fig2.update_layout(
                title=viz_config.get('roi_by_process_title', "ROI Analysis by Process Type"),
                height=viz_config.get('height2', 500),
                margin=viz_config.get('margin2', dict(l=50, r=50, t=60, b=50)),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig2.update_yaxes(title_text="Annual Savings (AZN)", secondary_y=False)
            fig2.update_yaxes(title_text="Payback Period (months)", secondary_y=True)
        else:
            # Create a fallback figure
            fig2 = go.Figure()
            fig2.add_annotation(
                text="Insufficient data for ROI by process visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Apply theme
        fig1 = viz_manager.apply_theme(fig1, 'bar')
        fig2 = viz_manager.apply_theme(fig2, 'bar')
        
        # Save visualizations
        viz_manager.save_visualization(fig1, "roi_waterfall")
        viz_manager.save_visualization(fig2, "roi_by_process")
        
        return fig1, fig2
    
    except Exception as e:
        logger.error(f"Error creating ROI projection: {e}", exc_info=True)
        # Create minimal versions to avoid breaking the dashboard
        fig1 = go.Figure()
        fig1.add_annotation(
            text="Error creating ROI waterfall chart. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        fig2 = go.Figure()
        fig2.add_annotation(
            text="Error creating ROI by process chart. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        return fig1, fig2

@timer_decorator
def create_safety_optimization(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager,
    config: Dict
) -> Tuple[go.Figure, go.Figure]:
    """
    Create safety optimization visualization
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        config: Configuration dictionary
        
    Returns:
        Tuple of Plotly figures (safety heatmap, process step safety)
    """
    logger.info("Creating safety optimization visualizations")
    
    # Get visualization-specific config
    viz_config = config.get('safety_optimization', {})
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # VISUALIZATION 1: Temperature-pressure safety matrix
        if "safety_parameters" in datasets:
            safety_data = datasets["safety_parameters"]
            logger.info("Using precomputed safety parameters")
        elif "Temperature_Category" in df.columns and "Pressure_Category" in df.columns:
            logger.info("Using temperature and pressure categories from main dataset")
            
            # Create from main dataset with categories
            agg_dict = {"Emal Həcmi (ton)": "sum"}
            
            # Determine incident measure
            if "Has_Incident" in df.columns:
                agg_dict["Has_Incident"] = "mean"
                incident_col = "Has_Incident"
            else:
                agg_dict["Təhlükəsizlik Hadisələri"] = ["sum", "mean"]
                incident_col = "Təhlükəsizlik Hadisələri"
            
            safety_data = df.groupby(['Temperature_Category', 'Pressure_Category']).agg(agg_dict)
            
            # Flatten columns if needed
            if isinstance(safety_data.columns, pd.MultiIndex):
                safety_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in safety_data.columns]
            
            # Reset index
            safety_data = safety_data.reset_index()
            
            # Determine incident rate column name
            if "Has_Incident" in safety_data.columns:
                incident_rate_col = "Has_Incident"
            elif "Təhlükəsizlik Hadisələri_mean" in safety_data.columns:
                incident_rate_col = "Təhlükəsizlik Hadisələri_mean"
            else:
                # Create it
                volume_col = next((col for col in safety_data.columns if "volume" in col.lower() or "həcmi" in col.lower()), None)
                incidents_col = next((col for col in safety_data.columns if "incidents" in col.lower() or "hadisələri" in col.lower()), None)
                
                if volume_col and incidents_col:
                    safety_data["Incident_Rate"] = safety_data[incidents_col] / safety_data[volume_col] * 100
                    incident_rate_col = "Incident_Rate"
                else:
                    # Default
                    incident_rate_col = safety_data.columns[2]  # Guess the correct column
        else:
            logger.info("Creating temperature and pressure categories")
            
            # Check if the temperature and pressure columns exist
            if 'Temperatur (°C)' not in df.columns or 'Təzyiq (bar)' not in df.columns:
                logger.warning("Missing temperature or pressure columns for safety visualization")
                raise ValueError("Cannot create safety visualization due to missing temperature or pressure columns")
            
            # Create categories for temperature and pressure
            temp_bins = viz_config.get('temp_bins', [0, 150, 300, 450, float('inf')])
            temp_labels = viz_config.get('temp_labels', ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)'])
            
            pressure_bins = viz_config.get('pressure_bins', [0, 10, 30, 50, float('inf')])
            pressure_labels = viz_config.get('pressure_labels', ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)'])
            
            df['Temperature_Category'] = pd.cut(df['Temperatur (°C)'], bins=temp_bins, labels=temp_labels)
            df['Pressure_Category'] = pd.cut(df['Təzyiq (bar)'], bins=pressure_bins, labels=pressure_labels)
            
            # Determine incident measure and aggregation
            agg_dict = {"Emal Həcmi (ton)": "sum"}
            
            if "Has_Incident" in df.columns:
                agg_dict["Has_Incident"] = "mean"
                incident_col = "Has_Incident"
                incident_rate_col = "Has_Incident"
            else:
                agg_dict["Təhlükəsizlik Hadisələri"] = ["sum", "mean"]
                incident_col = "Təhlükəsizlik Hadisələri"
                incident_rate_col = "Təhlükəsizlik Hadisələri_mean"
            
            # Create aggregated data
            safety_data = df.groupby(['Temperature_Category', 'Pressure_Category']).agg(agg_dict)
            
            # Flatten columns
            if isinstance(safety_data.columns, pd.MultiIndex):
                safety_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in safety_data.columns]
            
            # Reset index
            safety_data = safety_data.reset_index()
        
        # Create a pivot table
        # Make sure we have the right column names
        temp_col = 'Temperature_Category' if 'Temperature_Category' in safety_data.columns else safety_data.columns[0]
        pressure_col = 'Pressure_Category' if 'Pressure_Category' in safety_data.columns else safety_data.columns[1]
        
        # Determine incident rate column
        if "Incident_Rate" in safety_data.columns:
            incident_rate_col = "Incident_Rate"
        elif "Has_Incident" in safety_data.columns:
            incident_rate_col = "Has_Incident"
            # Scale to percentage
            safety_data[incident_rate_col] = safety_data[incident_rate_col] * 100
        else:
            # Use the first numeric column after the categories
            non_cat_cols = [col for col in safety_data.columns if col not in [temp_col, pressure_col]]
            incident_rate_col = non_cat_cols[0] if non_cat_cols else safety_data.columns[2]
        
        pivot_data = safety_data.pivot_table(
            index=temp_col, 
            columns=pressure_col, 
            values=incident_rate_col,
            aggfunc='mean'
        )
        
        # Reorder categories if possible
        try:
            temp_order = ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)']
            pressure_order = ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)']
            
            pivot_data = pivot_data.reindex(temp_order, axis=0)
            pivot_data = pivot_data.reindex(pressure_order, axis=1)
        except:
            logger.info("Could not reorder category levels for safety heatmap")
        
        # Create heatmap
        fig1 = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=viz_config.get('colorscale', 'Reds'),
            text=np.round(pivot_data.values, 1),
            texttemplate="%{text}%",
            textfont={"size": viz_config.get('textfont_size', 10)},
            colorbar=dict(title="Incident Rate (%)")
        ))
        
        # Update layout
        fig1.update_layout(
            title=viz_config.get('heatmap_title', "Safety Incident Rate by Temperature and Pressure"),
            height=viz_config.get('height', 600),
            margin=viz_config.get('margin', dict(l=50, r=50, t=60, b=50)),
            xaxis=dict(title="Pressure Category"),
            yaxis=dict(title="Temperature Category")
        )
        
        # Add annotation for optimal safety zone if configured
        if viz_config.get('show_optimal_zone', True):
            # Find the minimum value in the pivot table
            min_val = pivot_data.min().min()
            min_idx = np.unravel_index(np.nanargmin(pivot_data.values), pivot_data.shape)
            min_temp = pivot_data.index[min_idx[0]]
            min_pressure = pivot_data.columns[min_idx[1]]
            
            fig1.add_annotation(
                x=min_pressure,
                y=min_temp,
                text="Optimal Safety Zone",
                showarrow=True,
                font=dict(size=12, color="green"),
                arrowhead=2,
                ax=0,
                ay=-40
            )
        
        # VISUALIZATION 2: Process step safety comparison
        if "process_steps" in datasets:
            step_safety = datasets["process_steps"]
            logger.info("Using precomputed process steps data")
        else:
            logger.info("Calculating process step safety data")
            
            # Check if process step column exists
            if 'Proses Addımı' not in df.columns:
                logger.warning("Missing process step column for safety visualization")
                raise ValueError("Cannot create process step safety visualization due to missing process step column")
            
            # Determine aggregation
            agg_dict = {
                'Emal Həcmi (ton)': 'sum',
                'Emalın Səmərəliliyi (%)': 'mean'
            }
            
            if 'Energy_per_ton' in df.columns:
                agg_dict['Energy_per_ton'] = 'mean'
                
            if 'Təhlükəsizlik Hadisələri' in df.columns:
                agg_dict['Təhlükəsizlik Hadisələri'] = ['sum', 'mean']
                
            step_safety = df.groupby('Proses Addımı').agg(agg_dict)
            
            # Flatten columns if needed
            if isinstance(step_safety.columns, pd.MultiIndex):
                step_safety.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in step_safety.columns]
                
            # Reset index
            step_safety = step_safety.reset_index()
        
        # Calculate incident rate if not already present
        if 'Has_Incident' in df.columns and 'Proses Addımı' in df.columns:
            # Check if we need to add this metric
            if 'Has_Incident' not in step_safety.columns:
                incident_by_step = df.groupby('Proses Addımı')['Has_Incident'].mean() * 100
                step_safety = step_safety.merge(
                    incident_by_step.reset_index().rename(columns={'Has_Incident': 'Incident_Rate'}), 
                    on='Proses Addımı'
                )
                incident_rate_col = 'Incident_Rate'
            else:
                # Scale to percentage if needed
                if step_safety['Has_Incident'].max() <= 1:
                    step_safety['Incident_Rate'] = step_safety['Has_Incident'] * 100
                else:
                    step_safety['Incident_Rate'] = step_safety['Has_Incident']
                incident_rate_col = 'Incident_Rate'
        elif 'Təhlükəsizlik Hadisələri_mean' in step_safety.columns:
            # Calculate incidents per volume
            if 'Emal Həcmi (ton)' in step_safety.columns:
                step_safety['Incident_Rate'] = step_safety['Təhlükəsizlik Hadisələri_mean'] / step_safety['Emal Həcmi (ton)'] * 100
            else:
                step_safety['Incident_Rate'] = step_safety['Təhlükəsizlik Hadisələri_mean']
            incident_rate_col = 'Incident_Rate'
        else:
            # Try to find suitable column
            incident_rate_col = next((col for col in step_safety.columns if 'incident' in col.lower() or 'hadisələri' in col.lower()), None)
            if not incident_rate_col:
                incident_rate_col = step_safety.columns[1]  # Use second column as fallback
        
        # Sort by incident rate
        step_safety = step_safety.sort_values(incident_rate_col, ascending=False)
        
        # Create horizontal bar chart
        fig2 = go.Figure()
        
        # Add incident rate bars
        fig2.add_trace(go.Bar(
            y=step_safety['Proses Addımı'],
            x=step_safety[incident_rate_col],
            name="Incident Rate",
            orientation='h',
            marker=dict(
                color=step_safety[incident_rate_col],
                colorscale='Reds',
                colorbar=dict(title="Incident Rate")
            ),
            text=np.round(step_safety[incident_rate_col], 1),
            textposition="auto",
            texttemplate="%{text}%"
        ))
        
        # Update layout
        fig2.update_layout(
            title=viz_config.get('process_title', "Safety Incident Rate by Process Step"),
            height=viz_config.get('height2', 500),
            margin=viz_config.get('margin2', dict(l=150, r=50, t=60, b=50)),
            xaxis=dict(
                title="Incident Rate (%)",
                range=[0, max(100, step_safety[incident_rate_col].max() * 1.1)]
            ),
            yaxis=dict(title="Process Step")
        )
        
        # Apply theme
        fig1 = viz_manager.apply_theme(fig1, 'heatmap')
        fig2 = viz_manager.apply_theme(fig2, 'bar')
        
        # Save visualizations
        viz_manager.save_visualization(fig1, "safety_heatmap")
        viz_manager.save_visualization(fig2, "process_step_safety")
        
        return fig1, fig2
    
    except Exception as e:
        logger.error(f"Error creating safety optimization: {e}", exc_info=True)
        # Create minimal versions to avoid breaking the dashboard
        fig1 = go.Figure()
        fig1.add_annotation(
            text="Error creating safety heatmap. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        fig2 = go.Figure()
        fig2.add_annotation(
            text="Error creating process step safety chart. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        
        return fig1, fig2

@timer_decorator
def create_kpi_dashboard(
    datasets: Dict[str, pd.DataFrame], 
    viz_manager: VisualizationManager,
    config: Dict
) -> go.Figure:
    """
    Create KPI summary dashboard
    
    Args:
        datasets: Dictionary of DataFrames
        viz_manager: Visualization manager instance
        config: Configuration dictionary
        
    Returns:
        Plotly figure
    """
    logger.info("Creating KPI dashboard visualization")
    
    # Get visualization-specific config
    viz_config = config.get('kpi_dashboard', {})
    
    try:
        # Get the main dataset
        df = datasets["df"]
        
        # Calculate KPIs
        overall_efficiency = df['Emalın Səmərəliliyi (%)'].mean()
        
        # Get total incidents and volume - handling different column names
        if 'Təhlükəsizlik Hadisələri' in df.columns:
            total_incidents = df['Təhlükəsizlik Hadisələri'].sum()
        else:
            incident_col = next((col for col in df.columns if 'incident' in col.lower() or 'hadisələri' in col.lower()), None)
            total_incidents = df[incident_col].sum() if incident_col else 0
        
        total_volume = df['Emal Həcmi (ton)'].sum()
        
        # Get energy metric
        if 'Energy_per_ton' in df.columns:
            avg_energy = df['Energy_per_ton'].mean()
        elif 'Enerji İstifadəsi (kWh)' in df.columns and 'Emal Həcmi (ton)' in df.columns:
            avg_energy = df['Enerji İstifadəsi (kWh)'].sum() / total_volume
        else:
            avg_energy = 1.8  # Fallback value
        
        # Calculate incident rate
        incident_rate = total_incidents / total_volume * 1000  # Incidents per 1000 tons
        
        # Get best and worst processes if process efficiency column exists
        if 'Emalın Səmərəliliyi (%)' in df.columns:
            try:
                best_idx = df['Emalın Səmərəliliyi (%)'].idxmax()
                worst_idx = df['Emalın Səmərəliliyi (%)'].idxmin()
                
                best_process = df.loc[best_idx]
                worst_process = df.loc[worst_idx]
                
                improvement_potential = best_process['Emalın Səmərəliliyi (%)'] - overall_efficiency
            except:
                logger.warning("Could not identify best/worst processes")
                improvement_potential = 5  # Fallback value
        else:
            improvement_potential = 5  # Fallback value
        
        # Create gauge charts for KPIs
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}]
            ],
            subplot_titles=("Overall Process Efficiency", "Safety Incident Rate", 
                           "Average Energy Consumption", "Process Improvement Potential")
        )
        
        # Add efficiency gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_efficiency,
                title={"text": "Efficiency (%)"},
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": viz_config.get('efficiency_range', [85, 100])},
                    "bar": {"color": viz_config.get('efficiency_color', "darkblue")},
                    "steps": [
                        {"range": [85, 90], "color": "red"},
                        {"range": [90, 95], "color": "yellow"},
                        {"range": [95, 100], "color": "green"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": viz_config.get('efficiency_target', 95)
                    }
                }
            ),
            row=1, col=1
        )
        
        # Add safety gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=incident_rate,
                title={"text": "Incidents per 1000 tons"},
                number={"valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": viz_config.get('safety_color', "darkblue")},
                    "steps": [
                        {"range": [0, 2], "color": "green"},
                        {"range": [2, 5], "color": "yellow"},
                        {"range": [5, 10], "color": "red"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": viz_config.get('safety_target', 3)
                    }
                }
            ),
            row=1, col=2
        )
        
        # Add energy gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_energy,
                title={"text": "Energy (kWh/ton)"},
                number={"valueformat": ".2f"},
                gauge={
                    "axis": {"range": [1, 3]},
                    "bar": {"color": viz_config.get('energy_color', "darkblue")},
                    "steps": [
                        {"range": [1, 1.5], "color": "green"},
                        {"range": [1.5, 2], "color": "yellow"},
                        {"range": [2, 3], "color": "red"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": viz_config.get('energy_target', 1.6)
                    }
                }
            ),
            row=2, col=1
        )
        
        # Add improvement potential gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_efficiency,
                delta={"reference": 100, "increasing": {"color": "green"}},
                title={"text": "Improvement Potential"},
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [85, 100]},
                    "bar": {"color": viz_config.get('improvement_color', "darkblue")},
                    "steps": [
                        {"range": [85, 90], "color": "red"},
                        {"range": [90, 95], "color": "yellow"},
                        {"range": [95, 100], "color": "green"}
                    ],
                    "threshold": {
                        "line": {"color": "green", "width": 4},
                        "thickness": 0.75,
                        "value": overall_efficiency + improvement_potential  # Potential target
                    }
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=viz_config.get('title', "SOCAR Process Performance Dashboard"),
            height=viz_config.get('height', 800),
            margin=viz_config.get('margin', dict(l=50, r=50, t=60, b=50))
        )
        
        # Save visualizations
        viz_manager.save_visualization(fig, "kpi_dashboard")
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating KPI dashboard: {e}", exc_info=True)
        # Create a minimal version to avoid breaking the dashboard
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating KPI dashboard. See logs for details.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

@timer_decorator
def generate_all_visualizations(
    data_dir: str = "data", 
    output_dir: str = "charts",
    config_file: str = "visualization_config.yaml"
) -> Dict[str, Any]:
    """
    Generate all visualizations for the dashboard
    
    Args:
        data_dir: Directory containing the processed data files
        output_dir: Directory to save visualizations
        config_file: Path to the configuration file
        
    Returns:
        Dictionary of visualization results
    """
    start_time = time.time()
    logger.info(f"Starting visualization generation...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # 1. Load configuration
        config = load_config(config_file)
        
        # 2. Create visualization manager
        viz_manager = VisualizationManager(output_dir, config)
        
        # 3. Load processed data
        datasets = load_processed_data(data_dir)
        
        # 4. Get visualizations to generate
        visualizations_config = config.get('visualizations', {})
        
        # Track created visualizations
        result = {}
        
        # 5. Create all enabled visualizations
        if visualizations_config.get('process_efficiency', True):
            result["process_efficiency"] = create_process_efficiency_comparison(datasets, viz_manager, config)
        
        if visualizations_config.get('energy_safety', True):
            result["energy_safety"] = create_energy_safety_relationship(datasets, viz_manager, config)
        
        if visualizations_config.get('process_hierarchy', True):
            result["process_hierarchy"] = create_process_hierarchy(datasets, viz_manager, config)
        
        if visualizations_config.get('catalyst_analysis', True):
            result["catalyst_analysis"] = create_catalyst_analysis(datasets, viz_manager, config)
        
        if visualizations_config.get('parameter_correlation', True):
            result["parameter_correlation"] = create_parameter_correlation(datasets, viz_manager, config)
        
        if visualizations_config.get('roi_projection', True):
            result["roi_projection"] = create_roi_projection(datasets, viz_manager, config)
        
        if visualizations_config.get('safety_optimization', True):
            result["safety_optimization"] = create_safety_optimization(datasets, viz_manager, config)
        
        if visualizations_config.get('kpi_dashboard', True):
            result["kpi_dashboard"] = create_kpi_dashboard(datasets, viz_manager, config)
        
        # 6. Save visualization metadata
        metadata = {
            'visualization_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_directory': data_dir,
            'output_directory': output_dir,
            'config_file': config_file,
            'visualizations_created': list(result.keys()),
            'execution_time_seconds': round(time.time() - start_time, 2)
        }
        
        with open(f"{output_dir}/visualization_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Generated {len(result)} visualization sets in {time.time() - start_time:.2f} seconds")
        logger.info(f"All visualizations saved to {output_dir}")
        
        return result
    
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SOCAR Process Analysis Visualization Generator')
    parser.add_argument('--data-dir', '-d', type=str, default='socar-dashboard/data',
                       help='Directory containing the processed data files')
    parser.add_argument('--output-dir', '-o', type=str, default='socar-dashboard/charts',
                       help='Directory to save visualizations')
    parser.add_argument('--config', '-c', type=str, default='visualization_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()

# if __name__ == "__main__":
#     # Parse command line arguments
#     args = parse_arguments()
    
#     # Set debug level if requested
#     if args.debug:
#         logger.setLevel(logging.DEBUG)
#         logger.debug("Debug logging enabled")
    
#     # Generate all visualizations
#     generate_all_visualizations(args.data_dir, args.output_dir, args.config)
def create_html_report(output_dir: str, visualizations: Dict[str, Any], config: Dict) -> str:
    """
    Create a simple HTML report with all visualizations
    
    Args:
        output_dir: Directory containing the visualization files
        visualizations: Dictionary of visualization results
        config: Configuration dictionary
        
    Returns:
        Path to the generated HTML report
    """
    logger.info("Creating HTML visualization report")
    
    # Get report config
    report_config = config.get('report', {})
    report_title = report_config.get('title', 'SOCAR Process Analysis Visualization Report')
    
    # Create HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2a5885;
        }}
        .report-header {{
            background-color: #2a5885;
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            font-size: 0.8em;
            color: #f0f0f0;
        }}
        .visualization-section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .viz-container {{
            width: 100%;
            height: 600px;
            margin: 20px 0;
            border: 1px solid #ddd;
        }}
        footer {{
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
    <!-- Include Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="report-header">
        <h1>{report_title}</h1>
        <p class="timestamp">Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="visualization-sections">
'''
    
    # Add visualization sections
    viz_sections = {
        'process_efficiency': {
            'title': 'Process Efficiency Analysis',
            'description': 'Analysis of process efficiency across different process types with comparison to safety incidents.'
        },
        'energy_safety': {
            'title': 'Energy and Safety Relationship',
            'description': 'Examination of the relationship between energy consumption and safety incidents.'
        },
        'process_hierarchy': {
            'title': 'Process Hierarchy',
            'description': 'Hierarchical view of process types and steps, showing efficiency distribution and volume.'
        },
        'catalyst_analysis': {
            'title': 'Catalyst Performance Analysis',
            'description': 'Comparative analysis of different catalysts and their impact on process performance.'
        },
        'parameter_correlation': {
            'title': 'Parameter Correlation Analysis',
            'description': 'Correlation analysis between different process parameters and their impact on efficiency.'
        },
        'roi_projection': {
            'title': 'ROI Projection Analysis',
            'description': 'Projection of potential return on investment for process improvements.'
        },
        'safety_optimization': {
            'title': 'Safety Optimization Analysis',
            'description': 'Analysis of safety incidents based on process parameters for optimization opportunities.'
        },
        'kpi_dashboard': {
            'title': 'KPI Dashboard',
            'description': 'Summary dashboard of key performance indicators across all processes.'
        }
    }
    
    # Build each section
    for viz_type, viz_data in viz_sections.items():
        if viz_type in visualizations:
            viz_files = [f for f in os.listdir(output_dir) if f.startswith(viz_type.split('_')[0]) and f.endswith('.html')]
            
            html_content += f'''
    <div class="visualization-section" id="{viz_type}">
        <h2>{viz_data['title']}</h2>
        <p>{viz_data['description']}</p>
'''
            
            # Add iframe for each visualization file
            for viz_file in viz_files:
                html_content += f'''
        <h3>{viz_file.replace('.html', '').replace('_', ' ').title()}</h3>
        <div class="viz-container">
            <iframe src="{viz_file}" width="100%" height="100%" frameborder="0"></iframe>
        </div>
'''
            
            html_content += '''
    </div>
'''
    
    # Close HTML
    html_content += '''
    </div>
    
    <footer>
        <p>SOCAR Process Analysis Visualization Report</p>
        <p>Generated with Python and Plotly</p>
    </footer>
</body>
</html>
'''
    
    # Write HTML to file
    report_path = os.path.join(output_dir, 'visualization_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report created: {report_path}")
    return report_path

def profile_visualization_performance(func):
    """
    Decorator to profile the performance of visualization functions
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        import io
        
        # Create profiler
        profiler = cProfile.Profile()
        
        # Start profiling
        profiler.enable()
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Stop profiling
        profiler.disable()
        
        # Get statistics
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions by cumulative time
        
        # Log profiling results
        logger.info(f"Performance profile for {func.__name__}:")
        for line in s.getvalue().split('\n'):
            if line.strip():
                logger.info(line)
        
        return result
    
    return wrapper

def compare_with_previous_run(current_output_dir: str, previous_output_dir: str) -> Dict:
    """
    Compare current visualization run with a previous run
    
    Args:
        current_output_dir: Directory containing current visualizations
        previous_output_dir: Directory containing previous visualizations
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing with previous run: {previous_output_dir}")
    
    # Check if previous directory exists
    if not os.path.exists(previous_output_dir):
        logger.warning(f"Previous output directory not found: {previous_output_dir}")
        return {'error': 'Previous directory not found'}
    
    # Get list of visualization files in both directories
    current_files = {f for f in os.listdir(current_output_dir) if f.endswith('.json')}
    previous_files = {f for f in os.listdir(previous_output_dir) if f.endswith('.json')}
    
    # Compare visualizations
    comparison = {
        'new_visualizations': list(current_files - previous_files),
        'removed_visualizations': list(previous_files - current_files),
        'common_visualizations': list(current_files & previous_files),
        'changes': []
    }
    
    # Compare common visualizations
    for viz_file in comparison['common_visualizations']:
        try:
            with open(os.path.join(current_output_dir, viz_file), 'r') as f:
                current_viz = json.load(f)
            
            with open(os.path.join(previous_output_dir, viz_file), 'r') as f:
                previous_viz = json.load(f)
            
            # Compare data points (simplified)
            if 'data' in current_viz and 'data' in previous_viz:
                current_points = sum(len(trace.get('y', [])) for trace in current_viz['data'])
                previous_points = sum(len(trace.get('y', [])) for trace in previous_viz['data'])
                
                change_pct = ((current_points - previous_points) / previous_points * 100) if previous_points > 0 else 0
                
                comparison['changes'].append({
                    'file': viz_file,
                    'current_points': current_points,
                    'previous_points': previous_points,
                    'change_percent': round(change_pct, 2)
                })
        except Exception as e:
            logger.error(f"Error comparing visualization {viz_file}: {e}")
            comparison['changes'].append({
                'file': viz_file,
                'error': str(e)
            })
    
    # Log comparison results
    logger.info(f"Comparison results: {len(comparison['new_visualizations'])} new, "
               f"{len(comparison['removed_visualizations'])} removed, "
               f"{len(comparison['common_visualizations'])} common")
    
    return comparison

def create_dashboard_index(output_dir: str, visualizations: Dict[str, Any], config: Dict) -> str:
    """
    Create a dashboard index HTML file that lists all available visualizations
    
    Args:
        output_dir: Directory containing the visualization files
        visualizations: Dictionary of visualization results
        config: Configuration dictionary
        
    Returns:
        Path to the generated index file
    """
    logger.info("Creating dashboard index file")
    
    # Get dashboard config
    dashboard_config = config.get('dashboard', {})
    dashboard_title = dashboard_config.get('title', 'SOCAR Process Analysis Dashboard')
    
    # Create HTML content
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dashboard_title}</title>
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            padding-top: 60px;
            background-color: #f8f9fa;
        }}
        .navbar {{
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
        }}
        .viz-card {{
            margin-bottom: 20px;
            transition: transform 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,.05);
        }}
        .viz-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,.1);
        }}
        .card-img-top {{
            height: 200px;
            object-fit: cover;
            background-color: #e9ecef;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary fixed-top">
        <div class="container">
            <a class="navbar-brand" href="#">{dashboard_title}</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="#process-efficiency">Efficiency</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#energy-safety">Energy & Safety</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#process-hierarchy">Process Hierarchy</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#roi-projection">ROI Projection</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="visualization_report.html">Full Report</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- Page Content -->
    <div class="container">
        <header class="py-4 text-center">
            <h1 class="display-4">{dashboard_title}</h1>
            <p class="lead">Interactive visualizations for process analysis and optimization</p>
        </header>

        <hr>

        <!-- Visualization Categories -->
'''
    
    # Define visualization categories and their descriptions
    categories = {
        'process-efficiency': {
            'title': 'Process Efficiency',
            'description': 'Visualizations related to process efficiency analysis and comparisons.',
            'icon': 'bi-graph-up'
        },
        'energy-safety': {
            'title': 'Energy & Safety',
            'description': 'Explore the relationship between energy consumption and safety incidents.',
            'icon': 'bi-lightning-charge'
        },
        'process-hierarchy': {
            'title': 'Process Structure',
            'description': 'Hierarchical visualizations of process types, steps, and their relationships.',
            'icon': 'bi-diagram-3'
        },
        'catalyst-analysis': {
            'title': 'Catalyst Analysis',
            'description': 'Analysis of different catalysts and their performance metrics.',
            'icon': 'bi-radioactive'
        },
        'parameter-correlation': {
            'title': 'Parameter Correlation',
            'description': 'Correlation analysis between process parameters and efficiency.',
            'icon': 'bi-grid-3x3'
        },
        'roi-projection': {
            'title': 'ROI Projection',
            'description': 'Projected return on investment for process improvements.',
            'icon': 'bi-currency-dollar'
        },
        'safety-optimization': {
            'title': 'Safety Optimization',
            'description': 'Identifying optimal parameters for improved safety performance.',
            'icon': 'bi-shield-check'
        },
        'kpi-dashboard': {
            'title': 'KPI Dashboard',
            'description': 'Key performance indicators at a glance.',
            'icon': 'bi-speedometer2'
        }
    }
    
    # Get all visualization files
    viz_files = [f for f in os.listdir(output_dir) if f.endswith('.html') and f != 'visualization_report.html' and f != 'dashboard_index.html']
    
    # Group files by category
    viz_by_category = {}
    for category in categories:
        viz_by_category[category] = [f for f in viz_files if f.startswith(category.replace('-', '_')) or f.split('_')[0] in category]
    
    # Add categories and visualizations
    for category, info in categories.items():
        files = viz_by_category.get(category, [])
        if files:
            html_content += f'''
        <section id="{category}" class="mb-5">
            <h2><i class="bi {info['icon']}"></i> {info['title']}</h2>
            <p>{info['description']}</p>
            
            <div class="row">
'''
            
            # Add cards for each visualization in the category
            for viz_file in files:
                # Create a nice title from the filename
                viz_title = viz_file.replace('.html', '').replace('_', ' ').title()
                
                html_content += f'''
                <div class="col-md-4">
                    <div class="card viz-card">
                        <div class="card-img-top d-flex align-items-center justify-content-center bg-light text-primary p-4">
                            <i class="bi {info['icon']} display-1"></i>
                        </div>
                        <div class="card-body">
                            <h5 class="card-title">{viz_title}</h5>
                            <p class="card-text">Interactive visualization for {viz_title.lower()} analysis.</p>
                            <a href="{viz_file}" class="btn btn-primary" target="_blank">View Visualization</a>
                        </div>
                    </div>
                </div>
'''
            
            html_content += '''
            </div>
        </section>
'''
    
    # Close HTML
    html_content += '''
    </div>

    <!-- Footer -->
    <footer class="py-4 bg-dark text-white mt-5">
        <div class="container text-center">
            <p>SOCAR Process Analysis Dashboard &copy; 2025</p>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Bootstrap Icons -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
</body>
</html>
'''
    
    # Write HTML to file
    index_path = os.path.join(output_dir, 'dashboard_index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Dashboard index created: {index_path}")
    return index_path

@timer_decorator
def generate_summary_report(data_dir: str, output_dir: str, visualizations: Dict[str, Any], config: Dict) -> str:
    """
    Generate a summary report of the data analysis and visualizations
    
    Args:
        data_dir: Directory containing the processed data files
        output_dir: Directory containing the visualization files
        visualizations: Dictionary of visualization results
        config: Configuration dictionary
        
    Returns:
        Path to the generated report file
    """
    logger.info("Generating summary report")
    
    # Load datasets for statistics
    try:
        df = pd.read_csv(os.path.join(data_dir, 'processed_data.csv'))
        
        # Basic statistics
        process_types = df['Proses Tipi'].nunique() if 'Proses Tipi' in df.columns else 0
        process_steps = df['Proses Addımı'].nunique() if 'Proses Addımı' in df.columns else 0
        total_volume = df['Emal Həcmi (ton)'].sum() if 'Emal Həcmi (ton)' in df.columns else 0
        avg_efficiency = df['Emalın Səmərəliliyi (%)'].mean() if 'Emalın Səmərəliliyi (%)' in df.columns else 0
        
        # Get key metrics
        metrics = {
            'Total Records': len(df),
            'Process Types': process_types,
            'Process Steps': process_steps,
            'Total Volume (tons)': f"{total_volume:,.0f}",
            'Average Efficiency': f"{avg_efficiency:.2f}%",
            'Visualizations Created': len(visualizations),
            'Report Generated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create report
        report_content = {
            'summary': metrics,
            'visualizations': {k: type(v).__name__ for k, v in visualizations.items()},
            'data_overview': {
                'columns': list(df.columns),
                'row_count': len(df),
                'numeric_columns': list(df.select_dtypes(include=['number']).columns),
                'categorical_columns': list(df.select_dtypes(include=['object']).columns)
            }
        }
        
        # Save report
        report_path = os.path.join(output_dir, 'analysis_summary.json')
        with open(report_path, 'w') as f:
            json.dump(report_content, f, indent=4)
        
        logger.info(f"Summary report created: {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return ""

@timer_decorator
def check_visualization_quality(output_dir: str, visualizations: Dict[str, Any], config: Dict) -> Dict:
    """
    Check the quality of generated visualizations
    
    Args:
        output_dir: Directory containing the visualization files
        visualizations: Dictionary of visualization results
        config: Configuration dictionary
        
    Returns:
        Dictionary with quality check results
    """
    logger.info("Checking visualization quality")
    
    quality_results = {
        'checks_performed': 0,
        'all_files_present': True,
        'file_sizes_ok': True,
        'json_validity': True,
        'html_validity': True,
        'issues_found': []
    }
    
    # Check if all visualization files are present
    expected_files = []
    for viz_type in visualizations.keys():
        expected_files.append(f"{viz_type.split('_')[0]}.html")
        expected_files.append(f"{viz_type.split('_')[0]}.json")
    
    actual_files = os.listdir(output_dir)
    missing_files = [f for f in expected_files if f not in actual_files]
    
    if missing_files:
        quality_results['all_files_present'] = False
        quality_results['issues_found'].append(f"Missing files: {missing_files}")
    
    quality_results['checks_performed'] += 1
    
    # Check file sizes
    for file_name in actual_files:
        if file_name.endswith('.html') or file_name.endswith('.json'):
            file_path = os.path.join(output_dir, file_name)
            file_size = os.path.getsize(file_path)
            
            if file_size < 100:  # If file is too small, likely empty or corrupt
                quality_results['file_sizes_ok'] = False
                quality_results['issues_found'].append(f"File too small: {file_name} ({file_size} bytes)")
    
    quality_results['checks_performed'] += 1
    
    # Check JSON validity
    for file_name in actual_files:
        if file_name.endswith('.json'):
            file_path = os.path.join(output_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                quality_results['json_validity'] = False
                quality_results['issues_found'].append(f"Invalid JSON: {file_name} - {str(e)}")
    
    quality_results['checks_performed'] += 1
    
    # Check HTML structure (basic check)
    for file_name in actual_files:
        if file_name.endswith('.html'):
            file_path = os.path.join(output_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Very basic check for HTML structure
                    if not content.startswith('<!DOCTYPE html>') and not '<html' in content:
                        quality_results['html_validity'] = False
                        quality_results['issues_found'].append(f"Invalid HTML structure: {file_name}")
                        
                    # Check for incomplete HTML
                    if not '</html>' in content:
                        quality_results['html_validity'] = False
                        quality_results['issues_found'].append(f"Incomplete HTML: {file_name}")
                        
                    # Check for script inclusions
                    if not 'plotly' in content.lower():
                        quality_results['issues_found'].append(f"Missing Plotly script: {file_name}")
            except Exception as e:
                quality_results['html_validity'] = False
                quality_results['issues_found'].append(f"Error checking HTML: {file_name} - {str(e)}")
    
    quality_results['checks_performed'] += 1
    
    # Log results
    if quality_results['issues_found']:
        logger.warning(f"Visualization quality check found {len(quality_results['issues_found'])} issues")
        for issue in quality_results['issues_found']:
            logger.warning(f"Quality issue: {issue}")
    else:
        logger.info("Visualization quality check passed")
    
    return quality_results

# Enhanced main function
@timer_decorator
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Add a timestamp to the output directory for versioning if requested
        if args.add_timestamp:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            args.output_dir = f"{args.output_dir}_{timestamp}"
            logger.info(f"Using timestamped output directory: {args.output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate visualizations with profiling if requested
        if args.profile:
            # Wrap the function with profiling decorator
            profiled_generate = profile_visualization_performance(generate_all_visualizations)
            visualizations = profiled_generate(args.data_dir, args.output_dir, args.config)
        else:
            # Normal execution
            visualizations = generate_all_visualizations(args.data_dir, args.output_dir, args.config)
        
        # Load config for additional processing
        config = load_config(args.config)
        
        # Run quality checks if requested
        if args.quality_check:
            quality_results = check_visualization_quality(args.output_dir, visualizations, config)
            
            # Save quality results
            quality_path = os.path.join(args.output_dir, 'quality_check.json')
            with open(quality_path, 'w') as f:
                json.dump(quality_results, f, indent=4)
        
        # Compare with previous run if requested
        if args.compare_with:
            comparison = compare_with_previous_run(args.output_dir, args.compare_with)
            
            # Save comparison results
            comparison_path = os.path.join(args.output_dir, 'comparison.json')
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=4)
        
        # Generate additional outputs
        additional_outputs = []
        
        # Create HTML report if requested
        if args.html_report:
            report_path = create_html_report(args.output_dir, visualizations, config)
            additional_outputs.append(f"HTML Report: {report_path}")
        
        # Create dashboard index if requested
        if args.dashboard_index:
            index_path = create_dashboard_index(args.output_dir, visualizations, config)
            additional_outputs.append(f"Dashboard Index: {index_path}")
        
        # Generate summary report if requested
        if args.summary:
            summary_path = generate_summary_report(args.data_dir, args.output_dir, visualizations, config)
            additional_outputs.append(f"Summary Report: {summary_path}")
        
        # Create single file zip archive if requested
        if args.create_zip:
            # Create zip filename
            zip_path = args.zip_filename if args.zip_filename else f"{args.output_dir}_visualizations.zip"
            
            # Import ZipFile
            from zipfile import ZipFile
            
            # Create zip file
            with ZipFile(zip_path, 'w') as zipf:
                # Add all files from output directory
                for root, _, files in os.walk(args.output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(args.output_dir))
                        zipf.write(file_path, arcname)
            
            additional_outputs.append(f"Zip Archive: {zip_path}")
        
        # If any additional outputs were created, log them
        if additional_outputs:
            logger.info("Additional outputs created:")
            for output in additional_outputs:
                logger.info(f"  - {output}")
        
        # Final success message
        logger.info("Visualization generation completed successfully")
        if args.open_report and 'visualization_report.html' in os.listdir(args.output_dir):
            import webbrowser
            report_path = os.path.join(args.output_dir, 'visualization_report.html')
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            logger.info(f"Opened HTML report in browser: {report_path}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        # Exit with error code
        return 1
    
    return 0

def parse_arguments():
    """Parse command line arguments with more options"""
    parser = argparse.ArgumentParser(description='SOCAR Process Analysis Visualization Generator')
    parser.add_argument('--data-dir', '-d', type=str, default='socar-dashboard/data',
                       help='Directory containing the processed data files')
    parser.add_argument('--output-dir', '-o', type=str, default='socar-dashboard/charts',
                       help='Directory to save visualizations')
    parser.add_argument('--config', '-c', type=str, default='visualization_config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--quality-check', action='store_true',
                       help='Perform quality checks on generated visualizations')
    parser.add_argument('--html-report', action='store_true',
                       help='Generate HTML report with all visualizations')
    parser.add_argument('--dashboard-index', action='store_true',
                       help='Create dashboard index HTML file')
    parser.add_argument('--summary', action='store_true',
                       help='Generate summary report of data and visualizations')
    parser.add_argument('--compare-with', type=str,
                       help='Directory of previous visualization run to compare with')
    parser.add_argument('--add-timestamp', action='store_true',
                       help='Add timestamp to output directory for versioning')
    parser.add_argument('--create-zip', action='store_true',
                       help='Create zip archive of all visualization files')
    parser.add_argument('--zip-filename', type=str,
                       help='Custom filename for the zip archive')
    parser.add_argument('--open-report', action='store_true',
                       help='Open the HTML report in the default browser after generation')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Call enhanced main function
    exit_code = main()
    # Exit with the returned code
    exit(exit_code)