import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path

# Create charts directory if it doesn't exist
Path("charts").mkdir(parents=True, exist_ok=True)

def load_processed_data(data_dir="data"):
    """Load and validate all processed datasets"""
    datasets = {}
    required_files = ["processed_data.csv", "process_types.csv"]
    
    try:
        # Load main dataset
        datasets["df"] = pd.read_csv(f"{data_dir}/processed_data.csv")
        print(f"Loaded main dataset: {datasets['df'].shape[0]} rows, {datasets['df'].shape[1]} columns")
        
        # Load all available aggregated datasets
        for file in os.listdir(data_dir):
            if file.endswith(".csv") and file not in required_files:
                name = file.replace(".csv", "")
                datasets[name] = pd.read_csv(f"{data_dir}/{file}")
                print(f"Loaded {name}: {datasets[name].shape[0]} rows")
        
        # Check if we have the required datasets
        for required in required_files:
            required_name = required.replace(".csv", "")
            if required_name not in datasets and required_name != "df":
                # Try to load it directly
                file_path = f"{data_dir}/{required}"
                if os.path.exists(file_path):
                    datasets[required_name] = pd.read_csv(file_path)
                    print(f"Loaded {required_name}: {datasets[required_name].shape[0]} rows")
                else:
                    print(f"Warning: Required dataset {required_name} not found")
        
        return datasets
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_process_efficiency_comparison(datasets, output_dir="charts"):
    """Create process efficiency comparison visualization"""
    df = datasets.get("process_types", datasets.get("df").groupby("Proses Tipi").agg({
        "Emalın Səmərəliliyi (%)": "mean"
    }).reset_index())
    
    # Create a more informative bar chart with error bars if available
    if "process_types" in datasets:
        process_data = datasets["process_types"]
    else:
        # Calculate from main dataset
        process_data = datasets["df"].groupby("Proses Tipi").agg({
            "Emalın Səmərəliliyi (%)": ["mean", "std"],
            "Emal Həcmi (ton)": "sum",
            "Təhlükəsizlik Hadisələri": "sum"
        }).reset_index()
        process_data.columns = ["Proses Tipi", "Efficiency_Mean", "Efficiency_Std", "Volume", "Incidents"]
        process_data["Emalın Səmərəliliyi (%)"] = process_data["Efficiency_Mean"]
        
    # Create figure
    fig = go.Figure()
    
    # Add bar chart for efficiency
    fig.add_trace(go.Bar(
        x=process_data["Proses Tipi"],
        y=process_data["Emalın Səmərəliliyi (%)"],
        name="Efficiency (%)",
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        text=process_data["Emalın Səmərəliliyi (%)"].round(1).astype(str) + "%",
        textposition="auto"
    ))
    
    # Add safety incidents as a line on secondary axis if available
    if "Təhlükəsizlik Hadisələri" in process_data.columns:
        fig.add_trace(go.Scatter(
            x=process_data["Proses Tipi"],
            y=process_data["Təhlükəsizlik Hadisələri"],
            name="Safety Incidents",
            yaxis="y2",
            line=dict(color="red", width=3),
            marker=dict(size=10)
        ))
    
    # Update layout with second y-axis
    fig.update_layout(
        title="Process Efficiency vs. Safety by Type",
        xaxis=dict(title="Process Type"),
        yaxis=dict(
            title="Efficiency (%)",
            range=[85, 100]  # Start from 85% to emphasize differences
        ),
        yaxis2=dict(
            title="Safety Incidents",
            overlaying="y",
            side="right",
            rangemode="nonnegative"
        ),
        height=500,
        margin=dict(l=50, r=50, t=60, b=50),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                showarrow=False,
                text="Higher efficiency doesn't always mean safer processes",
                xref="paper",
                yref="paper",
                font=dict(size=12)
            )
        ]
    )
    
    # Save visualizations
    fig.write_html(f"{output_dir}/process_efficiency.html")
    with open(f"{output_dir}/process_efficiency.json", "w") as f:
        f.write(fig.to_json())
    
    return fig

def create_energy_safety_relationship(datasets, output_dir="charts"):
    """Create energy consumption vs safety incidents visualization"""
    df = datasets["df"]
    
    # Create a more sophisticated scatter plot with trendline
    fig = px.scatter(
        df, 
        x="Energy_per_ton", 
        y="Təhlükəsizlik Hadisələri",
        color="Proses Tipi",
        size="Emal Həcmi (ton)",
        hover_name="Proses Addımı",
        hover_data={
            "Emalın Səmərəliliyi (%)": True,
            "Prosesin Müddəti (saat)": True,
            "Təzyiq (bar)": True,
            "Temperatur (°C)": True
        },
        title="Energy Consumption vs. Safety Incidents",
        labels={
            "Energy_per_ton": "Energy per Ton (kWh/ton)", 
            "Təhlükəsizlik Hadisələri": "Safety Incidents",
            "Proses Tipi": "Process Type",
            "Emal Həcmi (ton)": "Processing Volume (tons)"
        },
        trendline="ols",  # Add ordinary least squares regression line
        trendline_scope="overall",
        trendline_color_override="red"
    )
    
    # Improve layout
    fig.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                showarrow=False,
                text="Higher energy consumption strongly correlates with increased safety incidents",
                xref="paper",
                yref="paper",
                font=dict(size=12)
            )
        ]
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
    
    # Save visualizations
    fig.write_html(f"{output_dir}/energy_safety.html")
    with open(f"{output_dir}/energy_safety.json", "w") as f:
        f.write(fig.to_json())
    
    return fig

def create_process_hierarchy(datasets, output_dir="charts"):
    """Create process hierarchy visualization with sunburst chart"""
    df = datasets["df"]
    
    # Group data by process type and step
    param_impact = df.groupby(['Proses Tipi', 'Proses Addımı']).agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Energy_per_ton': 'mean',
        'Təhlükəsizlik Hadisələri': 'sum',
        'Emal Həcmi (ton)': 'sum'
    }).reset_index()
    
    # Create a more informative sunburst chart
    fig = px.sunburst(
        param_impact, 
        path=['Proses Tipi', 'Proses Addımı'], 
        values='Emal Həcmi (ton)',
        color='Emalın Səmərəliliyi (%)',
        color_continuous_scale='RdBu',
        range_color=[85, 100],  # Set color range to emphasize differences
        hover_data=['Energy_per_ton', 'Təhlükəsizlik Hadisələri'],
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
    
    # Save visualizations
    fig.write_html(f"{output_dir}/process_hierarchy.html")
    with open(f"{output_dir}/process_hierarchy.json", "w") as f:
        f.write(fig.to_json())
    
    return fig

def create_catalyst_analysis(datasets, output_dir="charts"):
    """Create catalyst performance analysis visualization"""
    df = datasets["df"]
    
    # Aggregate catalyst performance
    catalyst_data = df.groupby('İstifadə Edilən Katalizatorlar').agg({
        'Emalın Səmərəliliyi (%)': 'mean',
        'Energy_per_ton': 'mean',
        'CO2_per_ton': 'mean',
        'Təhlükəsizlik Hadisələri': ['sum', 'mean'],
        'Emal Həcmi (ton)': 'sum'
    }).reset_index()
    
    # Flatten column names
    catalyst_data.columns = ['İstifadə Edilən Katalizatorlar', 'Efficiency', 'Energy_per_ton', 
                            'CO2_per_ton', 'Total_Incidents', 'Incident_Rate', 'Volume']
    
    # Scale incident rate to percentage
    catalyst_data['Incident_Rate'] = catalyst_data['Incident_Rate'] * 100
    
    # Create a parallel coordinates plot for catalyst comparison
    dimensions = [
        dict(range=[85, 100], label='Efficiency (%)', values=catalyst_data['Efficiency']),
        dict(range=[1, 2.5], label='Energy (kWh/ton)', values=catalyst_data['Energy_per_ton']),
        dict(range=[0, 0.6], label='CO₂ (kg/ton)', values=catalyst_data['CO2_per_ton']),
        dict(range=[0, 100], label='Incident Rate (%)', values=catalyst_data['Incident_Rate'])
    ]
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=catalyst_data['Efficiency'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Efficiency (%)')
            ),
            dimensions=dimensions,
            labelfont=dict(size=12),
            tickfont=dict(size=10)
        )
    )
    
    # Add a title
    fig.update_layout(
        title="Catalyst Performance Comparison",
        height=600,
        margin=dict(l=100, r=50, t=60, b=50)
    )
    
    # Save visualizations
    fig.write_html(f"{output_dir}/catalyst_parallel.html")
    with open(f"{output_dir}/catalyst_parallel.json", "w") as f:
        f.write(fig.to_json())
    
    # Create a second visualization showing catalyst performance matrix
    fig2 = px.scatter(
        catalyst_data,
        x='Energy_per_ton',
        y='Efficiency',
        size='Volume',
        color='Incident_Rate',
        hover_name='İstifadə Edilən Katalizatorlar',
        labels={
            'Energy_per_ton': 'Energy Consumption (kWh/ton)',
            'Efficiency': 'Process Efficiency (%)',
            'Volume': 'Processing Volume (tons)',
            'Incident_Rate': 'Incident Rate (%)'
        },
        title='Catalyst Performance Matrix',
        color_continuous_scale='RdYlGn_r'  # Red for high incident rate, green for low
    )
    
    # Improve layout
    fig2.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(title='Energy Consumption (kWh/ton)'),
        yaxis=dict(title='Process Efficiency (%)', range=[85, 100])
    )
    
    # Save visualizations
    fig2.write_html(f"{output_dir}/catalyst_matrix.html")
    with open(f"{output_dir}/catalyst_matrix.json", "w") as f:
        f.write(fig2.to_json())
    
    return fig, fig2

def create_parameter_correlation(datasets, output_dir="charts"):
    """Create parameter correlation visualization"""
    df = datasets["df"]
    
    # Select relevant numeric columns for correlation
    numeric_cols = [
        'Temperatur (°C)', 'Təzyiq (bar)', 'Prosesin Müddəti (saat)',
        'Emalın Səmərəliliyi (%)', 'Energy_per_ton', 'CO2_per_ton',
        'Təhlükəsizlik Hadisələri'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size":10}
    ))
    
    # Update layout
    fig.update_layout(
        title="Parameter Correlation Matrix",
        height=700,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(tickangle=45)
    )
    
    # Save visualizations
    fig.write_html(f"{output_dir}/parameter_correlation.html")
    with open(f"{output_dir}/parameter_correlation.json", "w") as f:
        f.write(fig.to_json())
    
    # Create a second visualization for the top correlations with efficiency
    efficiency_corr = corr_matrix['Emalın Səmərəliliyi (%)'].drop('Emalın Səmərəliliyi (%)').sort_values(ascending=False)
    
    # Create horizontal bar chart
    fig2 = go.Figure(data=go.Bar(
        x=efficiency_corr.values,
        y=efficiency_corr.index,
        orientation='h',
        marker=dict(
            color=efficiency_corr.values,
            colorscale='RdBu',
            cmin=-1, cmax=1
        ),
        text=np.round(efficiency_corr.values, 2),
        textposition='auto'
    ))
    
    # Update layout
    fig2.update_layout(
        title="Parameter Correlation with Process Efficiency",
        height=500,
        margin=dict(l=150, r=50, t=60, b=50),
        xaxis=dict(
            title="Correlation Coefficient",
            range=[-1, 1]
        ),
        yaxis=dict(title="Parameter")
    )
    
    # Save visualizations
    fig2.write_html(f"{output_dir}/efficiency_correlation.html")
    with open(f"{output_dir}/efficiency_correlation.json", "w") as f:
        f.write(fig2.to_json())
    
    return fig, fig2

def create_roi_projection(datasets, output_dir="charts"):
    """Create ROI projection visualization"""
    df = datasets["df"]
    
    # Check if we have ROI dataset or need to calculate it
    if "roi_projections" in datasets:
        roi_data = datasets["roi_projections"]
    else:
        # Calculate potential savings for each process type
        process_savings = df.groupby('Proses Tipi').agg({
            'Emal Həcmi (ton)': 'sum',
            'Əməliyyat Xərcləri (AZN)': 'sum',
            'Energy_per_ton': ['min', 'mean'],
            'Emalın Səmərəliliyi (%)': ['max', 'mean']
        }).reset_index()
        
        # Flatten columns
        process_savings.columns = ['Proses Tipi', 'Volume', 'OpCost', 'Min_Energy', 
                                 'Avg_Energy', 'Max_Efficiency', 'Avg_Efficiency']
        
        # Calculate improvement potential
        process_savings['Energy_Improvement'] = (
            (process_savings['Avg_Energy'] - process_savings['Min_Energy']) / 
            process_savings['Avg_Energy'] * 100
        )
        
        process_savings['Efficiency_Improvement'] = (
            (process_savings['Max_Efficiency'] - process_savings['Avg_Efficiency']) / 
            process_savings['Avg_Efficiency'] * 100
        )
        
        # Calculate annual savings potential (simplified model)
        process_savings['Annual_Cost'] = process_savings['OpCost'] * 12  # annualized
        
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
    total_opex = roi_data['Annual_Cost'].sum() if 'Annual_Cost' in roi_data.columns else roi_data['OpCost'].sum() * 12
    total_savings = roi_data['Total_Savings'].sum() if 'Total_Savings' in roi_data.columns else 0
    
    if 'Energy_Savings' in roi_data.columns:
        energy_savings = roi_data['Energy_Savings'].sum()
        efficiency_savings = roi_data['Efficiency_Savings'].sum()
        safety_savings = roi_data['Safety_Savings'].sum()
    else:
        # Estimate if not available
        energy_savings = total_savings * 0.4
        efficiency_savings = total_savings * 0.4
        safety_savings = total_savings * 0.2
    
    savings_data = [
        {"category": "Current Annual Costs", "value": total_opex},
        {"category": "Energy Optimization", "value": -energy_savings},
        {"category": "Efficiency Improvement", "value": -efficiency_savings},
        {"category": "Safety Enhancement", "value": -safety_savings},
        {"category": "Optimized Annual Costs", "value": total_opex - total_savings}
    ]
    
    # Convert to dataframe
    savings_df = pd.DataFrame(savings_data)
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
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
    fig.update_layout(
        title="Annual Cost Savings Potential",
        showlegend=False,
        height=600,
        margin=dict(l=50, r=50, t=60, b=50),
        yaxis=dict(title="Annual Cost (AZN)")
    )
    
    # Add annotations
    fig.add_annotation(
        x=4, y=savings_df['value'].iloc[-1] * 1.1,
        text=f"{total_savings:,.0f} AZN Potential Annual Savings ({total_savings/total_opex*100:.1f}%)",
        showarrow=False,
        font=dict(size=14, color="#2ca02c")
    )
    
    # Save visualizations
    fig.write_html(f"{output_dir}/roi_waterfall.html")
    with open(f"{output_dir}/roi_waterfall.json", "w") as f:
        f.write(fig.to_json())
    
    # Create ROI comparison chart by process type
    roi_by_process = roi_data[['Proses Tipi', 'Total_Savings', 'ROI_Months']] if 'Total_Savings' in roi_data.columns else roi_data
    
    # Create subplot with two y-axes
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add savings bars
    fig2.add_trace(
        go.Bar(
            x=roi_by_process['Proses Tipi'],
            y=roi_by_process['Total_Savings'] if 'Total_Savings' in roi_by_process.columns else roi_by_process['Energy_Savings'] + roi_by_process['Efficiency_Savings'] + roi_by_process['Safety_Savings'],
            name="Annual Savings",
            marker_color="#2ca02c",
            text=[f"{val:,.0f} AZN" for val in roi_by_process['Total_Savings']] if 'Total_Savings' in roi_by_process.columns else None,
            textposition="auto"
        ),
        secondary_y=False
    )
    
    # Add ROI months line
    fig2.add_trace(
        go.Scatter(
            x=roi_by_process['Proses Tipi'],
            y=roi_by_process['ROI_Months'] if 'ROI_Months' in roi_by_process.columns else roi_by_process['Investment'] / (roi_by_process['Total_Savings'] / 12),
            name="Payback Period",
            line=dict(color="#ff7f0e", width=3),
            mode='lines+markers+text',
            text=[f"{val:.1f} months" for val in roi_by_process['ROI_Months']] if 'ROI_Months' in roi_by_process.columns else None,
            textposition="top center"
        ),
        secondary_y=True
    )
    
    # Update layout
    fig2.update_layout(
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
    fig2.update_yaxes(title_text="Annual Savings (AZN)", secondary_y=False)
    fig2.update_yaxes(title_text="Payback Period (months)", secondary_y=True)
    
    # Save visualizations
    fig2.write_html(f"{output_dir}/roi_by_process.html")
    with open(f"{output_dir}/roi_by_process.json", "w") as f:
        f.write(fig2.to_json())
    
    return fig, fig2

def create_safety_optimization(datasets, output_dir="charts"):
    """Create safety optimization visualization"""
    df = datasets["df"]
    
    # Create temperature-pressure safety matrix
    if "safety_parameters" in datasets:
        safety_data = datasets["safety_parameters"]
    else:
        # Create from main dataset
        safety_data = df.groupby(['Temperature_Category', 'Pressure_Category']).agg({
            'Has_Incident': 'mean',
            'Təhlükəsizlik Hadisələri': 'sum',
            'Emal Həcmi (ton)': 'sum'
        }).reset_index()
        
        # If categories don't exist, create them
        if 'Temperature_Category' not in df.columns:
            temp_bins = [0, 150, 300, 450, float('inf')]
            temp_labels = ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)']
            df['Temperature_Category'] = pd.cut(df['Temperatur (°C)'], bins=temp_bins, labels=temp_labels)
            
            pressure_bins = [0, 10, 30, 50, float('inf')]
            pressure_labels = ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)']
            df['Pressure_Category'] = pd.cut(df['Təzyiq (bar)'], bins=pressure_bins, labels=pressure_labels)
            
            safety_data = df.groupby(['Temperature_Category', 'Pressure_Category']).agg({
                'Təhlükəsizlik Hadisələri': ['mean', 'sum'],
                'Emal Həcmi (ton)': 'sum'
            }).reset_index()
            
            # Flatten columns
            safety_data.columns = ['Temperature_Category', 'Pressure_Category', 
                                 'Incident_Rate', 'Total_Incidents', 'Volume']
        
        # Calculate incident rate as percentage
        if 'Has_Incident' in safety_data.columns:
            safety_data['Incident_Rate'] = safety_data['Has_Incident'] * 100
        elif 'Incident_Rate' not in safety_data.columns:
            safety_data['Incident_Rate'] = safety_data['Təhlükəsizlik Hadisələri'] / safety_data['Emal Həcmi (ton)'] * 100
    
    # Create heatmap
    # Convert categories to ordered categories for correct sorting
    temp_order = ['Low (<150°C)', 'Medium (150-300°C)', 'High (300-450°C)', 'Very High (>450°C)']
    pressure_order = ['Low (<10 bar)', 'Medium (10-30 bar)', 'High (30-50 bar)', 'Very High (>50 bar)']
    
    # Ensure we have the right column names
    temp_col = 'Temperature_Category' if 'Temperature_Category' in safety_data.columns else safety_data.columns[0]
    pressure_col = 'Pressure_Category' if 'Pressure_Category' in safety_data.columns else safety_data.columns[1]
    incident_col = 'Incident_Rate' if 'Incident_Rate' in safety_data.columns else 'Has_Incident' if 'Has_Incident' in safety_data.columns else safety_data.columns[2]
    
    # Create a pivot table
    pivot_data = safety_data.pivot_table(
        index=temp_col, 
        columns=pressure_col, 
        values=incident_col,
        aggfunc='mean'
    )
    
    # Reorder categories if possible
    try:
        pivot_data = pivot_data.reindex(temp_order, axis=0)
        pivot_data = pivot_data.reindex(pressure_order, axis=1)
    except:
        # If reindexing fails, continue with current order
        pass
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='Reds',
        text=np.round(pivot_data.values, 1),
        texttemplate="%{text}%",
        textfont={"size":10},
        colorbar=dict(title="Incident Rate (%)")
    ))
    
    # Update layout
    fig.update_layout(
        title="Safety Incident Rate by Temperature and Pressure",
        height=600,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(title="Pressure Category"),
        yaxis=dict(title="Temperature Category")
    )
    
    # Add annotation for optimal safety zone
    fig.add_annotation(
        x="Low (<10 bar)",
        y="Medium (150-300°C)",
        text="Optimal Safety Zone",
        showarrow=True,
        font=dict(size=12, color="green"),
        arrowhead=2,
        ax=0,
        ay=-40
    )
    
    # Save visualizations
    fig.write_html(f"{output_dir}/safety_heatmap.html")
    with open(f"{output_dir}/safety_heatmap.json", "w") as f:
        f.write(fig.to_json())
    
    # Create process step safety comparison
    step_safety = df.groupby('Proses Addımı').agg({
        'Təhlükəsizlik Hadisələri': ['sum', 'mean'],
        'Emal Həcmi (ton)': 'sum',
        'Emalın Səmərəliliyi (%)': 'mean',
        'Energy_per_ton': 'mean'
    }).reset_index()
    
    # Flatten columns
    if isinstance(step_safety.columns, pd.MultiIndex):
        step_safety.columns = [
            'Proses_Addımı', 'Total_Incidents', 'Avg_Incidents',
            'Volume', 'Efficiency', 'Energy_per_ton'
        ]
    
    # Calculate incident rate
    if 'Has_Incident' in df.columns:
        incident_by_step = df.groupby('Proses Addımı')['Has_Incident'].mean() * 100
        step_safety = step_safety.merge(incident_by_step.reset_index(), on='Proses Addımı')
    else:
        step_safety['Incident_Rate'] = step_safety['Avg_Incidents'] * 100
    
    # Sort by incident rate
    step_safety = step_safety.sort_values('Incident_Rate' if 'Incident_Rate' in step_safety.columns else 'Has_Incident', ascending=False)
    
    # Create horizontal bar chart
    fig2 = go.Figure()
    
    # Add incident rate bars
    fig2.add_trace(go.Bar(
        y=step_safety['Proses_Addımı'],
        x=step_safety['Incident_Rate'] if 'Incident_Rate' in step_safety.columns else step_safety['Has_Incident'],
        name="Incident Rate (%)",
        orientation='h',
        marker=dict(
            color=step_safety['Incident_Rate'] if 'Incident_Rate' in step_safety.columns else step_safety['Has_Incident'],
            colorscale='Reds',
            colorbar=dict(title="Incident Rate (%)")
        ),
        text=np.round(step_safety['Incident_Rate'] if 'Incident_Rate' in step_safety.columns else step_safety['Has_Incident'], 1),
        textposition="auto",
        texttemplate="%{text}%"
    ))
    
    # Update layout
    fig2.update_layout(
        title="Safety Incident Rate by Process Step",
        height=500,
        margin=dict(l=150, r=50, t=60, b=50),
        xaxis=dict(
            title="Incident Rate (%)",
            range=[0, 100]
        ),
        yaxis=dict(title="Process Step")
    )
    
    # Save visualizations
    fig2.write_html(f"{output_dir}/process_step_safety.html")
    with open(f"{output_dir}/process_step_safety.json", "w") as f:
        f.write(fig2.to_json())
    
    return fig, fig2

def create_kpi_dashboard(datasets, output_dir="charts"):
    """Create KPI summary dashboard"""
    df = datasets["df"]
    
    # Calculate KPIs
    overall_efficiency = df['Emalın Səmərəliliyi (%)'].mean()
    total_incidents = df['Təhlükəsizlik Hadisələri'].sum()
    total_volume = df['Emal Həcmi (ton)'].sum()
    avg_energy = df['Energy_per_ton'].mean()
    incident_rate = total_incidents / total_volume * 1000  # Incidents per 1000 tons
    
    # Get best and worst processes
    best_process = df.loc[df['Emalın Səmərəliliyi (%)'].idxmax()]
    worst_process = df.loc[df['Emalın Səmərəliliyi (%)'].idxmin()]
    
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
            gauge={
                "axis": {"range": [85, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [85, 90], "color": "red"},
                    {"range": [90, 95], "color": "yellow"},
                    {"range": [95, 100], "color": "green"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 95
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
            gauge={
                "axis": {"range": [0, 10]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 2], "color": "green"},
                    {"range": [2, 5], "color": "yellow"},
                    {"range": [5, 10], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 3
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
            gauge={
                "axis": {"range": [1, 3]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [1, 1.5], "color": "green"},
                    {"range": [1.5, 2], "color": "yellow"},
                    {"range": [2, 3], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 1.6
                }
            }
        ),
        row=2, col=1
    )
    
    # Add improvement potential gauge
    improvement_potential = 15  # Example value - should calculate from data
    
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=overall_efficiency,
            delta={"reference": 100, "increasing": {"color": "green"}},
            title={"text": "Improvement Potential"},
            gauge={
                "axis": {"range": [85, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [85, 90], "color": "red"},
                    {"range": [90, 95], "color": "yellow"},
                    {"range": [95, 100], "color": "green"}
                ],
                "threshold": {
                    "line": {"color": "green", "width": 4},
                    "thickness": 0.75,
                    "value": 98
                }
            }
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="SOCAR Process Performance Dashboard",
        height=800,
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    # Save visualizations
    fig.write_html(f"{output_dir}/kpi_dashboard.html")
    with open(f"{output_dir}/kpi_dashboard.json", "w") as f:
        f.write(fig.to_json())
    
    return fig

def generate_all_visualizations(data_dir="data", output_dir="charts"):
    """Generate all visualizations for the dashboard"""
    print("Starting visualization generation...")
    
    # 1. Load data
    datasets = load_processed_data(data_dir)
    
    # 2. Create all visualizations
    visualizations = {
        "process_efficiency": create_process_efficiency_comparison(datasets, output_dir),
        "energy_safety": create_energy_safety_relationship(datasets, output_dir),
        "process_hierarchy": create_process_hierarchy(datasets, output_dir),
        "catalyst_analysis": create_catalyst_analysis(datasets, output_dir),
        "parameter_correlation": create_parameter_correlation(datasets, output_dir),
        "roi_projection": create_roi_projection(datasets, output_dir),
        "safety_optimization": create_safety_optimization(datasets, output_dir),
        "kpi_dashboard": create_kpi_dashboard(datasets, output_dir)
    }
    
    print(f"Generated {len(visualizations)} visualization sets")
    print(f"All visualizations saved to {output_dir}")
    
    return visualizations

if __name__ == "__main__":
    # Generate all visualizations
    visualizations = generate_all_visualizations("socar-dashboard/data", "socar-dashboard/charts")