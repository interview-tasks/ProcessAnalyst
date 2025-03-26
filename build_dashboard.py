#!/usr/bin/env python
"""
SOCAR Process Analysis Dashboard Generator - SIMPLIFIED VERSION

This script:
1. Creates necessary directory structure
2. Generates sample data files (if needed)
3. Generates static chart files
4. Updates dashboard.js with proper paths
5. Places index.html in the root directory

Usage:
    python build_dashboard.py
"""

import os
import sys
import shutil
import logging
import json
from pathlib import Path

# Configure simple logging to console only
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dashboard_generator")

# Chart definitions - static sample charts
CHART_DEFINITIONS = {
    "process_efficiency.json": {
        "data": [
            {
                "x": ["Type A", "Type B", "Type C", "Type D"],
                "y": [92.5, 94.8, 91.2, 93.7],
                "type": "bar",
                "name": "Efficiency (%)",
                "text": ["92.5%", "94.8%", "91.2%", "93.7%"],
                "textposition": "auto",
                "marker": {"color": "#1f77b4"}
            },
            {
                "x": ["Type A", "Type B", "Type C", "Type D"],
                "y": [2.1, 1.4, 2.8, 1.9],
                "type": "scatter",
                "mode": "lines+markers",
                "name": "Safety Incidents",
                "yaxis": "y2",
                "marker": {"size": 10, "color": "#d62728"},
                "line": {"width": 3, "color": "#d62728"}
            }
        ],
        "layout": {
            "title": "Process Efficiency vs Safety by Type",
            "xaxis": {"title": "Process Type"},
            "yaxis": {"title": "Efficiency (%)", "range": [85, 100]},
            "yaxis2": {
                "title": "Safety Incidents",
                "overlaying": "y",
                "side": "right",
                "range": [0, 5]
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            },
            "hovermode": "x unified"
        }
    },
    "energy_safety.json": {
        "data": [
            {
                "x": [1.5, 1.8, 2.2, 1.6, 1.9, 2.3, 1.7, 2.0, 2.4, 1.5, 1.8, 2.1],
                "y": [1, 1, 2, 0, 1, 3, 0, 2, 4, 1, 2, 3],
                "text": ["Type A", "Type A", "Type A", "Type B", "Type B", "Type B", "Type C", "Type C", "Type C", "Type D", "Type D", "Type D"],
                "mode": "markers",
                "type": "scatter",
                "marker": {
                    "size": [30, 40, 35, 45, 35, 30, 40, 35, 30, 35, 40, 45],
                    "color": ["#1f77b4", "#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#ff7f0e", "#2ca02c", "#2ca02c", "#2ca02c", "#d62728", "#d62728", "#d62728"],
                    "opacity": 0.7
                },
                "name": "Process Data"
            }
        ],
        "layout": {
            "title": "Energy Consumption vs. Safety Incidents",
            "xaxis": {"title": "Energy per Ton (kWh/ton)"},
            "yaxis": {"title": "Safety Incidents"}
        }
    },
    "process_hierarchy.json": {
        "data": [
            {
                "type": "sunburst",
                "labels": ["Process A", "Step A1", "Step A2", "Step A3", "Process B", "Step B1", "Step B2", "Process C", "Step C1", "Step C2", "Step C3"],
                "parents": ["", "Process A", "Process A", "Process A", "", "Process B", "Process B", "", "Process C", "Process C", "Process C"],
                "values": [100, 30, 40, 30, 80, 45, 35, 120, 40, 40, 40],
                "branchvalues": "total",
                "marker": {
                    "colors": ["#1f77b4", "#aec7e8", "#aec7e8", "#aec7e8", "#ff7f0e", "#ffbb78", "#ffbb78", "#2ca02c", "#98df8a", "#98df8a", "#98df8a"]
                },
                "textinfo": "label+percent entry"
            }
        ],
        "layout": {
            "title": "Process Hierarchy, Efficiency and Volume",
            "margin": {"l": 0, "r": 0, "b": 0, "t": 40}
        }
    },
    "efficiency_correlation.json": {
        "data": [
            {
                "x": [0.82, 0.65, -0.48, 0.35, -0.22, -0.12],
                "y": ["Temperature (°C)", "Pressure (bar)", "Duration (hours)", "Catalyst Quality", "CO₂ Emissions", "Worker Count"],
                "type": "bar",
                "orientation": "h",
                "marker": {
                    "color": [0.82, 0.65, -0.48, 0.35, -0.22, -0.12],
                    "colorscale": "RdBu",
                    "cmin": -1,
                    "cmax": 1
                },
                "text": ["0.82", "0.65", "-0.48", "0.35", "-0.22", "-0.12"],
                "textposition": "auto"
            }
        ],
        "layout": {
            "title": "Parameter Correlation with Process Efficiency",
            "xaxis": {"title": "Correlation Coefficient", "range": [-1, 1]},
            "yaxis": {"title": "Parameter"}
        }
    },
    "safety_heatmap.json": {
        "data": [
            {
                "z": [
                    [1.2, 2.5, 4.3, 5.9],
                    [1.8, 2.2, 3.8, 4.5],
                    [2.3, 3.1, 4.2, 6.2],
                    [3.5, 4.2, 5.6, 7.8]
                ],
                "x": ["Low (<10 bar)", "Medium (10-30 bar)", "High (30-50 bar)", "Very High (>50 bar)"],
                "y": ["Low (<150°C)", "Medium (150-300°C)", "High (300-450°C)", "Very High (>450°C)"],
                "type": "heatmap",
                "colorscale": "Reds",
                "text": [
                    ["1.2%", "2.5%", "4.3%", "5.9%"],
                    ["1.8%", "2.2%", "3.8%", "4.5%"],
                    ["2.3%", "3.1%", "4.2%", "6.2%"],
                    ["3.5%", "4.2%", "5.6%", "7.8%"]
                ],
                "texttemplate": "%{text}",
                "showscale": True
            }
        ],
        "layout": {
            "title": "Safety Incident Rate by Temperature and Pressure",
            "xaxis": {"title": "Pressure Category"},
            "yaxis": {"title": "Temperature Category"}
        }
    },
    "process_step_safety.json": {
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": [5.2, 4.8, 3.5, 2.8, 2.2, 1.5, 0.8],
                "y": ["Initial Heating", "Catalyst Activation", "High-Pressure Separation", "Material Transfer", "Cooling", "Quality Control", "Packaging"],
                "marker": {
                    "color": ["#d62728", "#d62728", "#d62728", "#ff7f0e", "#ff7f0e", "#2ca02c", "#2ca02c"]
                },
                "text": ["5.2%", "4.8%", "3.5%", "2.8%", "2.2%", "1.5%", "0.8%"],
                "textposition": "auto"
            }
        ],
        "layout": {
            "title": "Safety Incident Rate by Process Step",
            "xaxis": {"title": "Incident Rate (%)", "range": [0, 6]},
            "yaxis": {"title": "Process Step"}
        }
    },
    "catalyst_matrix.json": {
        "data": [
            {
                "x": [2.1, 1.8, 1.9, 1.5, 1.7],
                "y": [91.5, 93.2, 92.4, 94.8, 92.8],
                "mode": "markers",
                "type": "scatter",
                "text": ["Catalyst A", "Catalyst B", "Catalyst C", "HydroCat-450", "Catalyst E"],
                "marker": {
                    "size": [120, 150, 100, 180, 130],
                    "color": [3.2, 2.5, 2.8, 1.2, 2.0],
                    "colorscale": "RdYlGn_r",
                    "colorbar": {"title": "Incident Rate (%)"},
                    "showscale": True
                }
            }
        ],
        "layout": {
            "title": "Catalyst Performance Matrix",
            "xaxis": {"title": "Energy Consumption (kWh/ton)", "range": [1.4, 2.2]},
            "yaxis": {"title": "Process Efficiency (%)", "range": [91, 95]}
        }
    },
    "catalyst_parallel.json": {
        "data": [
            {
                "type": "parcoords",
                "line": {
                    "color": [94.8, 93.2, 92.8, 92.4, 91.5],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": "Efficiency (%)"}
                },
                "dimensions": [
                    {"range": [91, 95], "label": "Efficiency (%)", "values": [94.8, 93.2, 92.8, 92.4, 91.5]},
                    {"range": [1.4, 2.2], "label": "Energy (kWh/ton)", "values": [1.5, 1.8, 1.7, 1.9, 2.1]},
                    {"range": [0.8, 3.5], "label": "CO₂ (kg/ton)", "values": [1.2, 1.8, 1.5, 2.0, 2.5]},
                    {"range": [0, 4], "label": "Incident Rate (%)", "values": [1.2, 2.5, 2.0, 2.8, 3.2]}
                ]
            }
        ],
        "layout": {
            "title": "Catalyst Parameter Comparison",
            "margin": {"l": 80, "r": 50, "t": 60, "b": 50}
        }
    },
    "roi_waterfall.json": {
        "data": [
            {
                "type": "waterfall",
                "orientation": "v",
                "measure": ["absolute", "relative", "relative", "relative", "total"],
                "x": ["Current Annual Costs", "Energy Optimization", "Efficiency Improvement", "Safety Enhancement", "Optimized Annual Costs"],
                "y": [5000000, -800000, -950000, -400000, 2850000],
                "text": ["₼5,000,000", "-₼800,000", "-₼950,000", "-₼400,000", "₼2,850,000"],
                "textposition": "outside",
                "connector": {"line": {"color": "rgb(63, 63, 63)"}},
                "decreasing": {"marker": {"color": "#2ca02c"}},
                "increasing": {"marker": {"color": "#d62728"}},
                "totals": {"marker": {"color": "#1f77b4"}}
            }
        ],
        "layout": {
            "title": "Annual Cost Savings Potential",
            "showlegend": False,
            "yaxis": {"title": "Annual Cost (AZN)"}
        }
    },
    "roi_by_process.json": {
        "data": [
            {
                "x": ["Type A", "Type B", "Type C", "Type D"],
                "y": [580000, 850000, 420000, 300000],
                "type": "bar",
                "name": "Annual Savings",
                "marker": {"color": "#2ca02c"},
                "text": ["₼580,000", "₼850,000", "₼420,000", "₼300,000"],
                "textposition": "auto"
            },
            {
                "x": ["Type A", "Type B", "Type C", "Type D"],
                "y": [18, 12, 22, 28],
                "type": "scatter",
                "mode": "lines+markers+text",
                "name": "Payback Period",
                "yaxis": "y2",
                "line": {"color": "#ff7f0e", "width": 3},
                "marker": {"size": 10},
                "text": ["18 months", "12 months", "22 months", "28 months"],
                "textposition": "top center"
            }
        ],
        "layout": {
            "title": "ROI Analysis by Process Type",
            "xaxis": {"title": "Process Type"},
            "yaxis": {"title": "Annual Savings (AZN)"},
            "yaxis2": {
                "title": "Payback Period (months)",
                "overlaying": "y",
                "side": "right",
                "range": [0, 30]
            },
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            }
        }
    }
}

# Sample data for processed_data.csv
SAMPLE_PROCESSED_DATA = """Proses ID,Proses Tipi,Proses Addımı,Emal Həcmi (ton),Temperatur (°C),Təzyiq (bar),Prosesin Müddəti (saat),İstifadə Edilən Katalizatorlar,Emalın Səmərəliliyi (%),Enerji İstifadəsi (kWh),Ətraf Mühitə Təsir (g CO2 ekvivalent),Təhlükəsizlik Hadisələri,Energy_per_ton
1,Type A,Step 1,100,250,20,12,Catalyst B,92.5,150000,300000,1,1.5
2,Type A,Step 2,150,280,25,10,Catalyst B,91.0,270000,520000,2,1.8
3,Type B,Step 1,120,320,35,8,Catalyst A,94.8,264000,450000,0,2.2
4,Type B,Step 3,90,350,40,14,Catalyst A,89.7,207000,380000,3,2.3
5,Type C,Step 2,200,280,25,10,Catalyst C,93.2,320000,600000,1,1.6
6,Type C,Step 1,180,270,22,9,Catalyst C,92.8,342000,650000,2,1.9
7,Type D,Step 3,150,330,38,13,HydroCat-450,90.5,345000,700000,3,2.3
8,Type D,Step 1,160,310,32,11,HydroCat-450,93.7,272000,520000,1,1.7"""

# Sample data for process_types.csv
SAMPLE_PROCESS_TYPES = """Proses Tipi,Emalın Səmərəliliyi (%),Emal Həcmi (ton),Energy_per_ton
Type A,91.8,250,1.7
Type B,92.3,210,2.2
Type C,93.0,380,1.8
Type D,92.1,310,2.0"""

# Improved dashboard.js with better path handling for GitHub Pages
DASHBOARD_JS = """/**
 * SOCAR Process Analysis Dashboard
 * Main dashboard functionality
 * 
 * FIXED VERSION: Updated for GitHub Pages
 */

// Dashboard configuration
const config = {
    // Updated paths for GitHub Pages deployment
    dataPath: './dashboard/data/',
    chartsPath: './dashboard/charts/',
    processDependencies: {
        overview: ['process_efficiency.json', 'energy_safety.json', 'roi_waterfall.json'],
        efficiency: ['process_hierarchy.json', 'efficiency_correlation.json'],
        energy: ['energy_safety.json'],
        safety: ['safety_heatmap.json', 'process_step_safety.json'],
        catalyst: ['catalyst_matrix.json', 'catalyst_parallel.json'],
        roi: ['roi_waterfall.json', 'roi_by_process.json']
    },
    chartContainers: {
        // Overview section
        'process-efficiency-chart': 'process_efficiency.json',
        'energy-safety-chart': 'energy_safety.json',
        'roi-waterfall-chart': 'roi_waterfall.json',
        
        // Process Efficiency section
        'process-hierarchy-chart': 'process_hierarchy.json',
        'efficiency-correlation-chart': 'efficiency_correlation.json',
        
        // Energy section
        'energy-safety-relationship-chart': 'energy_safety.json',
        
        // Safety section
        'safety-heatmap-chart': 'safety_heatmap.json',
        'process-step-safety-chart': 'process_step_safety.json',
        
        // Catalyst section
        'catalyst-matrix-chart': 'catalyst_matrix.json',
        'catalyst-parallel-chart': 'catalyst_parallel.json',
        
        // ROI section
        'roi-by-process-chart': 'roi_by_process.json'
    }
};

// Global state
let dashboardData = {
    chartData: {},
    processTypes: [],
    processSteps: [],
    catalysts: [],
    currentFilters: {
        processType: 'all',
        processStep: 'all',
        catalyst: 'all'
    },
    currentSection: 'overview'
};

/**
 * Initialize the dashboard
 */
async function initDashboard() {
    showLoadingIndicator();
    
    try {
        // Load main processed data
        await loadProcessedData();
        
        // Set up navigation and tab switching
        setupNavigation();
        
        // Set up filter controls
        setupFilters();
        
        // Load initial chart data
        await loadChartData('overview');
        
        // Render KPI data
        renderKPIs();
        
        // Set up ROI calculator
        setupROICalculator();
        
        // Set up export functionality
        setupExportButtons();
        
        // Show initial section
        showSection('overview');
    } catch (error) {
        console.error('Dashboard initialization error:', error);
        showErrorMessage('Failed to initialize dashboard. See console for details.');
    } finally {
        hideLoadingIndicator();
    }
}

/**
 * Load main processed data
 */
async function loadProcessedData() {
    try {
        // Enhanced error handling and logging for data loading
        console.log(`Attempting to load processed data from ${config.dataPath}processed_data.csv`);
        
        // Load main dataset
        const response = await fetch(`${config.dataPath}processed_data.csv`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
        }
        
        const csvText = await response.text();
        console.log(`Successfully loaded CSV with length: ${csvText.length} characters`);
        console.log(`CSV preview: ${csvText.substring(0, 200)}...`);
        
        const processedData = parseCSV(csvText);
        
        dashboardData.processedData = processedData;
        
        // Extract unique values for filters
        dashboardData.processTypes = [...new Set(processedData.map(row => row['Proses Tipi']))];
        dashboardData.processSteps = [...new Set(processedData.map(row => row['Proses Addımı']))];
        dashboardData.catalysts = [...new Set(processedData.map(row => row['İstifadə Edilən Katalizatorlar']))];
        
        // Load aggregated datasets
        try {
            const typesResponse = await fetch(`${config.dataPath}process_types.csv`);
            if (typesResponse.ok) {
                const typesText = await typesResponse.text();
                dashboardData.processTypesData = parseCSV(typesText);
                console.log('Successfully loaded process_types.csv');
            } else {
                console.warn(`Failed to load process_types.csv: ${typesResponse.status}`);
            }
        } catch (e) {
            console.warn('Error loading process_types.csv:', e);
        }
        
        console.log('Data loaded successfully:', dashboardData);
        
    } catch (error) {
        console.error('Error loading processed data:', error);
        // Create fallback data for demo purposes if data loading fails
        console.log('Creating fallback demo data');
        createFallbackData();
    }
}

/**
 * Create fallback data if real data cannot be loaded
 */
function createFallbackData() {
    // Simple fallback data for demonstration
    dashboardData.processedData = [
        {'Proses Tipi': 'Type A', 'Proses Addımı': 'Step 1', 'Emal Həcmi (ton)': 100, 'Emalın Səmərəliliyi (%)': 93.5, 'Təhlükəsizlik Hadisələri': 2, 'İstifadə Edilən Katalizatorlar': 'Catalyst B', 'Energy_per_ton': 1.5},
        {'Proses Tipi': 'Type A', 'Proses Addımı': 'Step 2', 'Emal Həcmi (ton)': 150, 'Emalın Səmərəliliyi (%)': 91.2, 'Təhlükəsizlik Hadisələri': 1, 'İstifadə Edilən Katalizatorlar': 'Catalyst B', 'Energy_per_ton': 1.8},
        {'Proses Tipi': 'Type B', 'Proses Addımı': 'Step 1', 'Emal Həcmi (ton)': 120, 'Emalın Səmərəliliyi (%)': 94.8, 'Təhlükəsizlik Hadisələri': 0, 'İstifadə Edilən Katalizatorlar': 'Catalyst A', 'Energy_per_ton': 2.2},
        {'Proses Tipi': 'Type B', 'Proses Addımı': 'Step 3', 'Emal Həcmi (ton)': 90, 'Emalın Səmərəliliyi (%)': 89.7, 'Təhlükəsizlik Hadisələri': 3, 'İstifadə Edilən Katalizatorlar': 'Catalyst A', 'Energy_per_ton': 2.3}
    ];
    
    // Extract unique values
    dashboardData.processTypes = ['Type A', 'Type B'];
    dashboardData.processSteps = ['Step 1', 'Step 2', 'Step 3'];
    dashboardData.catalysts = ['Catalyst A', 'Catalyst B', 'Catalyst C'];
    
    // Create simple aggregated data
    dashboardData.processTypesData = [
        {'Proses Tipi': 'Type A', 'Emalın Səmərəliliyi (%)': 92.3, 'Emal Həcmi (ton)': 250, 'Energy_per_ton': 1.8},
        {'Proses Tipi': 'Type B', 'Emalın Səmərəliliyi (%)': 92.5, 'Emal Həcmi (ton)': 210, 'Energy_per_ton': 1.7}
    ];
}

/**
 * Parse CSV data with improved error handling
 */
function parseCSV(csvText) {
    try {
        const lines = csvText.split('\\n');
        if (lines.length < 2) {
            console.error('CSV has too few lines:', lines.length);
            return [];
        }
        
        const headers = lines[0].split(',').map(header => header.trim());
        console.log('CSV Headers:', headers);
        
        const data = [];
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            
            // Handle quoted values that might contain commas
            let values = [];
            let inQuote = false;
            let currentValue = '';
            
            for (let j = 0; j < line.length; j++) {
                const char = line[j];
                
                if (char === '"' && (j === 0 || line[j-1] !== '\\\\')) {
                    inQuote = !inQuote;
                } else if (char === ',' && !inQuote) {
                    values.push(currentValue);
                    currentValue = '';
                } else {
                    currentValue += char;
                }
            }
            
            // Don't forget to add the last value
            values.push(currentValue);
            
            // If simple split would work better (no quotes)
            if (values.length !== headers.length) {
                values = line.split(',');
            }
            
            const row = {};
            for (let j = 0; j < Math.min(headers.length, values.length); j++) {
                const value = values[j]?.trim() || '';
                // Convert to number if possible
                row[headers[j]] = isNaN(value) ? value : parseFloat(value);
            }
            
            data.push(row);
        }
        
        console.log(`Successfully parsed ${data.length} rows from CSV`);
        return data;
    } catch (error) {
        console.error('Error parsing CSV:', error);
        return [];
    }
}

/**
 * Load chart data for a specific section
 */
async function loadChartData(section) {
    const dependencies = config.processDependencies[section] || [];
    
    for (const chartFile of dependencies) {
        if (!dashboardData.chartData[chartFile]) {
            try {
                console.log(`Loading chart data: ${config.chartsPath}${chartFile}`);
                const response = await fetch(`${config.chartsPath}${chartFile}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status} for ${chartFile}`);
                }
                
                const chartData = await response.json();
                console.log(`Successfully loaded chart data for ${chartFile}`);
                dashboardData.chartData[chartFile] = chartData;
            } catch (error) {
                console.error(`Error loading chart data for ${chartFile}:`, error);
                
                // Create fallback chart data
                console.log(`Creating fallback chart for ${chartFile}`);
                dashboardData.chartData[chartFile] = createFallbackChart(chartFile);
            }
        }
    }
}

/**
 * Create a fallback chart if the real data cannot be loaded
 */
function createFallbackChart(chartFile) {
    // Generate a simple fallback chart based on the chart type
    const chartType = chartFile.split('_')[0];
    
    // Basic trace structure
    let data = [];
    let layout = {
        title: `${chartType.charAt(0).toUpperCase() + chartType.slice(1)} Chart (Demo Data)`,
        showlegend: true
    };
    
    if (chartFile.includes('efficiency')) {
        // Efficiency-related charts
        data = [{
            x: dashboardData.processTypes,
            y: [92.5, 93.8, 90.2, 95.1],
            type: 'bar',
            name: 'Efficiency (%)'
        }];
    } else if (chartFile.includes('energy')) {
        // Energy-related charts
        data = [{
            x: [1.5, 1.8, 2.1, 1.7],
            y: [93, 92, 90, 94],
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 12
            },
            name: 'Process Data'
        }];
    } else if (chartFile.includes('safety')) {
        // Safety-related charts
        data = [{
            z: [[1, 2, 3], [3, 2, 1], [2, 1, 3]],
            x: ['Low', 'Medium', 'High'],
            y: ['Cold', 'Warm', 'Hot'],
            type: 'heatmap',
            colorscale: 'Reds'
        }];
    } else if (chartFile.includes('roi')) {
        // ROI-related charts
        data = [{
            x: ['Current', 'Savings', 'Optimized'],
            y: [100, -25, 75],
            type: 'waterfall',
            name: 'ROI Analysis'
        }];
    } else {
        // Generic fallback
        data = [{
            x: [1, 2, 3, 4],
            y: [10, 15, 13, 17],
            type: 'scatter',
            name: 'Demo Data'
        }];
    }
    
    return { data, layout };
}

/**
 * Render all charts for the current section
 */
function renderCharts(section) {
    const containers = Object.keys(config.chartContainers);
    
    for (const containerId of containers) {
        const chartFile = config.chartContainers[containerId];
        const containerElement = document.getElementById(containerId);
        
        if (containerElement && dashboardData.chartData[chartFile]) {
            try {
                console.log(`Rendering chart: ${containerId}`);
                const chartData = dashboardData.chartData[chartFile];
                
                // Add error handling for chart rendering
                if (!chartData || !chartData.data || !Array.isArray(chartData.data)) {
                    throw new Error(`Invalid chart data for ${chartFile}`);
                }
                
                Plotly.newPlot(containerId, chartData.data, chartData.layout, {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d']
                });
                
                console.log(`Successfully rendered chart: ${containerId}`);
            } catch (error) {
                console.error(`Error rendering chart ${containerId}:`, error);
                containerElement.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="bi bi-exclamation-triangle-fill me-2"></i>
                        Failed to render chart: ${error.message}
                    </div>
                `;
            }
        } else if (containerElement) {
            containerElement.innerHTML = `
                <div class="alert alert-warning">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    Chart data not available
                </div>
            `;
        }
    }
}

/**
 * Render KPI cards with summary data
 */
function renderKPIs() {
    // Calculate or extract KPI values
    let efficiencyValue, energyValue, safetyValue, savingsValue;
    
    if (dashboardData.processTypesData) {
        // Average efficiency across all process types
        efficiencyValue = dashboardData.processTypesData.reduce((sum, row) => 
            sum + row['Emalın Səmərəliliyi (%)'], 0) / dashboardData.processTypesData.length;
        
        // Average energy per ton
        energyValue = dashboardData.processTypesData.reduce((sum, row) => 
            sum + (row['Energy_per_ton'] || 0), 0) / dashboardData.processTypesData.length;
    } else {
        efficiencyValue = 92.5;
        energyValue = 1.8;
    }
    
    // Safety incidents per 1000 tons (from processed data)
    if (dashboardData.processedData) {
        const totalIncidents = dashboardData.processedData.reduce((sum, row) => 
            sum + (row['Təhlükəsizlik Hadisələri'] || 0), 0);
        const totalVolume = dashboardData.processedData.reduce((sum, row) => 
            sum + (row['Emal Həcmi (ton)'] || 0), 0);
        
        safetyValue = totalVolume > 0 ? (totalIncidents / totalVolume) * 1000 : 3.2;
    } else {
        safetyValue = 3.2;
    }
    
    // Potential savings from ROI projections
    savingsValue = 2.15;  // Default value
    
    // Update KPI elements
    const efficiencyElement = document.getElementById('kpi-efficiency');
    if (efficiencyElement) {
        efficiencyElement.textContent = `${efficiencyValue.toFixed(1)}%`;
    }
    
    const energyElement = document.getElementById('kpi-energy'); 
    if (energyElement) {
        energyElement.textContent = `${energyValue.toFixed(1)} kWh/ton`;
    }
    
    const safetyElement = document.getElementById('kpi-safety');
    if (safetyElement) {
        safetyElement.textContent = `${safetyValue.toFixed(1)} per 1000t`;
    }
    
    const savingsElement = document.getElementById('kpi-savings');
    if (savingsElement) {
        savingsElement.textContent = `₼${savingsValue.toFixed(2)}M`;
    }
}

/**
 * Set up navigation and tab switching
 */
function setupNavigation() {
    // Add click event listeners to all navigation tabs
    document.querySelectorAll('[data-section]').forEach(tab => {
        tab.addEventListener('click', async (event) => {
            event.preventDefault();
            
            const section = tab.getAttribute('data-section');
            showLoadingIndicator();
            
            try {
                // Load chart data if not already loaded
                await loadChartData(section);
                
                // Update active tab
                document.querySelectorAll('.nav-link').forEach(link => {
                    link.classList.remove('active');
                });
                tab.classList.add('active');
                
                // Show the selected section
                showSection(section);
                
                // Update section title
                const sectionTitleElement = document.getElementById('section-title');
                if (sectionTitleElement) {
                    sectionTitleElement.textContent = tab.textContent.trim();
                }
                    
                // Update current section in state
                dashboardData.currentSection = section;
            } catch (error) {
                console.error(`Error switching to section ${section}:`, error);
                showErrorMessage(`Failed to load ${section} section`);
            } finally {
                hideLoadingIndicator();
            }
        });
    });
}

/**
 * Show a specific section and hide others
 */
function showSection(sectionId) {
    document.querySelectorAll('.dashboard-section').forEach(section => {
        section.classList.remove('active');
    });
    
    const section = document.getElementById(`${sectionId}-section`);
    if (section) {
        section.classList.add('active');
        renderCharts(sectionId);
    }
}

/**
 * Set up filter controls
 */
function setupFilters() {
    // Populate filter options
    populateFilterOptions('process-type-filter', dashboardData.processTypes);
    populateFilterOptions('process-step-filter', dashboardData.processSteps);
    populateFilterOptions('catalyst-filter', dashboardData.catalysts);
    
    // Add event listeners to filters
    const typeFilter = document.getElementById('process-type-filter');
    if (typeFilter) {
        typeFilter.addEventListener('change', handleFilterChange);
    }
    
    const stepFilter = document.getElementById('process-step-filter');
    if (stepFilter) {
        stepFilter.addEventListener('change', handleFilterChange);
    }
    
    const catalystFilter = document.getElementById('catalyst-filter');
    if (catalystFilter) {
        catalystFilter.addEventListener('change', handleFilterChange);
    }
    
    // Also populate the process type input for ROI calculator
    populateFilterOptions('process-type-input', dashboardData.processTypes);
}

/**
 * Populate filter dropdown options
 */
function populateFilterOptions(elementId, options) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    // Clear existing options except the first one
    while (element.options.length > 1) {
        element.remove(1);
    }
    
    // Add options
    if (options && options.length) {
        options.forEach(option => {
            if (option) {
                const optionElement = document.createElement('option');
                optionElement.value = option;
                optionElement.textContent = option;
                element.appendChild(optionElement);
            }
        });
    }
}

/**
 * Handle filter changes
 */
function handleFilterChange(event) {
    const filterId = event.target.id;
    const filterValue = event.target.value;
    
    // Update current filters state
    if (filterId === 'process-type-filter') {
        dashboardData.currentFilters.processType = filterValue;
    } else if (filterId === 'process-step-filter') {
        dashboardData.currentFilters.processStep = filterValue;
    } else if (filterId === 'catalyst-filter') {
        dashboardData.currentFilters.catalyst = filterValue;
    }
    
    console.log('Filters changed:', dashboardData.currentFilters);
    
    // For this demo, we'll just log the filter changes
    showInfoMessage(`Filters applied: ${JSON.stringify(dashboardData.currentFilters)}`);
}

/**
 * Set up ROI calculator
 */
function setupROICalculator() {
    const form = document.getElementById('roi-calculator-form');
    
    if (form) {
        form.addEventListener('submit', (event) => {
            event.preventDefault();
            
            const processType = document.getElementById('process-type-input').value;
            const implementationCost = parseFloat(document.getElementById('implementation-cost').value);
            const efficiencyImprovement = parseFloat(document.getElementById('efficiency-improvement').value);
            
            calculateROI(processType, implementationCost, efficiencyImprovement);
        });
    }
}

/**
 * Calculate ROI based on inputs
 */
function calculateROI(processType, implementationCost, efficiencyImprovement) {
    const resultsContainer = document.getElementById('roi-results');
    if (!resultsContainer) return;
    
    try {
        // Find process data
        const processData = dashboardData.processTypesData?.find(p => p['Proses Tipi'] === processType);
        
        if (!processData) {
            throw new Error('Process type data not found');
        }
        
        // Simple ROI calculation (in a real application, this would be more complex)
        const annualVolume = processData['Emal Həcmi (ton)'] * 12; // Assuming monthly data
        const currentEfficiency = processData['Emalın Səmərəliliyi (%)'];
        const costPerTon = 500; // Placeholder value for demonstration
        
        // Potential savings calculation
        const currentAnnualCost = annualVolume * costPerTon;
        const newEfficiency = currentEfficiency + efficiencyImprovement;
        const savedVolume = annualVolume * (efficiencyImprovement / 100);
        const annualSavings = savedVolume * costPerTon;
        
        // ROI metrics
        const roiMonths = implementationCost / (annualSavings / 12);
        const roi5Years = (annualSavings * 5 - implementationCost) / implementationCost * 100;
        
        // Display results
        resultsContainer.innerHTML = `
            <div class="mb-3">
                <h5>Annual Savings</h5>
                <h3 class="text-success">₼${annualSavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</h3>
            </div>
            <div class="mb-3">
                <h5>ROI Period</h5>
                <h3>${roiMonths.toFixed(1)} months</h3>
            </div>
            <div class="mb-3">
                <h5>5-Year ROI</h5>
                <h3>${roi5Years.toFixed(1)}%</h3>
            </div>
            <div class="alert alert-success">
                <strong>Recommendation:</strong> ${
                    roiMonths <= 12 
                    ? 'Strongly recommended investment with fast payback period.' 
                    : roiMonths <= 24 
                    ? 'Recommended investment with moderate payback period.' 
                    : 'Consider alternatives with faster payback period.'
                }
            </div>
        `;
    } catch (error) {
        console.error('ROI calculation error:', error);
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                Error calculating ROI: ${error.message}
            </div>
        `;
    }
}

/**
 * Set up export buttons
 */
function setupExportButtons() {
    const pdfExport = document.getElementById('export-pdf');
    if (pdfExport) {
        pdfExport.addEventListener('click', () => {
            showInfoMessage('PDF export functionality would be implemented here');
        });
    }
    
    const pngExport = document.getElementById('export-png');
    if (pngExport) {
        pngExport.addEventListener('click', () => {
            // Get the current active section
            const section = document.querySelector('.dashboard-section.active');
            if (!section) return;
            
            // Find all charts in the section
            const charts = section.querySelectorAll('[id$="-chart"]');
            
            // For demonstration, we'll just export the first chart if any
            if (charts.length > 0) {
                const chartId = charts[0].id;
                try {
                    Plotly.downloadImage(chartId, {
                        format: 'png',
                        width: 1200,
                        height: 800,
                        filename: chartId
                    });
                } catch (e) {
                    console.error('Error exporting chart:', e);
                    showInfoMessage('Error exporting chart: ' + e.message);
                }
            } else {
                showInfoMessage('No charts found to export');
            }
        });
    }
    
    // Fullscreen button
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', () => {
            const section = document.querySelector('.dashboard-section.active');
            if (section) {
                toggleFullscreen(section);
            }
        });
    }
}

/**
 * Toggle fullscreen mode for an element
 */
function toggleFullscreen(element) {
    if (!document.fullscreenElement) {
        element.requestFullscreen().catch(err => {
            showErrorMessage(`Error attempting to enable fullscreen: ${err.message}`);
        });
    } else {
        document.exitFullscreen();
    }
}

/**
 * Helper functions for UI
 */
function showLoadingIndicator() {
    let loader = document.querySelector('.loading-overlay');
    
    if (!loader) {
        loader = document.createElement('div');
        loader.className = 'loading-overlay';
        loader.innerHTML = `
            <div class="spinner-border text-primary loading-spinner" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        `;
        document.body.appendChild(loader);
    }
    
    loader.style.display = 'flex';
}

function hideLoadingIndicator() {
    const loader = document.querySelector('.loading-overlay');
    if (loader) {
        loader.style.display = 'none';
    }
}

function showErrorMessage(message) {
    const alertElement = document.createElement('div');
    alertElement.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
    alertElement.style.zIndex = '9999';
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    document.body.appendChild(alertElement);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertElement.parentNode) {
            alertElement.parentNode.removeChild(alertElement);
        }
    }, 5000);
}

function showInfoMessage(message) {
    const alertElement = document.createElement('div');
    alertElement.className = 'alert alert-info alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
    alertElement.style.zIndex = '9999';
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    document.body.appendChild(alertElement);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (alertElement.parentNode) {
            alertElement.parentNode.removeChild(alertElement);
        }
    }, 3000);
}

// Initialize the dashboard when the document is loaded
document.addEventListener('DOMContentLoaded', initDashboard);
"""

def setup_directory_structure():
    """Set up the directory structure for the dashboard"""
    # Directories to create
    directories = [
        'dashboard',
        'dashboard/data',
        'dashboard/charts',
        'dashboard/assets',
        'dashboard/assets/css',
        'dashboard/assets/js',
        'dashboard/assets/img'
    ]
    
    # Create directories
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            raise
    
    logger.info("Directory structure set up successfully")
    return directories

def generate_charts():
    """Generate static chart files"""
    charts_dir = "dashboard/charts"
    
    # Make sure directory exists
    Path(charts_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate each chart file
    for chart_name, chart_data in CHART_DEFINITIONS.items():
        chart_path = os.path.join(charts_dir, chart_name)
        try:
            with open(chart_path, 'w') as f:
                json.dump(chart_data, f, indent=2)
            logger.info(f"Generated chart: {chart_name}")
        except Exception as e:
            logger.error(f"Error generating chart {chart_name}: {e}")
    
    logger.info(f"Generated {len(CHART_DEFINITIONS)} chart files")

def generate_sample_data():
    """Generate sample data files"""
    data_dir = "dashboard/data"
    
    # Make sure directory exists
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Create processed_data.csv
    processed_data_path = os.path.join(data_dir, "processed_data.csv")
    with open(processed_data_path, 'w') as f:
        f.write(SAMPLE_PROCESSED_DATA)
    logger.info(f"Generated sample data: processed_data.csv")
    
    # Create process_types.csv
    process_types_path = os.path.join(data_dir, "process_types.csv")
    with open(process_types_path, 'w') as f:
        f.write(SAMPLE_PROCESS_TYPES)
    logger.info(f"Generated sample data: process_types.csv")

def update_dashboard_js():
    """Update dashboard.js with fixed paths for GitHub Pages"""
    js_dir = "dashboard/assets/js"
    
    # Make sure directory exists
    Path(js_dir).mkdir(parents=True, exist_ok=True)
    
    # Write updated dashboard.js
    dashboard_js_path = os.path.join(js_dir, "dashboard.js")
    with open(dashboard_js_path, 'w') as f:
        f.write(DASHBOARD_JS)
    logger.info(f"Updated dashboard.js with GitHub Pages compatible paths")

def copy_index_to_root():
    """Copy index.html to root directory if it exists in dashboard"""
    source_path = "dashboard/index.html"
    dest_path = "index.html"
    
    if os.path.exists(source_path):
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied index.html from dashboard to root directory")
            return True
        except Exception as e:
            logger.error(f"Error copying index.html to root: {e}")
            return False
    elif os.path.exists(dest_path):
        logger.info(f"index.html already exists in root directory")
        return True
    else:
        logger.warning(f"index.html not found in either location")
        return False

def run():
    """Run the dashboard setup process"""
    logger.info("Starting SOCAR Dashboard setup")
    
    try:
        # 1. Set up directory structure
        setup_directory_structure()
        
        # 2. Generate static chart files
        generate_charts()
        
        # 3. Generate sample data
        generate_sample_data()
        
        # 4. Update dashboard.js with fixed paths
        update_dashboard_js()
        
        # 5. Copy index.html to root if it exists
        copy_index_to_root()
        
        logger.info("✅ Dashboard setup completed successfully")
        logger.info("You should now see charts in your GitHub Pages deployment")
        
    except Exception as e:
        logger.error(f"❌ Error during dashboard setup: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run())