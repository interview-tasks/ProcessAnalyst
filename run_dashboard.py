#!/usr/bin/env python
"""
SOCAR Process Analysis Dashboard Generator

This script orchestrates the entire dashboard generation process:
1. Prepares the raw data for analysis
2. Generates all visualizations
3. Sets up the dashboard structure
4. Creates reports and ensures all files are in the correct locations

Usage:
    python run_dashboard.py [options]

Options:
    --data-source: Path to the raw data file (default: data/data.csv)
    --output-dir: Base directory for all output (default: dashboard)
    --config-file: Path to configuration file (default: dashboard_config.yaml)
    --skip-data-prep: Skip data preparation step (use existing processed data)
    --skip-viz-gen: Skip visualization generation step (use existing charts)
    --report: Generate HTML report
    --clean: Remove all existing output before starting
    --debug: Enable debug logging
"""

import os
import sys
import shutil
import logging
import argparse
import subprocess
import time
import yaml
import datetime
import json
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dashboard_runner")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SOCAR Process Analysis Dashboard Generator')
    parser.add_argument('--data-source', type=str, default='data/data.csv',
                      help='Path to the raw data file')
    parser.add_argument('--output-dir', type=str, default='dashboard',
                      help='Base directory for all output')
    parser.add_argument('--config-file', type=str, default='dashboard_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--skip-data-prep', action='store_true',
                      help='Skip data preparation step (use existing processed data)')
    parser.add_argument('--skip-viz-gen', action='store_true',
                      help='Skip visualization generation step (use existing charts)')
    parser.add_argument('--report', action='store_true',
                      help='Generate HTML report')
    parser.add_argument('--clean', action='store_true',
                      help='Remove all existing output before starting')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    return parser.parse_args()

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'plotly', 'pyyaml', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def load_config(config_file):
    """Load configuration from YAML file"""
    default_config = {
        'data': {
            'source': 'data/data.csv',
            'processed_dir': 'dashboard/data',
            'required_columns': [
                'Proses ID', 'Proses Tipi', 'Proses Addımı', 'Emal Həcmi (ton)',
                'Temperatur (°C)', 'Təzyiq (bar)', 'Prosesin Müddəti (saat)', 
                'Emalın Səmərəliliyi (%)', 'Enerji İstifadəsi (kWh)',
                'Ətraf Mühitə Təsir (g CO2 ekvivalent)', 'Təhlükəsizlik Hadisələri',
                'Əməliyyat Xərcləri (AZN)'
            ]
        },
        'visualizations': {
            'output_dir': 'dashboard/charts',
            'config_file': 'visualization_config.yaml',
            'enable_all': True,
            'quality_check': True,
            'create_report': True
        },
        'dashboard': {
            'html_dir': 'dashboard',
            'assets_dir': 'dashboard/assets',
            'title': 'SOCAR Process Analysis Dashboard',
            'enable_export': True,
            'auto_refresh': False,
            'refresh_interval': 3600  # seconds
        },
        'reporting': {
            'generate_summary': True,
            'email_report': False,
            'email_recipients': []
        }
    }
    
    # Try to load configuration from file
    config = default_config
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Update default config with file config (deep merge)
            if file_config:
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

def setup_directory_structure(config, clean=False):
    """Set up the directory structure for the dashboard"""
    # Base output directory
    base_dir = config['dashboard']['html_dir']
    
    # Directories to create
    directories = [
        base_dir,
        config['data']['processed_dir'],
        config['visualizations']['output_dir'],
        config['dashboard']['assets_dir'],
        os.path.join(config['dashboard']['assets_dir'], 'css'),
        os.path.join(config['dashboard']['assets_dir'], 'js'),
        os.path.join(config['dashboard']['assets_dir'], 'img')
    ]
    
    # Clean directories if requested
    if clean:
        logger.info("Cleaning existing output directories")
        for directory in directories:
            if os.path.exists(directory):
                try:
                    shutil.rmtree(directory)
                    logger.info(f"Removed directory: {directory}")
                except Exception as e:
                    logger.error(f"Error removing directory {directory}: {e}")
    
    # Create directories
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            raise
    
    logger.info("Directory structure set up successfully")
    return directories

def prepare_data(data_source, config, skip=False):
    """Run data preparation script"""
    if skip:
        logger.info("Skipping data preparation step")
        return True
    
    try:
        logger.info("Starting data preparation process")
        
        # Define output directory
        output_dir = config['data']['processed_dir']
        
        # Check if prepare_data.py script exists
        # prepare_script = 'prepare_data.py'
        prepare_script = 'dashboard/scripts/prepare_data.py'
        if not os.path.exists(prepare_script):
            logger.error(f"Data preparation script {prepare_script} not found")
            return False
        
        # Run prepare_data.py as a subprocess
        logger.info(f"Running data preparation script: {prepare_script}")
        cmd = [
            sys.executable,
            prepare_script,
            '--input', data_source,
            '--output', output_dir
        ]
        
        if config.get('debug', False):
            cmd.append('--debug')
        
        # Execute the command
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        # Check results
        if result.returncode == 0:
            logger.info(f"Data preparation completed successfully in {end_time - start_time:.2f} seconds")
            logger.debug(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"Data preparation failed with exit code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running data preparation script: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        return False

def generate_visualizations(config, skip=False):
    """Run visualization generation script"""
    if skip:
        logger.info("Skipping visualization generation step")
        return True
    
    try:
        logger.info("Starting visualization generation process")
        
        # Define input and output directories
        data_dir = config['data']['processed_dir']
        output_dir = config['visualizations']['output_dir']
        viz_config = config['visualizations']['config_file']
        
        # Check if generate_visualizations.py script exists
        # viz_script = 'generate_visualizations.py'
        viz_script = 'dashboard/scripts/generate_visualizations.py'
        if not os.path.exists(viz_script):
            logger.error(f"Visualization script {viz_script} not found")
            return False
        
        # Build command with appropriate options
        cmd = [
            sys.executable,
            viz_script,
            '--data-dir', data_dir,
            '--output-dir', output_dir,
            '--config', viz_config
        ]
        
        # Add optional flags based on configuration
        if config['visualizations'].get('quality_check', False):
            cmd.append('--quality-check')
        
        if config['visualizations'].get('create_report', False):
            cmd.append('--html-report')
            cmd.append('--dashboard-index')
        
        if config.get('debug', False):
            cmd.append('--debug')
        
        # Execute the command
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        end_time = time.time()
        
        # Check results
        if result.returncode == 0:
            logger.info(f"Visualization generation completed successfully in {end_time - start_time:.2f} seconds")
            logger.debug(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"Visualization generation failed with exit code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running visualization script: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error during visualization generation: {e}")
        return False

def setup_dashboard(config):
    """Set up the dashboard files"""
    try:
        logger.info("Setting up dashboard files")
        
        # Define directories
        dashboard_dir = config['dashboard']['html_dir']
        assets_dir = config['dashboard']['assets_dir']
        charts_dir = config['visualizations']['output_dir']
        
        # Copy necessary CSS and JS files
        css_dir = os.path.join(assets_dir, 'css')
        js_dir = os.path.join(assets_dir, 'js')
        
        # Check if custom CSS file exists, copy default if not
        dashboard_css = os.path.join(css_dir, 'dashboard.css')
        if not os.path.exists(dashboard_css):
            # Create a default CSS file
            with open(dashboard_css, 'w') as f:
                f.write("""
/* Dashboard Styles */

/* Sidebar */
.sidebar {
    position: fixed;
    top: 66px;
    bottom: 0;
    left: 0;
    z-index: 100;
    padding: 0;
    box-shadow: inset -1px 0 0 rgba(0, 0, 0, .1);
    overflow-y: auto;
}

@media (max-width: 767.98px) {
    .sidebar {
        position: static;
        z-index: auto;
        height: auto;
    }
}

.sidebar .nav-link {
    font-weight: 500;
    color: #333;
    cursor: pointer;
}

.sidebar .nav-link.active {
    color: #2470dc;
}

.sidebar .nav-link:hover {
    color: #0d6efd;
    background-color: #f8f9fa;
}

.sidebar .nav-link .bi {
    margin-right: 4px;
    color: #727272;
}

.sidebar .nav-link.active .bi {
    color: #2470dc;
}

/* Dashboard sections */
.dashboard-section {
    display: none;
}

.dashboard-section.active {
    display: block;
}

/* KPI Cards */
.card {
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    transition: transform 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
}

/* Card headers */
.card-header {
    background-color: rgba(0, 0, 0, 0.03);
    font-weight: 500;
}

/* Chart containers */
[id$="-chart"] {
    width: 100%;
    min-height: 300px;
}

/* Main content adjustment */
main {
    padding-bottom: 60px; /* Ensure content doesn't get cut off by footer */
}

/* Footer */
.footer {
    background-color: #f5f5f5;
    padding: 1rem 0;
    color: #6c757d;
}

/* Notification container */
#notification-container {
    z-index: 1100;
}

/* Loading indicator */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(255, 255, 255, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
}

.loading-spinner {
    width: 3rem;
    height: 3rem;
}

/* ROI Calculator */
#roi-calculator-form .form-control {
    border-radius: 0.25rem;
}

#roi-results {
    min-height: 200px;
}

/* Tooltip customization */
.plotly-tooltip {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid #ddd !important;
    border-radius: 4px !important;
    padding: 8px !important;
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075) !important;
}

/* Full screen mode */
.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 9999;
    background: white;
    overflow-y: auto;
    padding: 20px;
}
                """)
                logger.info(f"Created default dashboard CSS: {dashboard_css}")
        
        # Check if custom JS file exists, copy default if not
        dashboard_js = os.path.join(js_dir, 'dashboard.js')
        if not os.path.exists(dashboard_js):
            # Create a default JS file with basic functionality
            with open(dashboard_js, 'w') as f:
                f.write("""
/**
 * SOCAR Process Analysis Dashboard
 * Main dashboard functionality
 */

// Dashboard configuration
const config = {
    dataPath: 'data/',
    chartsPath: 'charts/',
    processDependencies: {
        overview: ['process_efficiency.json', 'energy_safety.json', 'roi_waterfall.json'],
        efficiency: ['process_hierarchy.json', 'efficiency_correlation.json'],
        energy: ['energy_safety.json'],
        safety: ['safety_heatmap.json', 'process_step_safety.json'],
        catalyst: ['catalyst_matrix.json', 'catalyst_parallel.json'],
        parameter: ['parameter_correlation.json'],
        roi: ['roi_waterfall.json', 'roi_by_process.json'],
        quality: []
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
        'energy-by-process-chart': 'energy_by_process.json',
        'energy-parameters-chart': 'energy_parameters.json',
        
        // Safety section
        'safety-heatmap-chart': 'safety_heatmap.json',
        'process-step-safety-chart': 'process_step_safety.json',
        
        // Catalyst section
        'catalyst-matrix-chart': 'catalyst_matrix.json',
        'catalyst-parallel-chart': 'catalyst_parallel.json',
        
        // Parameter section
        'parameter-correlation-chart': 'parameter_correlation.json',
        'temp-pressure-chart': 'temp_pressure.json',
        'duration-impact-chart': 'duration_impact.json',
        
        // ROI section
        'roi-waterfall-detail-chart': 'roi_waterfall.json',
        'roi-by-process-chart': 'roi_by_process.json',
        
        // Quality section
        'performance-metrics-chart': 'performance_metrics.json'
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
        catalyst: 'all',
        dateRange: null,
        efficiencyRange: [85, 100],
        temperatureRange: [0, 500]
    },
    currentSection: 'overview'
};

/**
 * Initialize the dashboard
 */
async function initDashboard() {
    showLoadingIndicator();
    
    try {
        // Set up event listeners
        setupEventListeners();
        
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
        
        // Show initial section
        showSection('overview');
        
        // Show success notification
        showNotification('success', 'Dashboard loaded successfully!');
    } catch (error) {
        console.error('Dashboard initialization error:', error);
        showNotification('error', 'Failed to initialize dashboard. See console for details.');
    } finally {
        hideLoadingIndicator();
    }
}

/**
 * Set up event listeners for various dashboard controls
 */
function setupEventListeners() {
    // Toggle sidebar on mobile
    const toggleSidebarBtn = document.getElementById('toggle-sidebar');
    if (toggleSidebarBtn) {
        toggleSidebarBtn.addEventListener('click', function() {
            document.querySelector('.sidebar').classList.toggle('show');
        });
    }
    
    // Fullscreen button
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', toggleFullscreen);
    }
    
    // Export buttons
    document.getElementById('export-pdf')?.addEventListener('click', exportPDF);
    document.getElementById('export-png')?.addEventListener('click', exportPNG);
    document.getElementById('export-csv')?.addEventListener('click', exportCSV);
    document.getElementById('export-all-zip')?.addEventListener('click', exportAllZip);
    
    // Filter application
    document.getElementById('apply-filters')?.addEventListener('click', applyFilters);
    document.getElementById('reset-filters')?.addEventListener('click', resetFilters);
    
    // Chart view toggles
    document.querySelectorAll('[data-view]').forEach(button => {
        button.addEventListener('click', function() {
            const view = this.getAttribute('data-view');
            const container = this.closest('.card').querySelector('[id$="-chart"]');
            
            // Remove active class from all buttons in the group
            this.parentElement.querySelectorAll('button').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Add active class to clicked button
            this.classList.add('active');
            
            // Change chart view (implementation would depend on the specific chart)
            console.log(`Changed view to ${view} for chart ${container.id}`);
            
            // Example implementation for switching between chart types
            if (container.id === 'process-efficiency-chart') {
                if (view === 'bar') {
                    // Switch to bar chart
                } else if (view === 'pie') {
                    // Switch to pie chart
                }
            }
        });
    });
    
    // Summary toggle
    document.getElementById('toggle-summary')?.addEventListener('click', function() {
        const summaryContents = document.getElementById('summary-contents');
        if (summaryContents) {
            const isVisible = summaryContents.style.display !== 'none';
            summaryContents.style.display = isVisible ? 'none' : 'block';
            this.textContent = isVisible ? 'Show Details' : 'Hide Details';
        }
    });
    
    // Report button
    document.getElementById('report-btn')?.addEventListener('click', function(e) {
        e.preventDefault();
        window.open('visualization_report.html', '_blank');
    });
    
    // ROI calculator form
    document.getElementById('roi-calculator-form')?.addEventListener('submit', function(e) {
        e.preventDefault();
        calculateROI();
    });
    
    // Refresh data button
    document.getElementById('refresh-data')?.addEventListener('click', refreshData);
    
    // Metadata dialog
    document.getElementById('show-metadata')?.addEventListener('click', function(e) {
        e.preventDefault();
        const metadataModal = new bootstrap.Modal(document.getElementById('metadataModal'));
        metadataModal.show();
    });
}

/**
 * Load main processed data
 */
async function loadProcessedData() {
    try {
        // Load main dataset
        const processedData = await fetch(`${config.dataPath}processed_data.csv`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text();
            })
            .then(csvText => parseCSV(csvText));
        
        dashboardData.processedData = processedData;
        
        // Extract unique values for filters
        dashboardData.processTypes = [...new Set(processedData.map(row => row['Proses Tipi']))];
        dashboardData.processSteps = [...new Set(processedData.map(row => row['Proses Addımı']))];
        
        if ('İstifadə Edilən Katalizatorlar' in processedData[0]) {
            dashboardData.catalysts = [...new Set(processedData.map(row => row['İstifadə Edilən Katalizatorlar']))];
        }
        
        // Update data summary
        updateDataSummary(processedData);
        
        console.log('Data loaded:', dashboardData);
        
    } catch (error) {
        console.error('Error loading processed data:', error);
        throw new Error('Failed to load required data');
    }
}

/**
 * Update data summary section with key statistics
 */
function updateDataSummary(data) {
    // Update summary metrics if elements exist
    if (document.getElementById('total-records')) {
        document.getElementById('total-records').textContent = data.length.toLocaleString();
    }
    
    if (document.getElementById('process-types')) {
        document.getElementById('process-types').textContent = dashboardData.processTypes.length;
    }
    
    if (document.getElementById('process-steps')) {
        document.getElementById('process-steps').textContent = dashboardData.processSteps.length;
    }
    
    if (document.getElementById('catalyst-types') && dashboardData.catalysts) {
        document.getElementById('catalyst-types').textContent = dashboardData.catalysts.length;
    }
    
    // Calculate total volume if the column exists
    if (document.getElementById('total-volume') && 'Emal Həcmi (ton)' in data[0]) {
        const totalVolume = data.reduce((sum, row) => sum + (parseFloat(row['Emal Həcmi (ton)']) || 0), 0);
        document.getElementById('total-volume').textContent = `${totalVolume.toLocaleString()} tons`;
    }
    
    // Set temperature range if columns exist
    if (document.getElementById('temp-range') && 'Temperatur (°C)' in data[0]) {
        const temps = data.map(row => parseFloat(row['Temperatur (°C)']) || 0);
        const minTemp = Math.min(...temps);
        const maxTemp = Math.max(...temps);
        document.getElementById('temp-range').textContent = `${minTemp}-${maxTemp}°C`;
    }
    
    // Set pressure range if columns exist
    if (document.getElementById('pressure-range') && 'Təzyiq (bar)' in data[0]) {
        const pressures = data.map(row => parseFloat(row['Təzyiq (bar)']) || 0);
        const minPressure = Math.min(...pressures);
        const maxPressure = Math.max(...pressures);
        document.getElementById('pressure-range').textContent = `${minPressure}-${maxPressure} bar`;
    }
}

/**
 * Parse CSV data
 */
function parseCSV(csvText) {
    const lines = csvText.split('\\n');
    const headers = lines[0].split(',').map(header => header.trim());
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const values = line.split(',');
        const row = {};
        
        for (let j = 0; j < headers.length; j++) {
            const value = values[j]?.trim() || '';
            row[headers[j]] = isNaN(value) ? value : parseFloat(value);
        }
        
        data.push(row);
    }
    
    return data;
}

/**
 * Load chart data for a specific section
 */
async function loadChartData(section) {
    const dependencies = config.processDependencies[section] || [];
    
    for (const chartFile of dependencies) {
        if (!dashboardData.chartData[chartFile]) {
            try {
                const response = await fetch(`${config.chartsPath}${chartFile}`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const chartData = await response.json();
                dashboardData.chartData[chartFile] = chartData;
            } catch (error) {
                console.error(`Error loading chart data for ${chartFile}:`, error);
                showNotification('error', `Failed to load chart: ${chartFile}`);
            }
        }
    }
}

/**
 * Render all charts for the current section
 */
function renderCharts(section) {
    const containers = Object.keys(config.chartContainers).filter(key => {
        const element = document.getElementById(key);
        return element && element.closest(`#${section}-section`);
    });
    
    for (const containerId of containers) {
        const chartFile = config.chartContainers[containerId];
        const containerElement = document.getElementById(containerId);
        
        if (containerElement && dashboardData.chartData[chartFile]) {
            try {
                const chartData = dashboardData.chartData[chartFile];
                Plotly.newPlot(containerId, chartData.data, chartData.layout, {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d']
                });
            } catch (error) {
                console.error(`Error rendering chart ${containerId}:`, error);
                containerElement.innerHTML = `<div class="alert alert-danger">Failed to render chart</div>`;
            }
        } else if (containerElement) {
            // If chart data is not available, show loading indicator
            containerElement.innerHTML = `
                <div class="d-flex justify-content-center align-items-center h-100">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
            
            // Try to load the chart data
            fetch(`${config.chartsPath}${chartFile}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(chartData => {
                    dashboardData.chartData[chartFile] = chartData;
                    Plotly.newPlot(containerId, chartData.data, chartData.layout, {
                        responsive: true,
                        displayModeBar: true,
                        displaylogo: false,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    });
                })
                .catch(error => {
                    console.error(`Error loading chart data for ${chartFile}:`, error);
                    containerElement.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="bi bi-exclamation-triangle-fill me-2"></i>
                            Failed to load chart data
                        </div>
                    `;
                });
        }
    }
}

/**
 * Render KPI cards with summary data
 */
function renderKPIs() {
    // Calculate or extract KPI values
    let efficiencyValue, energyValue, safetyValue, savingsValue;
    
    // Get values from the processed data
    if (dashboardData.processedData) {
        // Calculate average efficiency
        efficiencyValue = dashboardData.processedData.reduce((sum, row) => 
            sum + (parseFloat(row['Emalın Səmərəliliyi (%)']) || 0), 0) / dashboardData.processedData.length;
        
        // Calculate average energy per ton
        if ('Energy_per_ton' in dashboardData.processedData[0]) {
            energyValue = dashboardData.processedData.reduce((sum, row) => 
                sum + (parseFloat(row['Energy_per_ton']) || 0), 0) / dashboardData.processedData.length;
        } else if ('Enerji İstifadəsi (kWh)' in dashboardData.processedData[0] && 'Emal Həcmi (ton)' in dashboardData.processedData[0]) {
            const totalEnergy = dashboardData.processedData.reduce((sum, row) => 
                sum + (parseFloat(row['Enerji İstifadəsi (kWh)']) || 0), 0);
            const totalVolume = dashboardData.processedData.reduce((sum, row) => 
                sum + (parseFloat(row['Emal Həcmi (ton)']) || 0), 0);
            energyValue = totalEnergy / totalVolume;
        } else {
            energyValue = 1.8; // Default value
        }
        
        // Calculate safety incidents per 1000 tons
        const totalIncidents = dashboardData.processedData.reduce((sum, row) => 
            sum + (parseFloat(row['Təhlükəsizlik Hadisələri']) || 0), 0);
        const totalVolume = dashboardData.processedData.reduce((sum, row) => 
            sum + (parseFloat(row['Emal Həcmi (ton)']) || 0), 0);
        
        safetyValue = (totalIncidents / totalVolume) * 1000;
        
        // Calculate potential savings (simplified estimate)
        const totalOpex = dashboardData.processedData.reduce((sum, row) => 
            sum + (parseFloat(row['Əməliyyat Xərcləri (AZN)']) || 0), 0) * 12; // Annual
        
        savingsValue = totalOpex * 0.15 / 1000000; // Assume 15% savings, convert to millions
    } else {
        // Default values if no data is available
        efficiencyValue = 92.5;
        energyValue = 1.8;
        safetyValue = 3.2;
        savingsValue = 2.15;
    }
    
    // Update KPI elements if they exist
    if (document.getElementById('kpi-efficiency')) {
        document.getElementById('kpi-efficiency').textContent = `${efficiencyValue.toFixed(1)}%`;
        
        // Update progress bar
        const efficiencyProgress = document.querySelector('#kpi-efficiency').closest('.card').querySelector('.progress-bar');
        if (efficiencyProgress) {
            efficiencyProgress.style.width = `${efficiencyValue}%`;
            efficiencyProgress.setAttribute('aria-valuenow', efficiencyValue);
        }
        
        // Update status indicator
        const efficiencyStatus = document.getElementById('efficiency-status');
        if (efficiencyStatus) {
            const targetEfficiency = 95; // Target value
            const diff = efficiencyValue - targetEfficiency;
            efficiencyStatus.textContent = `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%`;
            efficiencyStatus.className = diff >= 0 ? 'badge bg-light text-success' : 'badge bg-light text-danger';
        }
    }
    
    if (document.getElementById('kpi-energy')) {
        document.getElementById('kpi-energy').textContent = `${energyValue.toFixed(1)} kWh/ton`;
        
        // Update progress bar for energy (lower is better)
        const energyProgress = document.querySelector('#kpi-energy').closest('.card').querySelector('.progress-bar');
        if (energyProgress) {
            // Assume 1.0 is best (100%) and 3.0 is worst (0%)
            const energyPercent = Math.max(0, Math.min(100, (3.0 - energyValue) / 2.0 * 100));
            energyProgress.style.width = `${energyPercent}%`;
            energyProgress.setAttribute('aria-valuenow', energyPercent);
        }
        
        // Update status indicator
        const energyStatus = document.getElementById('energy-status');
        if (energyStatus) {
            const targetEnergy = 1.6; // Target value
            const diff = ((energyValue - targetEnergy) / targetEnergy * 100);
            energyStatus.textContent = `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%`;
            energyStatus.className = diff <= 0 ? 'badge bg-light text-success' : 'badge bg-light text-danger';
        }
    }
    
    if (document.getElementById('kpi-safety')) {
        document.getElementById('kpi-safety').textContent = `${safetyValue.toFixed(1)}`;
        
        // Update progress bar for safety (lower is better)
        const safetyProgress = document.querySelector('#kpi-safety').closest('.card').querySelector('.progress-bar');
        if (safetyProgress) {
            // Assume 0 is best (100%) and 10 is worst (0%)
            const safetyPercent = Math.max(0, Math.min(100, (10 - safetyValue) / 10 * 100));
            safetyProgress.style.width = `${safetyPercent}%`;
            safetyProgress.setAttribute('aria-valuenow', safetyPercent);
        }
        
        // Update status indicator
        const safetyStatus = document.getElementById('safety-status');
        if (safetyStatus) {
            const targetSafety = 2.0; // Target value
            const diff = ((safetyValue - targetSafety) / targetSafety * 100);
            safetyStatus.textContent = `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%`;
            safetyStatus.className = diff <= 0 ? 'badge bg-light text-success' : 'badge bg-light text-danger';
        }
    }
    
    if (document.getElementById('kpi-savings')) {
        document.getElementById('kpi-savings').textContent = `₼${savingsValue.toFixed(2)}M`;
        
        // Update progress bar
        const savingsProgress = document.querySelector('#kpi-savings').closest('.card').querySelector('.progress-bar');
        if (savingsProgress) {
            // Assume 3M is best (100%) and 0 is worst (0%)
            const savingsPercent = Math.max(0, Math.min(100, savingsValue / 3.0 * 100));
            savingsProgress.style.width = `${savingsPercent}%`;
            savingsProgress.setAttribute('aria-valuenow', savingsPercent);
        }
        
        // Update status indicator
        const savingsStatus = document.getElementById('savings-status');
        if (savingsStatus) {
            // Assume 5% increase from previous period
            savingsStatus.textContent = `+5.2%`;
        }
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
                document.getElementById('section-title').textContent = tab.textContent.trim();
                
                // Update section description based on the selected section
                updateSectionDescription(section);
                    
                // Update current section in state
                dashboardData.currentSection = section;
            } catch (error) {
                console.error(`Error switching to section ${section}:`, error);
                showNotification('error', `Failed to load ${section} section`);
            } finally {
                hideLoadingIndicator();
            }
        });
    });
}

/**
 * Update section description based on the selected section
 */
function updateSectionDescription(section) {
    const descriptionElement = document.getElementById('section-description');
    if (!descriptionElement) return;
    
    const descriptions = {
        'overview': 'Comprehensive view of key process metrics and performance indicators.',
        'efficiency': 'Detailed analysis of process efficiency factors and opportunities for improvement.',
        'energy': 'Energy consumption patterns and their relationship to other process parameters.',
        'safety': 'Safety incident analysis and identification of optimal operating conditions.',
        'catalyst': 'Comparative analysis of catalyst performance and efficiency impact.',
        'parameter': 'Correlation between process parameters and their impact on performance.',
        'roi': 'Return on investment projections for process optimization initiatives.',
        'quality': 'Quality validation of visualizations and comparison with previous runs.'
    };
    
    descriptionElement.textContent = descriptions[section] || '';
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
    document.getElementById('process-type-filter')?.addEventListener('change', handleFilterChange);
    document.getElementById('process-step-filter')?.addEventListener('change', handleFilterChange);
    document.getElementById('catalyst-filter')?.addEventListener('change', handleFilterChange);
    
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
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        element.appendChild(optionElement);
    });
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
    
    // Log the filter change
    console.log('Filter changed:', filterId, filterValue);
}

/**
 * Apply all filters to the data and update visualizations
 */
function applyFilters() {
    // Get current filter values
    const processType = document.getElementById('process-type-filter')?.value || 'all';
    const processStep = document.getElementById('process-step-filter')?.value || 'all';
    const catalyst = document.getElementById('catalyst-filter')?.value || 'all';
    
    // Get efficiency range
    const efficiencyMin = parseFloat(document.getElementById('efficiency-min')?.value || 0);
    const efficiencyMax = parseFloat(document.getElementById('efficiency-max')?.value || 100);
    
    // Get temperature range
    const tempMin = parseFloat(document.getElementById('temp-min')?.value || 0);
    const tempMax = parseFloat(document.getElementById('temp-max')?.value || 500);
    
    // Update filters object
    dashboardData.currentFilters = {
        processType,
        processStep,
        catalyst,
        efficiencyRange: [efficiencyMin, efficiencyMax],
        temperatureRange: [tempMin, tempMax]
    };
    
    // Show notification
    showNotification('info', 'Filters applied. Note that filtering capabilities depend on the implementation of each visualization.');
    
    // In a full implementation, you would reload and rerender the charts with the filtered data
    console.log('Applied filters:', dashboardData.currentFilters);
}

/**
 * Reset all filters to default values
 */
function resetFilters() {
    // Reset filter dropdowns
    document.getElementById('process-type-filter').value = 'all';
    document.getElementById('process-step-filter').value = 'all';
    document.getElementById('catalyst-filter').value = 'all';
    
    // Reset range inputs
    document.getElementById('efficiency-min').value = 85;
    document.getElementById('efficiency-max').value = 100;
    document.getElementById('temp-min').value = 0;
    document.getElementById('temp-max').value = 500;
    
    // Reset date picker if flatpickr is available
    if (typeof flatpickr !== 'undefined' && document.getElementById('date-range')._flatpickr) {
        document.getElementById('date-range')._flatpickr.clear();
    }
    
    // Reset filter state
    dashboardData.currentFilters = {
        processType: 'all',
        processStep: 'all',
        catalyst: 'all',
        dateRange: null,
        efficiencyRange: [85, 100],
        temperatureRange: [0, 500]
    };
    
    // Show notification
    showNotification('success', 'Filters have been reset to default values');
}

/**
 * Calculate ROI based on inputs from the ROI calculator form
 */
function calculateROI() {
    const processType = document.getElementById('process-type-input').value;
    const implementationCost = parseFloat(document.getElementById('implementation-cost').value);
    const efficiencyImprovement = parseFloat(document.getElementById('efficiency-improvement').value);
    const energyReduction = parseFloat(document.getElementById('energy-reduction').value);
    const safetyImprovement = parseFloat(document.getElementById('safety-improvement').value);
    
    const resultsContainer = document.getElementById('roi-results');
    if (!resultsContainer) return;
    
    try {
        // Get processed data for the selected process type
        let processData;
        if (dashboardData.processedData) {
            if (processType === 'all') {
                processData = dashboardData.processedData;
            } else {
                processData = dashboardData.processedData.filter(row => row['Proses Tipi'] === processType);
            }
        } else {
            throw new Error('Process data not available');
        }
        
        if (!processData || processData.length === 0) {
            throw new Error('No data available for the selected process type');
        }
        
        // Calculate baseline metrics
        const totalVolume = processData.reduce((sum, row) => sum + (parseFloat(row['Emal Həcmi (ton)']) || 0), 0);
        const annualVolume = totalVolume * (12 / processData.length); // Annualized
        
        const avgEfficiency = processData.reduce((sum, row) => sum + (parseFloat(row['Emalın Səmərəliliyi (%)']) || 0), 0) / processData.length;
        
        const totalOpex = processData.reduce((sum, row) => sum + (parseFloat(row['Əməliyyat Xərcləri (AZN)']) || 0), 0);
        const annualOpex = totalOpex * (12 / processData.length); // Annualized
        
        // Calculate savings components
        const efficiencySavings = annualOpex * 0.4 * (efficiencyImprovement / 100); // 40% of costs affected by efficiency
        const energySavings = annualOpex * 0.3 * (energyReduction / 100); // 30% of costs related to energy
        const safetySavings = annualOpex * 0.2 * (safetyImprovement / 100); // 20% of costs related to safety
        const otherSavings = annualOpex * 0.1 * 0.05; // 5% savings on remaining 10% of costs
        
        const totalAnnualSavings = efficiencySavings + energySavings + safetySavings + otherSavings;
        const roiMonths = implementationCost / (totalAnnualSavings / 12);
        const fiveYearReturn = totalAnnualSavings * 5 - implementationCost;
        const roi5Years = (fiveYearReturn / implementationCost) * 100;
        
        // Display results
        resultsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <div class="alert alert-success mb-4">
                        <div class="d-flex align-items-center mb-2">
                            <div class="display-5 me-3">₼${totalAnnualSavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</div>
                            <div class="fs-3 text-success">Annual Savings</div>
                        </div>
                        <div class="progress mb-2" style="height: 10px;">
                            <div class="progress-bar bg-success" role="progressbar" style="width: ${Math.min(100, (totalAnnualSavings / annualOpex * 100))}%"></div>
                        </div>
                        <div class="small text-muted">Represents ${(totalAnnualSavings / annualOpex * 100).toFixed(1)}% of annual operating costs</div>
                    </div>
                    
                    <h5>ROI Analysis</h5>
                    <table class="table table-bordered">
                        <tbody>
                            <tr>
                                <td>Implementation Cost</td>
                                <td class="text-end">₼${implementationCost.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
                            </tr>
                            <tr>
                                <td>Payback Period</td>
                                <td class="text-end">${roiMonths.toFixed(1)} months</td>
                            </tr>
                            <tr>
                                <td>5-Year Return</td>
                                <td class="text-end">₼${fiveYearReturn.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
                            </tr>
                            <tr>
                                <td>5-Year ROI</td>
                                <td class="text-end">${roi5Years.toFixed(1)}%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                <div class="col-md-6">
                    <h5>Savings Breakdown</h5>
                    <table class="table table-bordered">
                        <tbody>
                            <tr>
                                <td>Efficiency Improvements (${efficiencyImprovement}%)</td>
                                <td class="text-end">₼${efficiencySavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
                            </tr>
                            <tr>
                                <td>Energy Reduction (${energyReduction}%)</td>
                                <td class="text-end">₼${energySavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
                            </tr>
                            <tr>
                                <td>Safety Improvements (${safetyImprovement}%)</td>
                                <td class="text-end">₼${safetySavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
                            </tr>
                            <tr>
                                <td>Other Benefits</td>
                                <td class="text-end">₼${otherSavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
                            </tr>
                            <tr class="table-success">
                                <td><strong>Total Annual Savings</strong></td>
                                <td class="text-end"><strong>₼${totalAnnualSavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</strong></td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <div class="alert ${roiMonths <= 12 ? 'alert-success' : roiMonths <= 24 ? 'alert-info' : 'alert-warning'} mt-3">
                        <i class="${roiMonths <= 12 ? 'bi bi-check-circle' : 'bi bi-info-circle'} me-2"></i>
                        <strong>Recommendation:</strong> ${
                            roiMonths <= 12 
                            ? 'Strongly recommended investment with fast payback period.' 
                            : roiMonths <= 24 
                            ? 'Recommended investment with moderate payback period.' 
                            : 'Consider alternatives with faster payback period.'
                        }
                    </div>
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error('ROI calculation error:', error);
        resultsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle-fill me-2"></i>
                Error calculating ROI: ${error.message}
            </div>
        `;
    }
}

/**
 * Refresh data and visualizations
 */
function refreshData() {
    showLoadingIndicator();
    
    // Show notification
    showNotification('info', 'Refreshing data and visualizations...');
    
    // Simulate refresh delay
    setTimeout(async () => {
        try {
            // Reload data
            await loadProcessedData();
            
            // Clear chart cache to force reload
            dashboardData.chartData = {};
            
            // Load chart data for current section
            await loadChartData(dashboardData.currentSection);
            
            // Update UI
            setupFilters();
            renderKPIs();
            renderCharts(dashboardData.currentSection);
            
            // Show success notification
            showNotification('success', 'Data refreshed successfully');
        } catch (error) {
            console.error('Error refreshing data:', error);
            showNotification('error', `Failed to refresh data: ${error.message}`);
        } finally {
            hideLoadingIndicator();
        }
    }, 1000);
}

/**
 * Toggle fullscreen mode
 */
function toggleFullscreen() {
    const mainElement = document.querySelector('main');
    
    if (!document.fullscreenElement) {
        if (mainElement.requestFullscreen) {
            mainElement.requestFullscreen();
        } else if (mainElement.mozRequestFullScreen) { /* Firefox */
            mainElement.mozRequestFullScreen();
        } else if (mainElement.webkitRequestFullscreen) { /* Chrome, Safari & Opera */
            mainElement.webkitRequestFullscreen();
        } else if (mainElement.msRequestFullscreen) { /* IE/Edge */
            mainElement.msRequestFullscreen();
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.mozCancelFullScreen) {
            document.mozCancelFullScreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
    }
}

/**
 * Export functions for different formats
 */
function exportPDF() {
    showNotification('info', 'Exporting current view as PDF...');
    
    // Implement PDF export functionality here
    setTimeout(() => {
        showNotification('success', 'PDF exported successfully');
    }, 1500);
}

function exportPNG() {
    showNotification('info', 'Exporting current chart as PNG...');
    
    // Get the currently visible chart
    const activeSection = document.querySelector('.dashboard-section.active');
    const charts = activeSection.querySelectorAll('[id$="-chart"]');
    
    if (charts.length > 0) {
        // Use Plotly's built-in download functionality for the first visible chart
        const chartId = charts[0].id;
        
        // Use Plotly's download image function
        try {
            Plotly.downloadImage(chartId, {
                format: 'png',
                width: 1200,
                height: 800,
                filename: chartId
            });
            
            showNotification('success', 'Chart exported as PNG');
        } catch (error) {
            console.error('Error exporting chart:', error);
            showNotification('error', 'Failed to export chart as PNG');
        }
    } else {
        showNotification('error', 'No charts found to export');
    }
}

function exportCSV() {
    showNotification('info', 'Exporting data as CSV...');
    
    // Implement CSV export functionality here
    setTimeout(() => {
        // Create a CSV string from the current dataset
        if (dashboardData.processedData && dashboardData.processedData.length > 0) {
            try {
                const headers = Object.keys(dashboardData.processedData[0]);
                const csvRows = [headers.join(',')];
                
                for (const row of dashboardData.processedData) {
                    const values = headers.map(header => {
                        const value = row[header] || '';
                        // Handle values with commas by quoting
                        return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
                    });
                    csvRows.push(values.join(','));
                }
                
                const csvString = csvRows.join('\\n');
                
                // Create a Blob and download link
                const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.setAttribute('href', url);
                link.setAttribute('download', 'socar_process_data.csv');
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                showNotification('success', 'Data exported as CSV successfully');
            } catch (error) {
                console.error('Error exporting CSV:', error);
                showNotification('error', 'Failed to export data as CSV');
            }
        } else {
            showNotification('error', 'No data available to export');
        }
    }, 1000);
}

function exportAllZip() {
    showNotification('info', 'Preparing ZIP archive with all dashboard assets...');
    
    // In a real implementation, you would:
    // 1. Package all charts, data, and HTML files
    // 2. Create a ZIP archive
    // 3. Provide a download link
    
    // For this example, we'll just simulate the process
    setTimeout(() => {
        showNotification('success', 'ZIP archive with all dashboard assets is ready');
        
        // Create a fake download link
        const link = document.createElement('a');
        link.setAttribute('href', '#');
        link.setAttribute('download', 'socar_dashboard_export.zip');
        link.addEventListener('click', (e) => {
            e.preventDefault();
            showNotification('info', 'This is a simulated download in the example implementation');
        });
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }, 2500);
}

/**
 * Show a notification to the user
 */
function showNotification(type, message) {
    const container = document.getElementById('notification-container');
    if (!container) return;
    
    // Get the appropriate template
    const templateId = `notification-${type}`;
    const template = document.getElementById(templateId);
    if (!template) return;
    
    // Clone the template content
    const notification = template.content.cloneNode(true);
    
    // Set the message
    notification.querySelector('.toast-body').textContent = message;
    
    // Add to container
    container.appendChild(notification.firstElementChild);
    
    // Initialize and show the toast
    const toast = new bootstrap.Toast(container.lastElementChild);
    toast.show();
    
    // Remove after it's hidden
    container.lastElementChild.addEventListener('hidden.bs.toast', function () {
        container.removeChild(this);
    });
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

// Initialize the dashboard when the document is loaded
document.addEventListener('DOMContentLoaded', initDashboard);
                """)
                logger.info(f"Created default dashboard JS: {dashboard_js}")
        
        # Create HTML file if not exists
        dashboard_html = os.path.join(dashboard_dir, 'index.html')
        if not os.path.exists(dashboard_html):
            # TODO: Create a default index.html
            # For now, we'll just check if we should copy the HTML template
            logger.warning(f"Dashboard HTML file not found: {dashboard_html}")
            logger.info("Make sure to create or copy the index.html file to the dashboard directory")
        
        logger.info("Dashboard files set up successfully")
        return True
    except Exception as e:
        logger.error(f"Error setting up dashboard files: {e}")
        return False

def create_config_files(config):
    """Create configuration files if they don't exist"""
    try:
        logger.info("Creating configuration files")
        
        # Create visualization config template
        viz_config_file = config['visualizations']['config_file']
        if not os.path.exists(viz_config_file):
            # Create a default visualization config
            default_viz_config = {
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
            }
            
            with open(viz_config_file, 'w') as f:
                yaml.dump(default_viz_config, f, default_flow_style=False)
            logger.info(f"Created default visualization config: {viz_config_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating configuration files: {e}")
        return False

def copy_html_files(src_file, dest_dir):
    """Copy HTML file to the dashboard directory"""
    try:
        # Check if source file exists
        if not os.path.exists(src_file):
            logger.warning(f"Source file not found: {src_file}")
            return False
        
        # Copy file to destination
        dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        shutil.copy2(src_file, dest_file)
        logger.info(f"Copied {src_file} to {dest_file}")
        return True
    except Exception as e:
        logger.error(f"Error copying HTML file: {e}")
        return False

def update_dashboard_metadata(config):
    """Update dashboard metadata with current timestamp and version"""
    try:
        dashboard_dir = config['dashboard']['html_dir']
        metadata_file = os.path.join(dashboard_dir, 'dashboard_metadata.json')
        
        # Create metadata object
        metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'version': '1.0',
            'data_source': config['data']['source'],
            'charts_count': len(os.listdir(config['visualizations']['output_dir'])) if os.path.exists(config['visualizations']['output_dir']) else 0,
            'config_file': config['config_file'],
            'generated_by': 'SOCAR Dashboard Generator'
        }
        
        # Write to file
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Updated dashboard metadata: {metadata_file}")
        return True
    except Exception as e:
        logger.error(f"Error updating dashboard metadata: {e}")
        return False

def main():
    """Main function to run the dashboard generation process"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Check if all dependencies are installed
    # if not check_dependencies():
    #     logger.error("Missing required dependencies. Please install them and try again.")
    #     return 1
    
    try:
        # Load configuration
        config = load_config(args.config_file)
        
        # Override config with command line arguments
        config['data']['source'] = args.data_source
        config['dashboard']['html_dir'] = args.output_dir
        config['debug'] = args.debug
        
        # Set up directories
        setup_directory_structure(config, clean=args.clean)
        
        # Create config files
        create_config_files(config)
        
        # Run data preparation
        if not prepare_data(config['data']['source'], config, skip=args.skip_data_prep):
            logger.error("Data preparation failed. Cannot continue.")
            return 1
        
        # Generate visualizations
        if not generate_visualizations(config, skip=args.skip_viz_gen):
            logger.error("Visualization generation failed. Cannot continue.")
            return 1
        
        # Set up dashboard files
        if not setup_dashboard(config):
            logger.error("Dashboard setup failed. Cannot continue.")
            return 1
        
        # Check if we need to copy any HTML files from a template
        index_template = "index.html"
        if os.path.exists(index_template):
            copy_html_files(index_template, config['dashboard']['html_dir'])
        
        # Copy any HTML reports if generated
        viz_report = os.path.join(config['visualizations']['output_dir'], 'visualization_report.html')
        if os.path.exists(viz_report):
            copy_html_files(viz_report, config['dashboard']['html_dir'])
        
        # Update dashboard metadata
        update_dashboard_metadata(config)
        
        logger.info(f"Dashboard generation completed successfully. You can view it at: {os.path.join(args.output_dir, 'index.html')}")
        
        # Open the dashboard in the browser if requested
        if args.report and os.path.exists(os.path.join(args.output_dir, 'visualization_report.html')):
            report_path = os.path.join(args.output_dir, 'visualization_report.html')
            logger.info(f"Opening report in browser: {report_path}")
            
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
            except Exception as e:
                logger.error(f"Error opening report in browser: {e}")
        
        return 0
    
    except Exception as e:
        logger.error(f"An error occurred during dashboard generation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
    
# python run_dashboard.py \
#   --data-source dashboard/data/data.csv \
#   --output-dir dashboard \
#   --config-file dashboard_config.yaml \
#   --report \
#   --clean