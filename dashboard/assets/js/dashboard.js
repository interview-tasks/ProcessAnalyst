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
        // Load main dataset
        const processedData = await fetch(`${config.dataPath}processed_data.csv`)
            .then(response => response.text())
            .then(csvText => parseCSV(csvText));
        
        dashboardData.processedData = processedData;
        
        // Extract unique values for filters
        dashboardData.processTypes = [...new Set(processedData.map(row => row['Proses Tipi']))];
        dashboardData.processSteps = [...new Set(processedData.map(row => row['Proses Addımı']))];
        dashboardData.catalysts = [...new Set(processedData.map(row => row['İstifadə Edilən Katalizatorlar']))];
        
        // Load aggregated datasets
        dashboardData.processTypesData = await fetch(`${config.dataPath}process_types.csv`)
            .then(response => response.text())
            .then(csvText => parseCSV(csvText));
        
        dashboardData.roiProjections = await fetch(`${config.dataPath}roi_projections.csv`)
            .then(response => response.text())
            .then(csvText => parseCSV(csvText));
        
        console.log('Data loaded:', dashboardData);
        
    } catch (error) {
        console.error('Error loading processed data:', error);
        throw new Error('Failed to load required data');
    }
}

/**
 * Parse CSV data
 */
function parseCSV(csvText) {
    const lines = csvText.split('\n');
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
                const chartData = await response.json();
                dashboardData.chartData[chartFile] = chartData;
            } catch (error) {
                console.error(`Error loading chart data for ${chartFile}:`, error);
            }
        }
    }
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
            sum + row['Energy_per_ton'], 0) / dashboardData.processTypesData.length;
    } else {
        efficiencyValue = 92.5;
        energyValue = 1.8;
    }
    
    // Safety incidents per 1000 tons (from processed data)
    if (dashboardData.processedData) {
        const totalIncidents = dashboardData.processedData.reduce((sum, row) => 
            sum + row['Təhlükəsizlik Hadisələri'], 0);
        const totalVolume = dashboardData.processedData.reduce((sum, row) => 
            sum + row['Emal Həcmi (ton)'], 0);
        
        safetyValue = (totalIncidents / totalVolume) * 1000;
    } else {
        safetyValue = 3.2;
    }
    
    // Potential savings from ROI projections
    if (dashboardData.roiProjections) {
        savingsValue = dashboardData.roiProjections.reduce((sum, row) => 
            sum + (row['Potential_Annual_Savings'] || 0), 0) / 1000000; // Convert to millions
    } else {
        savingsValue = 2.15;
    }
    
    // Update KPI elements
    document.getElementById('kpi-efficiency').textContent = `${efficiencyValue.toFixed(1)}%`;
    document.getElementById('kpi-energy').textContent = `${energyValue.toFixed(1)} kWh/ton`;
    document.getElementById('kpi-safety').textContent = `${safetyValue.toFixed(1)} per 1000t`;
    document.getElementById('kpi-savings').textContent = `₼${savingsValue.toFixed(2)}M`;
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
                document.getElementById('section-title').textContent = 
                    tab.textContent.trim();
                    
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
    document.getElementById('process-type-filter').addEventListener('change', handleFilterChange);
    document.getElementById('process-step-filter').addEventListener('change', handleFilterChange);
    document.getElementById('catalyst-filter').addEventListener('change', handleFilterChange);
    
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
    
    // TODO: Apply filters to charts (would need to modify the chart data)
    console.log('Filters changed:', dashboardData.currentFilters);
    
    // For a full implementation, we would need to apply these filters to the chart data
    // and redraw the charts. In a real application, this would involve either:
    // 1. Re-fetching filtered data from a backend API
    // 2. Filtering the data client-side and regenerating the charts
    
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
    document.getElementById('export-pdf').addEventListener('click', () => {
        showInfoMessage('PDF export functionality would be implemented here');
    });
    
    document.getElementById('export-png').addEventListener('click', () => {
        // Get the current active section
        const section = document.querySelector('.dashboard-section.active');
        if (!section) return;
        
        // Find all charts in the section
        const charts = section.querySelectorAll('[id$="-chart"]');
        
        // For demonstration, we'll just export the first chart if any
        if (charts.length > 0) {
            const chartId = charts[0].id;
            Plotly.downloadImage(chartId, {
                format: 'png',
                width: 1200,
                height: 800,
                filename: chartId
            });
        } else {
            showInfoMessage('No charts found to export');
        }
    });
    
    // Fullscreen button
    document.getElementById('fullscreen-btn').addEventListener('click', () => {
        const section = document.querySelector('.dashboard-section.active');
        if (section) {
            toggleFullscreen(section);
        }
    });
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
        alertElement.remove();
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
        alertElement.remove();
    }, 3000);
}

// Initialize the dashboard when the document is loaded
document.addEventListener('DOMContentLoaded', initDashboard);