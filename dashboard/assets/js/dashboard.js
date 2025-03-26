/**
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
        const lines = csvText.split('\n');
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
                
                if (char === '"' && (j === 0 || line[j-1] !== '\\')) {
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
