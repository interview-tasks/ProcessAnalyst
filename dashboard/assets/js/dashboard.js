/**
 * SOCAR Process Analysis Dashboard
 * Main dashboard functionality
 */

// Dashboard configuration
const config = {
    dataPath: './dashboard/data/',
    chartsPath: './dashboard/charts/',
    processDependencies: {
        overview: ['process_efficiency.json', 'energy_safety.json', 'roi_by_process.json'],
        efficiency: ['process_hierarchy.json', 'efficiency_correlation.json'],
        energy: ['energy_safety.json'],
        safety: ['safety_heatmap.json', 'process_step_safety.json'],
        catalyst: ['catalyst_matrix.json', 'catalyst_parallel.json'],
        parameter: ['parameter_correlation.json'],
        roi: ['roi_by_process.json'],
        quality: []
    },
    chartContainers: {
        // Overview section
        'process-efficiency-chart': 'process_efficiency.json',
        'energy-safety-chart': 'energy_safety.json',
        'roi-waterfall-chart': 'roi_by_process.json',
        
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
        'roi-waterfall-detail-chart': 'roi_by_process.json',
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
        showNotification('Dashboard loaded successfully');
    } catch (error) {
        console.error('Dashboard initialization error:', error);
        showErrorMessage('Failed to initialize dashboard. See console for details.');
    } finally {
        hideLoadingIndicator();
    }
}

/**
 * Load main processed data with improved error handling
 */
async function loadProcessedData() {
    try {
        // Enhanced error handling for data loading
        console.log(`Attempting to load processed data from ${config.dataPath}processed_data.csv`);
        
        // Load main dataset
        const response = await fetch(`${config.dataPath}processed_data.csv`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
        }
        
        const csvText = await response.text();
        console.log(`Successfully loaded CSV with length: ${csvText.length} characters`);
        
        if (csvText.trim().length === 0) {
            throw new Error("CSV file is empty");
        }
        
        const processedData = parseCSV(csvText);
        
        if (processedData.length === 0) {
            throw new Error("No data rows found in CSV");
        }
        
        dashboardData.processedData = processedData;
        
        // Extract unique values for filters
        dashboardData.processTypes = [...new Set(processedData.map(row => row['Proses Tipi']))];
        dashboardData.processSteps = [...new Set(processedData.map(row => row['Proses Addımı']))];
        
        if ('İstifadə Edilən Katalizatorlar' in processedData[0]) {
            dashboardData.catalysts = [...new Set(processedData.map(row => row['İstifadə Edilən Katalizatorlar']))];
        }
        
        // Load aggregated datasets
        try {
            // First try process_types.csv
            const typesResponse = await fetch(`${config.dataPath}process_types.csv`);
            if (typesResponse.ok) {
                const typesText = await typesResponse.text();
                dashboardData.processTypesData = parseCSV(typesText);
                console.log('Successfully loaded process_types.csv');
            } else {
                console.warn(`Failed to load process_types.csv: ${typesResponse.status}`);
            }
            
            // Try to load other aggregated datasets
            const aggregations = ['process_steps', 'safety_parameters'];
            for (const agg of aggregations) {
                try {
                    const aggResponse = await fetch(`${config.dataPath}${agg}.csv`);
                    if (aggResponse.ok) {
                        const aggText = await aggResponse.text();
                        dashboardData[agg] = parseCSV(aggText);
                        console.log(`Successfully loaded ${agg}.csv`);
                    }
                } catch (e) {
                    console.warn(`Error loading ${agg}.csv:`, e);
                }
            }
        } catch (e) {
            console.warn('Error loading aggregated datasets:', e);
        }
        
        console.log('Data loaded successfully:', dashboardData);
        
    } catch (error) {
        console.error('Error loading processed data:', error);
        // Create minimal fallback data for demo purposes if data loading fails
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
    dashboardData.catalysts = ['Catalyst A', 'Catalyst B'];
    
    // Create simple aggregated data
    dashboardData.processTypesData = [
        {'Proses Tipi': 'Type A', 'Emalın Səmərəliliyi (%)': 92.3, 'Emal Həcmi (ton)': 250, 'Energy_per_ton': 1.8, 'Təhlükəsizlik Hadisələri': 3},
        {'Proses Tipi': 'Type B', 'Emalın Səmərəliliyi (%)': 92.5, 'Emal Həcmi (ton)': 210, 'Energy_per_ton': 1.7, 'Təhlükəsizlik Hadisələri': 3}
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
 * Load chart data for a specific section with improved error handling
 */
async function loadChartData(section) {
    const dependencies = config.processDependencies[section] || [];
    let loadedCharts = 0;
    
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
                loadedCharts++;
            } catch (error) {
                console.error(`Error loading chart data for ${chartFile}:`, error);
                
                // Create fallback chart data
                console.log(`Creating fallback chart for ${chartFile}`);
                dashboardData.chartData[chartFile] = createFallbackChart(chartFile);
            }
        } else {
            loadedCharts++;
        }
    }
    
    if (loadedCharts === 0 && dependencies.length > 0) {
        console.warn(`Failed to load any charts for section: ${section}`);
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
            x: dashboardData.processTypes || ['Type A', 'Type B', 'Type C', 'Type D'],
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
    } else if (chartFile.includes('catalyst')) {
        if (chartFile.includes('matrix')) {
            data = [{
                x: [1.5, 1.8, 2.1, 1.7],
                y: [93, 91, 94, 92],
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: 15,
                    color: [1, 2, 3, 2]
                },
                text: ['Catalyst A', 'Catalyst B', 'Catalyst C', 'Catalyst D'],
                name: 'Catalysts'
            }];
        } else {
            // Basic fallback for parallel charts
            data = [{
                type: 'parcoords',
                line: {
                    color: 'blue'
                },
                dimensions: [
                    {range: [90, 100], label: 'Efficiency', values: [95, 92, 98, 91]},
                    {range: [0, 3], label: 'Energy', values: [1.2, 2.2, 1.8, 2.5]},
                    {range: [0, 5], label: 'Incidents', values: [0, 2, 1, 3]}
                ]
            }];
        }
    } else if (chartFile.includes('roi')) {
        // ROI-related charts
        data = [{
            x: dashboardData.processTypes || ['Type A', 'Type B', 'Type C', 'Type D'],
            y: [150000, 220000, 180000, 120000],
            type: 'bar',
            name: 'Annual Savings'
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
 * Render all charts for the current section with improved error handling
 */
function renderCharts(section) {
    const containers = Object.keys(config.chartContainers).filter(key => {
        const element = document.getElementById(key);
        return element && element.closest(`#${section}-section`);
    });
    
    if (containers.length === 0) {
        console.warn(`No chart containers found for section: ${section}`);
    }
    
    let renderedCount = 0;
    
    for (const containerId of containers) {
        const chartFile = config.chartContainers[containerId];
        const containerElement = document.getElementById(containerId);
        
        if (!containerElement) {
            console.warn(`Container element not found: ${containerId}`);
            continue;
        }
        
        try {
            // Clear any previous content
            Plotly.purge(containerId);
            
            // Show loading state
            containerElement.innerHTML = `
                <div class="d-flex justify-content-center align-items-center h-100">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
            
            if (dashboardData.chartData[chartFile]) {
                const chartData = dashboardData.chartData[chartFile];
                
                // Validate chart data
                if (!chartData || !chartData.data || !Array.isArray(chartData.data)) {
                    throw new Error(`Invalid chart data structure for ${chartFile}`);
                }
                
                // Try to render the chart
                setTimeout(() => {
                    try {
                        Plotly.newPlot(containerId, chartData.data, chartData.layout, {
                            responsive: true,
                            displayModeBar: true,
                            displaylogo: false,
                            modeBarButtonsToRemove: ['lasso2d', 'select2d']
                        });
                        renderedCount++;
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
                }, 100); // Short timeout to let the DOM update
            } else {
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
                        renderedCount++;
                        console.log(`Successfully rendered chart: ${containerId}`);
                    })
                    .catch(error => {
                        console.error(`Error loading chart data for ${chartFile}:`, error);
                        
                        // Create and render fallback chart
                        const fallbackChart = createFallbackChart(chartFile);
                        Plotly.newPlot(containerId, fallbackChart.data, fallbackChart.layout, {
                            responsive: true,
                            displayModeBar: true,
                            displaylogo: false,
                            modeBarButtonsToRemove: ['lasso2d', 'select2d']
                        });
                        
                        // Add warning indicator
                        containerElement.insertAdjacentHTML('beforeend', `
                            <div class="alert alert-warning position-absolute top-0 end-0 m-2" style="z-index: 1000; opacity: 0.8;">
                                <small><i class="bi bi-exclamation-triangle-fill"></i> Demo data</small>
                            </div>
                        `);
                    });
            }
        } catch (error) {
            console.error(`Error setting up chart ${containerId}:`, error);
            containerElement.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    Failed to set up chart: ${error.message}
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
    
    // Get values from the processed data
    if (dashboardData.processTypesData && dashboardData.processTypesData.length > 0) {
        // Calculate average efficiency
        efficiencyValue = dashboardData.processTypesData.reduce((sum, row) => 
            sum + (row['Emalın Səmərəliliyi (%)'] || 0), 0) / dashboardData.processTypesData.length;
        
        // Calculate average energy per ton
        if ('Energy_per_ton' in dashboardData.processTypesData[0]) {
            energyValue = dashboardData.processTypesData.reduce((sum, row) => 
                sum + (row['Energy_per_ton'] || 0), 0) / dashboardData.processTypesData.length;
        } else {
            energyValue = 1.8; // Default value
        }
    } else if (dashboardData.processedData && dashboardData.processedData.length > 0) {
        // Calculate from main data if aggregate not available
        efficiencyValue = dashboardData.processedData.reduce((sum, row) => 
            sum + (row['Emalın Səmərəliliyi (%)'] || 0), 0) / dashboardData.processedData.length;
        
        if ('Energy_per_ton' in dashboardData.processedData[0]) {
            energyValue = dashboardData.processedData.reduce((sum, row) => 
                sum + (row['Energy_per_ton'] || 0), 0) / dashboardData.processedData.length;
        } else if ('Enerji İstifadəsi (kWh)' in dashboardData.processedData[0] && 'Emal Həcmi (ton)' in dashboardData.processedData[0]) {
            const totalEnergy = dashboardData.processedData.reduce((sum, row) => 
                sum + (row['Enerji İstifadəsi (kWh)'] || 0), 0);
            const totalVolume = dashboardData.processedData.reduce((sum, row) => 
                sum + (row['Emal Həcmi (ton)'] || 0), 0);
            energyValue = totalEnergy / totalVolume;
        } else {
            energyValue = 1.8; // Default value
        }
    } else {
        // Default values if no data
        efficiencyValue = 92.5;
        energyValue = 1.8;
    }
    
    // Safety incidents per 1000 tons
    if (dashboardData.processedData && dashboardData.processedData.length > 0) {
        const totalIncidents = dashboardData.processedData.reduce((sum, row) => 
            sum + (row['Təhlükəsizlik Hadisələri'] || 0), 0);
        const totalVolume = dashboardData.processedData.reduce((sum, row) => 
            sum + (row['Emal Həcmi (ton)'] || 0), 0);
        
        safetyValue = totalVolume > 0 ? (totalIncidents / totalVolume) * 1000 : 3.2;
    } else {
        safetyValue = 3.2; // Default value
    }
    
    // Potential savings (simplified)
    savingsValue = 2.15;  // Default value
    
    // Update KPI elements if they exist
    updateKPIElement('kpi-efficiency', `${efficiencyValue.toFixed(1)}%`);
    updateKPIElement('kpi-energy', `${energyValue.toFixed(1)} kWh/ton`);
    updateKPIElement('kpi-safety', `${safetyValue.toFixed(1)}`);
    updateKPIElement('kpi-savings', `₼${savingsValue.toFixed(2)}M`);
    
    // Update progress bars if they exist
    updateProgressBar('kpi-efficiency', efficiencyValue);
    updateProgressBar('kpi-energy', 100 - ((energyValue - 1) / 2 * 100)); // Scale 1-3 kWh to 100-0%
    updateProgressBar('kpi-safety', 100 - (safetyValue / 10 * 100)); // Scale 0-10 to 100-0%
    updateProgressBar('kpi-savings', savingsValue / 3 * 100); // Scale 0-3M to 0-100%
}

/**
 * Helper function to update KPI element text
 */
function updateKPIElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

/**
 * Helper function to update progress bar
 */
function updateProgressBar(id, percentage) {
    const element = document.getElementById(id);
    if (element) {
        const progressBar = element.closest('.card').querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
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
                const sectionTitleElement = document.getElementById('section-title');
                if (sectionTitleElement) {
                    sectionTitleElement.textContent = tab.textContent.trim();
                }
                    
                // Update current section in state
                dashboardData.currentSection = section;
            } catch (error) {
                console.error(`Error switching to section ${section}:`, error);
                showErrorMessage(`Failed to load ${section} section: ${error.message}`);
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
    } else {
        console.error(`Section not found: ${sectionId}`);
        showErrorMessage(`Section not found: ${sectionId}`);
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
    
    // Set up apply and reset buttons
    const applyBtn = document.getElementById('apply-filters');
    if (applyBtn) {
        applyBtn.addEventListener('click', applyFilters);
    }
    
    const resetBtn = document.getElementById('reset-filters');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetFilters);
    }
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
    
    console.log('Filter changed:', filterId, filterValue);
}

/**
 * Apply filters
 */
function applyFilters() {
    // Get current filter values
    const processType = document.getElementById('process-type-filter')?.value || 'all';
    const processStep = document.getElementById('process-step-filter')?.value || 'all';
    const catalyst = document.getElementById('catalyst-filter')?.value || 'all';
    
    // Get efficiency range if available
    const efficiencyMin = parseFloat(document.getElementById('efficiency-min')?.value || 0);
    const efficiencyMax = parseFloat(document.getElementById('efficiency-max')?.value || 100);
    
    // Get temperature range if available
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
    showNotification('Filters applied. Note: Filter functionality is limited in this version.');
    
    // Reload current section
    renderCharts(dashboardData.currentSection);
}

/**
 * Reset filters
 */
function resetFilters() {
    // Reset filter dropdowns
    const typeFilter = document.getElementById('process-type-filter');
    if (typeFilter) typeFilter.value = 'all';
    
    const stepFilter = document.getElementById('process-step-filter');
    if (stepFilter) stepFilter.value = 'all';
    
    const catalystFilter = document.getElementById('catalyst-filter');
    if (catalystFilter) catalystFilter.value = 'all';
    
    // Reset range inputs
    const efficiencyMin = document.getElementById('efficiency-min');
    if (efficiencyMin) efficiencyMin.value = 85;
    
    const efficiencyMax = document.getElementById('efficiency-max');
    if (efficiencyMax) efficiencyMax.value = 100;
    
    const tempMin = document.getElementById('temp-min');
    if (tempMin) tempMin.value = 0;
    
    const tempMax = document.getElementById('temp-max');
    if (tempMax) tempMax.value = 500;
    
    // Reset filter state
    dashboardData.currentFilters = {
        processType: 'all',
        processStep: 'all',
        catalyst: 'all',
        efficiencyRange: [85, 100],
        temperatureRange: [0, 500]
    };
    
    // Show notification
    showNotification('Filters have been reset to default values');
    
    // Reload current section
    renderCharts(dashboardData.currentSection);
}

/**
 * Set up ROI calculator
 */
function setupROICalculator() {
    const form = document.getElementById('roi-calculator-form');
    
    if (form) {
        form.addEventListener('submit', (event) => {
            event.preventDefault();
            
            const processType = document.getElementById('process-type-input')?.value || 'Type A';
            const implementationCost = parseFloat(document.getElementById('implementation-cost')?.value || 1000000);
            const efficiencyImprovement = parseFloat(document.getElementById('efficiency-improvement')?.value || 5);
            const energyReduction = parseFloat(document.getElementById('energy-reduction')?.value || 10);
            const safetyImprovement = parseFloat(document.getElementById('safety-improvement')?.value || 15);
            
            calculateROI(processType, implementationCost, efficiencyImprovement, energyReduction, safetyImprovement);
        });
    }
}

/**
 * Calculate ROI based on inputs
 */
function calculateROI(processType, implementationCost, efficiencyImprovement, energyReduction, safetyImprovement) {
    const resultsContainer = document.getElementById('roi-results');
    if (!resultsContainer) return;
    
    try {
        // Find process data or use fallback
        let processData;
        if (dashboardData.processTypesData) {
            processData = dashboardData.processTypesData.find(p => p['Proses Tipi'] === processType);
        }
        
        // Use fallback data if not found
        if (!processData) {
            processData = {
                'Proses Tipi': processType,
                'Emalın Səmərəliliyi (%)': 92.5,
                'Emal Həcmi (ton)': 1000,
                'Əməliyyat Xərcləri (AZN)': 500000
            };
        }
        
        // Calculate metrics
        const annualVolume = processData['Emal Həcmi (ton)'] * 12; // Assuming monthly data
        const costPerTon = 500; // Placeholder value
        
        // Calculate savings
        const annualCost = annualVolume * costPerTon;
        
        // Savings components
        const efficiencySavings = annualCost * 0.4 * (efficiencyImprovement / 100);
        const energySavings = annualCost * 0.3 * (energyReduction / 100);
        const safetySavings = annualCost * 0.2 * (safetyImprovement / 100);
        const otherSavings = annualCost * 0.1 * 0.05; // 5% on remaining 10%
        
        const totalSavings = efficiencySavings + energySavings + safetySavings + otherSavings;
        
        // ROI metrics
        const roiMonths = implementationCost / (totalSavings / 12);
        const fiveYearReturn = totalSavings * 5 - implementationCost;
        const roi5Years = (fiveYearReturn / implementationCost) * 100;
        
        // Create results HTML
        resultsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <div class="alert alert-success mb-4">
                        <div class="d-flex align-items-center mb-2">
                            <div class="display-5 me-3">₼${totalSavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</div>
                            <div class="fs-3 text-success">Annual Savings</div>
                        </div>
                        <div class="progress mb-2" style="height: 10px;">
                            <div class="progress-bar bg-success" role="progressbar" style="width: ${Math.min(100, (totalSavings / annualCost * 100))}%"></div>
                        </div>
                        <div class="small text-muted">Represents ${(totalSavings / annualCost * 100).toFixed(1)}% of annual operating costs</div>
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
                                <td class="text-end"><strong>₼${totalSavings.toLocaleString('en-US', {maximumFractionDigits: 0})}</strong></td>
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
 * Set up event listeners
 */
function setupEventListeners() {
    // Toggle sidebar on mobile
    const toggleSidebarBtn = document.getElementById('toggle-sidebar');
    if (toggleSidebarBtn) {
        toggleSidebarBtn.addEventListener('click', function() {
            document.querySelector('.sidebar')?.classList.toggle('show');
        });
    }
    
    // Fullscreen button
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', toggleFullscreen);
    }
    
    // Export buttons
    document.getElementById('export-pdf')?.addEventListener('click', () => showNotification('PDF export functionality would be implemented here'));
    document.getElementById('export-png')?.addEventListener('click', exportCurrentChartAsPNG);
    document.getElementById('export-csv')?.addEventListener('click', () => showNotification('CSV export functionality would be implemented here'));
    document.getElementById('export-all-zip')?.addEventListener('click', () => showNotification('ZIP export functionality would be implemented here'));
    
    // Summary toggle
    document.getElementById('toggle-summary')?.addEventListener('click', function() {
        const summaryContents = document.getElementById('summary-contents');
        if (summaryContents) {
            const isVisible = summaryContents.style.display !== 'none';
            summaryContents.style.display = isVisible ? 'none' : 'block';
            this.textContent = isVisible ? 'Show Details' : 'Hide Details';
        }
    });
    
    // Refresh data button
    document.getElementById('refresh-data')?.addEventListener('click', refreshDashboard);
    
    // Add any other event listeners here
}

/**
 * Export current chart as PNG
 */
function exportCurrentChartAsPNG() {
    // Get the currently visible chart
    const activeSection = document.querySelector('.dashboard-section.active');
    if (!activeSection) {
        showErrorMessage('No active section found');
        return;
    }
    
    const charts = activeSection.querySelectorAll('[id$="-chart"]');
    
    if (charts.length > 0) {
        // Use Plotly's built-in download functionality for the first visible chart
        const chartId = charts[0].id;
        
        try {
            Plotly.downloadImage(chartId, {
                format: 'png',
                width: 1200,
                height: 800,
                filename: chartId
            });
            
            showNotification('Chart exported as PNG');
        } catch (error) {
            console.error('Error exporting chart:', error);
            showErrorMessage('Failed to export chart as PNG: ' + error.message);
        }
    } else {
        showErrorMessage('No charts found to export');
    }
}

/**
 * Toggle fullscreen mode
 */
function toggleFullscreen() {
    const mainElement = document.querySelector('main');
    if (!mainElement) return;
    
    if (!document.fullscreenElement) {
        mainElement.requestFullscreen().catch(err => {
            showErrorMessage(`Error attempting to enable fullscreen: ${err.message}`);
        });
    } else {
        document.exitFullscreen();
    }
}

/**
 * Refresh the dashboard
 */
function refreshDashboard() {
    showLoadingIndicator();
    
    // Show notification
    showNotification('Refreshing dashboard...');
    
    // Clear chart cache to force reload
    dashboardData.chartData = {};
    
    // Reload data and charts
    setTimeout(async () => {
        try {
            // Reload data
            await loadProcessedData();
            
            // Reload chart data for current section
            await loadChartData(dashboardData.currentSection);
            
            // Update UI
            setupFilters();
            renderKPIs();
            renderCharts(dashboardData.currentSection);
            
            // Show success notification
            showNotification('Dashboard refreshed successfully');
        } catch (error) {
            console.error('Error refreshing dashboard:', error);
            showErrorMessage(`Failed to refresh dashboard: ${error.message}`);
        } finally {
            hideLoadingIndicator();
        }
    }, 500);
}

/**
 * Show a notification message
 */
function showNotification(message) {
    const container = document.createElement('div');
    container.className = 'toast align-items-center text-white bg-primary border-0 position-fixed top-0 start-50 translate-middle-x mt-3';
    container.style.zIndex = '9999';
    container.setAttribute('role', 'alert');
    container.setAttribute('aria-live', 'assertive');
    container.setAttribute('aria-atomic', 'true');
    container.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    document.body.appendChild(container);
    
    // Initialize and show toast
    const toast = new bootstrap.Toast(container, { delay: 3000 });
    toast.show();
    
    // Remove after hiding
    container.addEventListener('hidden.bs.toast', function() {
        document.body.removeChild(container);
    });
}

/**
 * Show an error message
 */
function showErrorMessage(message) {
    const container = document.createElement('div');
    container.className = 'toast align-items-center text-white bg-danger border-0 position-fixed top-0 start-50 translate-middle-x mt-3';
    container.style.zIndex = '9999';
    container.setAttribute('role', 'alert');
    container.setAttribute('aria-live', 'assertive');
    container.setAttribute('aria-atomic', 'true');
    container.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="bi bi-exclamation-triangle-fill me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    document.body.appendChild(container);
    
    // Initialize and show toast
    const toast = new bootstrap.Toast(container, { delay: 5000 });
    toast.show();
    
    // Remove after hiding
    container.addEventListener('hidden.bs.toast', function() {
        document.body.removeChild(container);
    });
}

/**
 * Show loading indicator
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

/**
 * Hide loading indicator
 */
function hideLoadingIndicator() {
    const loader = document.querySelector('.loading-overlay');
    if (loader) {
        loader.style.display = 'none';
    }
}

// Initialize the dashboard when the document is loaded
document.addEventListener('DOMContentLoaded', initDashboard);