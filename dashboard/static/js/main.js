// Process Analytics Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initDashboard();
    
    // Set up navigation
    setupNavigation();
    
    // Setup filters
    setupFilters();
});

// Global data store
const dashboardData = {
    summary: null,
    processTypes: null,
    efficiencyByProcess: null,
    energyVsEfficiency: null,
    co2VsCost: null,
    catalystEfficiency: null,
    processDuration: null,
    tempPressureEfficiency: null,
    timeline: null
};

// Dashboard initialization
async function initDashboard() {
    try {
        // Load all data in parallel
        await Promise.all([
            fetchSummaryData(),
            fetchProcessTypesData(),
            fetchEfficiencyByProcessData(),
            fetchEnergyVsEfficiencyData(),
            fetchCO2VsCostData(),
            fetchCatalystEfficiencyData(),
            fetchProcessDurationData(),
            fetchEfficiencyByTempPressureData(),
            fetchTimelineData()
        ]);
        
        // Render all visualizations
        renderDashboard();
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        showError('Dashboard yüklənərkən xəta baş verdi. Lütfən, səhifəni yeniləyin.');
    }
}

// Navigation setup
function setupNavigation() {
    const navLinks = document.querySelectorAll('.sidebar .nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all links
            navLinks.forEach(item => item.classList.remove('active'));
            
            // Add active class to clicked link
            this.classList.add('active');
            
            // Show the corresponding section
            const targetId = this.getAttribute('href').substring(1);
            showSection(targetId);
        });
    });
    
    // Initially show the overview section
    showSection('overview');
}

// Show specific section, hide others
function showSection(sectionId) {
    const sections = document.querySelectorAll('.dashboard-section');
    
    sections.forEach(section => {
        if (section.id === sectionId) {
            section.style.display = 'block';
        } else {
            section.style.display = 'none';
        }
    });
    
    // Scroll to top of the section
    window.scrollTo(0, 0);
}

// Setup data filters
function setupFilters() {
    const processTypeFilter = document.getElementById('processTypeFilter');
    
    // Wait for process types data to be loaded
    const checkDataLoaded = setInterval(() => {
        if (dashboardData.processTypes) {
            clearInterval(checkDataLoaded);
            
            // Populate filter dropdown
            dashboardData.processTypes.forEach(item => {
                const option = document.createElement('option');
                option.value = item.process_type;
                option.textContent = item.process_type;
                processTypeFilter.appendChild(option);
            });
            
            // Add event listener
            processTypeFilter.addEventListener('change', filterDashboardData);
        }
    }, 100);
}

// Filter dashboard data based on selected filters
function filterDashboardData() {
    const selectedProcessType = document.getElementById('processTypeFilter').value;
    
    // Apply filter and re-render
    if (selectedProcessType === 'all') {
        // Reset to all data
        renderDashboard();
    } else {
        // Filter data and re-render
        renderFilteredDashboard(selectedProcessType);
    }
}

// Data fetching functions
async function fetchSummaryData() {
    try {
        const response = await fetch('/api/data/summary');
        dashboardData.summary = await response.json();
    } catch (error) {
        console.error('Error fetching summary data:', error);
    }
}

async function fetchProcessTypesData() {
    try {
        const response = await fetch('/api/data/process_types');
        dashboardData.processTypes = await response.json();
    } catch (error) {
        console.error('Error fetching process types data:', error);
    }
}

async function fetchEfficiencyByProcessData() {
    try {
        const response = await fetch('/api/data/efficiency_by_process');
        dashboardData.efficiencyByProcess = await response.json();
    } catch (error) {
        console.error('Error fetching efficiency by process data:', error);
    }
}

async function fetchEnergyVsEfficiencyData() {
    try {
        const response = await fetch('/api/data/energy_vs_efficiency');
        dashboardData.energyVsEfficiency = await response.json();
    } catch (error) {
        console.error('Error fetching energy vs efficiency data:', error);
    }
}

async function fetchCO2VsCostData() {
    try {
        const response = await fetch('/api/data/co2_vs_cost');
        dashboardData.co2VsCost = await response.json();
    } catch (error) {
        console.error('Error fetching CO2 vs cost data:', error);
    }
}

async function fetchCatalystEfficiencyData() {
    try {
        const response = await fetch('/api/data/catalyst_efficiency');
        dashboardData.catalystEfficiency = await response.json();
    } catch (error) {
        console.error('Error fetching catalyst efficiency data:', error);
    }
}

async function fetchProcessDurationData() {
    try {
        const response = await fetch('/api/data/process_duration');
        dashboardData.processDuration = await response.json();
    } catch (error) {
        console.error('Error fetching process duration data:', error);
    }
}

async function fetchEfficiencyByTempPressureData() {
    try {
        const response = await fetch('/api/data/efficiency_by_temp_pressure');
        dashboardData.tempPressureEfficiency = await response.json();
    } catch (error) {
        console.error('Error fetching efficiency by temp and pressure data:', error);
    }
}

async function fetchTimelineData() {
    try {
        const response = await fetch('/api/data/timeline');
        dashboardData.timeline = await response.json();
    } catch (error) {
        console.error('Error fetching timeline data:', error);
    }
}

// Rendering functions
function renderDashboard() {
    // Render summary cards
    renderSummaryCards();
    
    // Render charts
    renderProcessTypesChart();
    renderEfficiencyByProcessChart();
    renderEnergyVsEfficiencyChart();
    renderCO2VsCostChart();
    renderCatalystEfficiencyChart();
    renderProcessDurationChart();
    renderTempPressureEfficiencyChart();
    renderTimelineChart();
}

function renderFilteredDashboard(processType) {
    // Filter the data
    const filteredData = {
        energyVsEfficiency: dashboardData.energyVsEfficiency.filter(item => item.process_type === processType),
        co2VsCost: dashboardData.co2VsCost.filter(item => item.process_type === processType),
        tempPressureEfficiency: dashboardData.tempPressureEfficiency.filter(item => item.process_type === processType),
        timeline: dashboardData.timeline.filter(item => item.process_type === processType)
    };
    
    // Re-render with filtered data
    renderEnergyVsEfficiencyChart(filteredData.energyVsEfficiency);
    renderCO2VsCostChart(filteredData.co2VsCost);
    renderTempPressureEfficiencyChart(filteredData.tempPressureEfficiency);
    renderTimelineChart(filteredData.timeline);
    
    // Process type specific charts remain the same
}

function renderSummaryCards() {
    if (!dashboardData.summary) return;
    
    document.getElementById('totalProcesses').textContent = dashboardData.summary.total_processes;
    document.getElementById('avgEfficiency').textContent = `${dashboardData.summary.avg_efficiency}%`;
    document.getElementById('totalEnergy').textContent = `${formatNumber(dashboardData.summary.total_energy)} kWh`;
    document.getElementById('safetyIncidents').textContent = dashboardData.summary.safety_incidents;
}

function renderProcessTypesChart() {
    if (!dashboardData.processTypes) return;
    
    const data = dashboardData.processTypes;
    
    const pieData = [{
        values: data.map(item => item.count),
        labels: data.map(item => item.process_type),
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
    }];
    
    const layout = {
        margin: { t: 0, b: 0, l: 0, r: 0 },
        showlegend: true,
        legend: { orientation: 'h', y: 0 }
    };
    
    Plotly.newPlot('processTypesChart', pieData, layout);
}

function renderEfficiencyByProcessChart() {
    if (!dashboardData.efficiencyByProcess) return;
    
    const data = dashboardData.efficiencyByProcess;
    
    const barData = [{
        x: data.map(item => item.process_type),
        y: data.map(item => item.avg_efficiency),
        type: 'bar',
        marker: {
            color: '#1f77b4'
        }
    }];
    
    const layout = {
        margin: { t: 10, b: 50, l: 50, r: 20 },
        yaxis: {
            title: 'Orta Səmərəlilik (%)'
        }
    };
    
    Plotly.newPlot('efficiencyByProcessChart', barData, layout);
}

function renderEnergyVsEfficiencyChart(filteredData = null) {
    const data = filteredData || dashboardData.energyVsEfficiency;
    if (!data) return;
    
    // Group by process type
    const processTypes = [...new Set(data.map(item => item.process_type))];
    
    const scatterData = processTypes.map(processType => {
        const processData = data.filter(item => item.process_type === processType);
        
        return {
            x: processData.map(item => item.energy_usage),
            y: processData.map(item => item.efficiency),
            mode: 'markers',
            type: 'scatter',
            name: processType,
            marker: {
                size: 10
            }
        };
    });
    
    const layout = {
        margin: { t: 10, b: 50, l: 50, r: 20 },
        xaxis: {
            title: 'Enerji İstifadəsi (kWh)'
        },
        yaxis: {
            title: 'Səmərəlilik (%)'
        },
        legend: {
            orientation: 'h',
            y: 1.1
        }
    };
    
    Plotly.newPlot('energyVsEfficiencyChart', scatterData, layout);
}

function renderCO2VsCostChart(filteredData = null) {
    const data = filteredData || dashboardData.co2VsCost;
    if (!data) return;
    
    // Group by process type
    const processTypes = [...new Set(data.map(item => item.process_type))];
    
    const scatterData = processTypes.map(processType => {
        const processData = data.filter(item => item.process_type === processType);
        
        return {
            x: processData.map(item => item.co2_per_ton),
            y: processData.map(item => item.cost_per_ton),
            mode: 'markers',
            type: 'scatter',
            name: processType,
            marker: {
                size: 10
            }
        };
    });
    
    const layout = {
        margin: { t: 10, b: 50, l: 50, r: 20 },
        xaxis: {
            title: 'CO₂ Emissiyası (ton başına qram)'
        },
        yaxis: {
            title: 'Əməliyyat Xərcləri (ton başına AZN)'
        },
        legend: {
            orientation: 'h',
            y: 1.1
        }
    };
    
    Plotly.newPlot('co2VsCostChart', scatterData, layout);
}

function renderCatalystEfficiencyChart() {
    if (!dashboardData.catalystEfficiency) return;
    
    const data = dashboardData.catalystEfficiency;
    
    const barData = [{
        x: data.map(item => item.avg_efficiency),
        y: data.map(item => item.catalyst),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: '#2ca02c'
        }
    }];
    
    const layout = {
        margin: { t: 10, b: 50, l: 250, r: 20 },
        xaxis: {
            title: 'Orta Səmərəlilik (%)'
        }
    };
    
    Plotly.newPlot('catalystEfficiencyChart', barData, layout);
}

function renderProcessDurationChart() {
    if (!dashboardData.processDuration) return;
    
    const data = dashboardData.processDuration;
    
    const barData = [{
        x: data.map(item => item.process_type),
        y: data.map(item => item.avg_duration),
        type: 'bar',
        marker: {
            color: '#ff7f0e'
        }
    }];
    
    const layout = {
        margin: { t: 10, b: 50, l: 50, r: 20 },
        yaxis: {
            title: 'Orta Müddət (saat)'
        }
    };
    
    Plotly.newPlot('processDurationChart', barData, layout);
}

function renderTempPressureEfficiencyChart(filteredData = null) {
    const data = filteredData || dashboardData.tempPressureEfficiency;
    if (!data) return;
    
    // Group by process type
    const processTypes = [...new Set(data.map(item => item.process_type))];
    
    const scatterData = processTypes.map(processType => {
        const processData = data.filter(item => item.process_type === processType);
        
        return {
            x: processData.map(item => item.temperature),
            y: processData.map(item => item.pressure),
            z: processData.map(item => item.efficiency),
            mode: 'markers',
            type: 'scatter3d',
            name: processType,
            marker: {
                size: 5,
                color: processData.map(item => item.efficiency),
                colorscale: 'Viridis',
                colorbar: {
                    title: 'Səmərəlilik (%)'
                }
            }
        };
    });
    
    const layout = {
        scene: {
            xaxis: { title: 'Temperatur (°C)' },
            yaxis: { title: 'Təzyiq (bar)' },
            zaxis: { title: 'Səmərəlilik (%)' }
        },
        margin: { l: 0, r: 0, b: 0, t: 10 }
    };
    
    Plotly.newPlot('tempPressureEfficiencyChart', scatterData, layout);
}

function renderTimelineChart(filteredData = null) {
    const data = filteredData || dashboardData.timeline;
    if (!data) return;
    
    // Group by process type
    const processTypes = [...new Set(data.map(item => item.process_type))];
    
    const traceData = processTypes.map(processType => {
        const processData = data.filter(item => item.process_type === processType);
        
        return {
            x: processData.map(item => [item.start_date, item.end_date]),
            y: processData.map(item => `${item.process_id}: ${processType}`),
            mode: 'lines',
            type: 'scatter',
            name: processType,
            line: { width: 15 }
        };
    });
    
    const layout = {
        margin: { t: 10, b: 50, l: 120, r: 20 },
        xaxis: {
            title: 'Tarix'
        },
        yaxis: {
            title: 'Proses ID',
            autorange: 'reversed'
        },
        legend: {
            orientation: 'h',
            y: 1.1
        }
    };
    
    Plotly.newPlot('timelineChart', traceData, layout);
}

// Helper functions
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function showError(message) {
    // Implementation for error display
    console.error(message);
}