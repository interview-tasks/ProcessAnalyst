# SOCAR Process Analyst

![SOCAR Process Analyst Logo](https://via.placeholder.com/800x200.png?text=SOCAR+Process+Analyst)

## Project Overview

SOCAR Process Analyst is a comprehensive data analytics and visualization solution for oil and gas processing operations. The project consists of two main components:

1. **Data Analysis Module** - Jupyter notebooks for in-depth data exploration and statistical analysis
2. **Interactive Dashboard** - A web-based dashboard for real-time monitoring and visualization

This solution helps oil and gas processing facilities optimize their operations by providing insights into process efficiency, energy consumption, environmental impact, and catalyst performance.

## Project Structure

```
SOCAR_ProcessAnalyst/
├── README.md                      # Main project README
├── analysis/                      # Data analysis component
│   ├── README.md                  # Analysis module documentation
│   ├── analyse.ipynb              # Jupyter notebook for analysis
│   └── data/                      # Data directory
│       ├── charts/                # Generated chart images
│       ├── data.csv               # Process data (CSV format)
│       └── data.xlsx              # Process data (Excel format)
└── dashboard/                     # Interactive dashboard component
    ├── README.md                  # Dashboard documentation
    ├── app.py                     # Flask application
    ├── data/                      # Dashboard data
    │   └── data.csv               # Process data copy for dashboard
    ├── requirements.txt           # Python dependencies
    ├── static/                    # Static assets
    │   ├── css/                   # Stylesheets
    │   │   └── style.css          # Dashboard styling
    │   └── js/                    # JavaScript files
    │       └── main.js            # Dashboard interactivity
    └── templates/                 # HTML templates
        └── index.html             # Dashboard HTML template
```

## Key Features

- **Process Efficiency Analysis**: Identify optimal operating parameters
- **Energy Consumption Optimization**: Track and reduce energy usage
- **Environmental Impact Assessment**: Monitor and minimize emissions
- **Catalyst Performance Evaluation**: Compare and optimize catalyst usage
- **Interactive Visualizations**: Filter and explore process data dynamically
- **Safety Incident Tracking**: Monitor and improve safety performance

## Getting Started

### Prerequisites

- Python 3.9+
- Pipenv or Virtualenv for dependency management
- Git for version control

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SOCAR_ProcessAnalyst.git
   cd SOCAR_ProcessAnalyst
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Components

### 1. Analysis Module

The analysis module provides in-depth statistical analysis of process data using Jupyter notebooks. See [analysis/README.md](analysis/README.md) for detailed documentation.

Features:
- Statistical analysis of process parameters
- Correlation studies between variables
- Efficiency optimization models
- Predictive analytics for process outcomes
- Chart generation for reporting

### 2. Interactive Dashboard

The dashboard provides a web-based interface for exploring process data with interactive visualizations. See [dashboard/README.md](dashboard/README.md) for detailed documentation.

Features:
- Real-time visualization of process metrics
- Interactive filtering and data exploration
- Multi-dimensional data views
- KPI tracking and performance monitoring
- Mobile-responsive design

## Deployment

This project can be deployed to various platforms:

### Local Development
Follow the installation instructions above for local development.

### Production Deployment
The dashboard component is designed to be deployed to [Render](https://render.com), a cloud hosting service. See [dashboard/README.md](dashboard/README.md) for detailed deployment instructions.

## Data Structure

The project uses CSV data with the following key fields:

- Process ID
- Process Type
- Process Step
- Processing Volume (tons)
- Temperature (°C)
- Pressure (bar)
- Process Duration (hours)
- Catalysts Used
- Processing Efficiency (%)
- Energy Usage (kWh)
- Environmental Impact (g CO2 equivalent)
- Safety Incidents
- Processing Products
- Operational Costs (AZN)
- Equipment Used
- Worker Count
- Process Start Date
- Process End Date
- Supplier Name
- Process Groups
- Energy per ton
- CO2 per ton
- Cost per ton

## Development Roadmap

### Phase 1: Data Analysis and Visualization (Completed)
- Initial data processing and cleaning
- Basic statistical analysis
- Chart generation
- Dashboard prototype

### Phase 2: Interactive Dashboard (Current)
- Flask backend implementation
- Interactive data visualizations
- User interface development
- Filtering and data exploration

### Phase 3: Predictive Analytics (Planned)
- Machine learning models for process optimization
- Predictive maintenance algorithms
- Anomaly detection for quality control
- Real-time optimization recommendations

### Phase 4: Integration and Deployment (Planned)
- Integration with production systems
- Real-time data processing pipeline
- Automated reporting and alerts
- Mobile application development

## Technologies Used

- **Backend**: Python, Flask, Pandas, NumPy, SciPy
- **Frontend**: HTML/CSS/JavaScript, Bootstrap, Plotly.js
- **Data Analysis**: Jupyter Notebooks, Pandas, Matplotlib, Seaborn
- **Deployment**: Render (Cloud Platform)

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is proprietary to SOCAR and is not licensed for public use or distribution.

## Contact

For questions or support related to this project, please contact the SOCAR Process Analysis team.