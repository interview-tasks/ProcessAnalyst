# Process Analyst Dashboard

## Overview

The Process Analyst Dashboard is an interactive web application built with Flask backend and HTML/CSS/JavaScript frontend. It visualizes oil and gas processing data, providing insights into process efficiency, energy usage, environmental impact, and catalyst performance.

![Dashboard Preview](https://via.placeholder.com/800x450.png?text=Process+Analyst+Dashboard)

## Features

- **Interactive Visualizations**: Multiple charts and graphs for different process metrics
- **Data Filtering**: Filter data by process type and other parameters
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: API-driven data serving for interactive visualization
- **Multi-section Analysis**: Dedicated views for efficiency, energy, environmental impact, etc.

## Directory Structure

```
dashboard/
├── app.py                # Flask application
├── data/
│   └── data.csv          # Process data
├── requirements.txt      # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css     # Dashboard styling
│   └── js/
│       └── main.js       # Dashboard interactivity
└── templates/
    └── index.html        # Dashboard HTML template
```

## Installation

### Local Development

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Visit `http://127.0.0.1:5000` in your browser

## API Endpoints

The dashboard provides the following API endpoints:

| Endpoint | Description |
|----------|-------------|
| `/api/data/summary` | Summary statistics of all process data |
| `/api/data/process_types` | Distribution of process types |
| `/api/data/efficiency_by_process` | Average efficiency by process type |
| `/api/data/energy_vs_efficiency` | Energy usage vs efficiency data |
| `/api/data/co2_vs_cost` | CO2 emissions vs operational costs |
| `/api/data/catalyst_efficiency` | Average efficiency by catalyst type |
| `/api/data/process_duration` | Average process duration by type |
| `/api/data/efficiency_by_temp_pressure` | Efficiency by temperature and pressure |
| `/api/data/timeline` | Process timeline data |

## Dashboard Sections

### 1. Overview
Provides summary metrics and high-level process distribution charts.

### 2. Efficiency Analysis
Visualizes the relationship between energy usage and process efficiency, as well as parameter influences.

### 3. Energy Usage
Analyzes energy consumption patterns and process duration by type.

### 4. Environmental Impact
Examines CO2 emissions, operational costs, and environmental performance.

### 5. Catalyst Analysis
Compares the performance of different catalysts used in processes.

### 6. Timeline
Displays a chronological view of recent processes with duration and type.

## Deployment to Render

### Prerequisites
- A [Render](https://render.com) account
- Git repository with your dashboard code

### Deployment Steps

1. Push your code to a Git repository (GitHub, GitLab, etc.).

2. Log in to your Render account.

3. Create a new Web Service:
   - Click "New" and select "Web Service"
   - Connect your Git repository
   - Select the branch to deploy

4. Configure the Web Service:
   - Name: `process-analyst`
   - Environment: `Python 3`
   - Build Command: `pip install -r dashboard/requirements.txt`
   - Start Command: `cd dashboard && gunicorn app:app`
   - Select appropriate instance type

5. Set Environment Variables:
   - Click "Environment" tab
   - Add `PYTHON_VERSION: 3.9.0` (or your preferred version)
   - Add `FLASK_ENV: production`

6. Deploy the service:
   - Click "Create Web Service"
   - Wait for the build and deployment to complete

7. Access your deployed dashboard at the provided Render URL

### Important Render Deployment Notes

- Ensure your `requirements.txt` includes `gunicorn` for production deployment
- Make sure static file paths use Flask's `url_for` to work correctly in production
- Set up appropriate error handling for production environment

## Customization

### Adding New Charts

1. Add a new API endpoint in `app.py`
2. Create a corresponding rendering function in `main.js`
3. Add HTML container in `index.html`
4. Call the rendering function in the dashboard initialization

### Data Updates

The dashboard currently uses static data from `data.csv`. To implement real-time data updating:

1. Set up a database connection in `app.py`
2. Modify API endpoints to query the database
3. Implement data refresh in `main.js`

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Frontend**: 
  - HTML5/CSS3/JavaScript
  - Bootstrap 5 (responsive layout)
  - Plotly.js (interactive charts)
  - jQuery (DOM manipulation)
- **Data Processing**: Pandas (Python data analysis library)

## Dependencies

All required Python packages are listed in `requirements.txt`. The main dependencies are:

- Flask (web framework)
- Pandas (data processing)
- Gunicorn (WSGI HTTP Server for production)

Frontend dependencies are loaded via CDN:
- Bootstrap 5
- Plotly.js
- Font Awesome
- jQuery

## Troubleshooting

### Common Issues

1. **Data Not Loading**
   - Check that `data.csv` is in the correct location
   - Verify CSV format is consistent with expected columns

2. **Charts Not Rendering**
   - Check browser console for JavaScript errors
   - Verify that Plotly.js is loaded correctly

3. **Render Deployment Issues**
   - Ensure `gunicorn` is in your requirements.txt
   - Check build logs for any installation errors
   - Verify that your start command includes the correct path

4. **Performance Issues**
   - Consider implementing data pagination for large datasets
   - Use data aggregation for complex visualizations

## Contributing

To contribute to this dashboard:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Create a pull request

## License

This project is proprietary to and is not licensed for public use or distribution.

## Contact

For any questions or support, please contact the Process Analysis team.