# Process Analyst

![Process Analyst Logo](https://media.istockphoto.com/id/925690772/vector/icon-icon-with-the-concept-of-searching-analyzing-for-business-finance-and-marketing.jpg?s=612x612&w=0&k=20&c=eYqm89QSN6MWpXRICxTCCQdX6o-mg-Lu9rZLWK6WNgM=)

## Project Overview

Process Analyst is a comprehensive data analytics and visualization solution for oil and gas processing operations. This platform provides three different interfaces to access and leverage process data insights:

1. **GitHub Repository** - Access to source code, analysis notebooks, and development resources
2. **Web Dashboard** - Interactive visualization and analysis through a browser interface
3. **Telegram Bot** - Quick insights and alerts through convenient messaging

Each interface serves different user needs while accessing the same underlying analytics engine.

## Core Features

- **Process Efficiency Analysis**: Identify optimal operating parameters to maximize output
- **Energy Consumption Optimization**: Track and reduce energy usage across operations
- **Environmental Impact Assessment**: Monitor and minimize CO2 emissions
- **Catalyst Performance Evaluation**: Compare and optimize catalyst usage
- **Interactive Visualizations**: Filter and explore process data dynamically
- **Safety Incident Tracking**: Monitor and improve safety performance

## Data Insights

The platform analyzes process data with 23 key fields including:

- Process types and steps
- Operating parameters (temperature, pressure, duration)
- Resource metrics (energy, catalysts, worker count)
- Efficiency metrics (processing efficiency, energy per ton)
- Environmental impact (CO2 emissions)
- Economic indicators (operational costs, cost per ton)

## Three Ways to Access Process Analyst

### 1. GitHub Repository
**URL**: [https://github.com/Ismat-Samadov/ProcessAnalyst](https://github.com/Ismat-Samadov/ProcessAnalyst)

The GitHub repository provides:

- **Source Code Access**: View and download all project components
- **Jupyter Notebooks**: Run detailed analysis scripts locally
- **Documentation**: Comprehensive project documentation
- **Development Resources**: Contribute to or extend the platform

#### Repository Structure

```
ProcessAnalyst/
├── README.md                      # Main project README
├── analysis/                      # Data analysis component
│   ├── README.md                  # Analysis module documentation
│   ├── analyse.ipynb              # Jupyter notebook for analysis
│   └── data/                      # Data directory
│       ├── charts/                # Generated chart images
│       ├── data.csv               # Process data (CSV format)
│       └── data.xlsx              # Process data (Excel format)
├── dashboard/                     # Interactive dashboard component
│   ├── README.md                  # Dashboard documentation
│   ├── app.py                     # Flask application
│   ├── data/                      # Dashboard data
│   │   └── data.csv               # Process data for dashboard
│   ├── requirements.txt           # Python dependencies
│   ├── static/                    # Static assets
│   │   ├── css/                   # Stylesheets
│   │   │   └── style.css          # Dashboard styling
│   │   └── js/                    # JavaScript files
│   │       └── main.js            # Dashboard interactivity
│   └── templates/                 # HTML templates
│       └── index.html             # Dashboard HTML template
└── telegram/                      # Telegram bot component
    ├── README.md                  # Bot documentation
    ├── app.py                     # Bot application
    ├── data/                      # Bot data
    │   └── data.csv               # Process data for bot
    └── requirements.txt           # Python dependencies
```

#### Getting Started with the Repository

1. Clone the repository:
   ```bash
   git clone https://github.com/Ismat-Samadov/ProcessAnalyst.git
   cd ProcessAnalyst
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

4. Run the analysis notebooks:
   ```bash
   cd analysis
   jupyter notebook analyse.ipynb
   ```

5. Launch the dashboard locally:
   ```bash
   cd dashboard
   python app.py
   ```

### 2. Web Dashboard
**URL**: [https://processanalyst.onrender.com/](https://processanalyst.onrender.com/)

The web dashboard provides:

- **Interactive Visualizations**: Explore data through dynamic charts and graphs
- **Filtering Capabilities**: Drill down into specific processes, timeframes, or parameters
- **Performance Metrics**: Track KPIs and benchmarks in real-time
- **Mobile-Responsive Design**: Access insights from any device

#### Dashboard Features

- **Overview Panel**: High-level metrics and KPIs
- **Process Analysis**: Detailed breakdowns by process type and step
- **Efficiency Tracker**: Monitor energy, environmental, and cost efficiency
- **Catalyst Comparison**: Compare performance across different catalysts
- **Custom Reports**: Generate tailored reports for specific needs

#### Using the Dashboard

1. Navigate to [https://processanalyst.onrender.com/](https://processanalyst.onrender.com/)
2. Use the process type filter to focus on specific process categories
3. Navigate between different sections using the sidebar menu
4. Interact with charts to drill down into data
5. View detailed analyses for efficiency, energy, environmental impact, and catalysts

### 3. Telegram Bot
**URL**: [https://web.telegram.org/k/#@analyst_bot](https://web.telegram.org/k/#@analyst_bot)

The Telegram bot provides:

- **Quick Insights**: Get key metrics and stats on demand
- **Automated Alerts**: Receive notifications about process anomalies
- **Data Visualizations**: View charts and graphs directly in Telegram
- **Convenient Access**: Use from any device with Telegram installed

#### Bot Commands

- `/start` - Begin interaction with the bot (**REQUIRED FIRST STEP**)
- `/help` - View available commands and instructions
- `Əsas Məlumatlar` - Get basic information about process data
- `Səmərəlilik Analizi` - View efficiency analysis
- `Enerji İstifadəsi` - See energy usage patterns
- `Ətraf Mühit Təsiri` - View environmental impact analysis
- `Xərc Analizi` - Get cost analysis information
- `OpenAI Təhlili` - Generate AI-powered insights

#### Using the Telegram Bot

1. Open Telegram and search for `@analyst_bot`
2. **IMPORTANT**: You MUST send the `/start` command to initialize the bot
3. After sending `/start`, a keyboard menu will appear
4. Use the keyboard menu to request specific analyses
5. Receive visual charts and data summaries directly in your chat
6. Get AI-powered insights through natural language processing

## Technologies Used

- **Data Analysis**: Python, Pandas, NumPy, SciPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly.js
- **Web Dashboard**: Flask, Bootstrap, HTML/CSS/JavaScript
- **Telegram Bot**: pyTelegramBotAPI, OpenAI API integration
- **Deployment**: Render (cloud platform)

## Project Roadmap

### Current Capabilities
- Statistical analysis of process parameters
- Interactive data visualization
- Performance metric tracking
- Basic reporting and alerts

### Upcoming Features
- Machine learning models for process optimization
- Predictive maintenance algorithms
- Anomaly detection for quality control
- Real-time optimization recommendations
- Advanced integration with production systems

## Prerequisites and Requirements

- **For Repository Use**: 
  - Python 3.9+
  - Git for version control
  - Dependencies listed in requirements.txt
  
- **For Dashboard Access**:
  - Modern web browser (Chrome, Firefox, Safari, Edge)
  - Internet connection
  
- **For Telegram Bot**:
  - Telegram account
  - Mobile device or desktop Telegram client

## Support

Process Analyst is developed and maintained by Ismat Samadov, combining expertise in process engineering, data science, and software development.

For support or inquiries:
- **Technical Issues**: File an issue on the GitHub repository
- **Dashboard Feedback**: Use the feedback form on the dashboard
- **Bot Assistance**: Use the `/help` command in Telegram
