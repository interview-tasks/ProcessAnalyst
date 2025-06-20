# Oil & Gas Processing Data Analysis Bot

A Telegram bot that delivers interactive data analysis, visualizations, and AI-powered insights for oil and gas processing data in Azerbaijani.

@prossi_bot


https://web.telegram.org/k/#@prossi_bot
## Overview

This application integrates Telegram's messaging platform with powerful data analysis capabilities and Google's Gemini AI to provide engineers and managers in the petroleum industry with quick access to critical operational metrics. The bot analyzes processing data to deliver insights on efficiency, energy consumption, environmental impact, and operational costs.

## Features

- **Interactive UI**: Simple keyboard-based interface for accessing different analytics
- **Comprehensive Data Analysis**: Detailed statistical summaries of your processing data
- **Visual Analytics**: Four different chart types for visualizing key metrics:
  - **Efficiency Analysis**: Boxplots showing process efficiency by process type
  - **Energy Usage**: Scatter plots relating processing volume to energy consumption
  - **Environmental Impact**: Bar charts of CO2 emissions by process type
  - **Cost Analysis**: Visualization of operational costs by process type
- **AI-Powered Insights**: Azerbaijani language analysis of the data using Google Gemini 1.5 Flash
- **Robust Error Handling**: Comprehensive logging and graceful error handling

## Technical Implementation

### Stack

- **Python 3.11+**
- **Flask**: Web application framework
- **pyTelegramBotAPI**: Framework for Telegram Bot API
- **Pandas & NumPy**: Data analysis libraries
- **Matplotlib, Seaborn & Plotly**: Data visualization
- **Google Gemini API**: Advanced natural language processing

### Architecture

The application uses a webhook-based architecture in production, where:

1. The Telegram API sends message updates to our webhook endpoint
2. The Flask application processes these updates
3. The application analyzes data and generates visualizations
4. Responses are sent back to users through the Telegram API

Data processing follows a pipeline pattern:
1. Data loading from CSV
2. Statistical analysis
3. Visualization generation or AI insights
4. Response formatting and delivery

## Deployment

### Prerequisites

- Python 3.11 or higher
- A Telegram Bot token from BotFather
- A Google Gemini API key
- A Render.com account (or another hosting service)

### Local Development Setup

1. Clone this repository

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token
   GEMINI_API_KEY=your_gemini_api_key
   APP_URL=http://localhost:5000  # For local development
   ENVIRONMENT=development
   ```

5. Place your `data.csv` file either in the root directory or in a `data/` subfolder

6. Run the application:
   ```bash
   python app.py
   ```

### Render.com Deployment

1. Push your code to GitHub

2. Create a new Web Service on Render.com, pointing to your repository

3. Configure the following Build settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -b 0.0.0.0:$PORT app:app`

4. Add the following environment variables:
   - `TELEGRAM_TOKEN`: Your Telegram bot token
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `APP_URL`: Your Render.com app URL (without trailing slash)
   - `ENVIRONMENT`: Set to `production`
   - `PORT`: Set to `10000`

5. Upload your `data.csv` file by either:
   - Including it in your repository
   - Adding it through the Render.com dashboard after deployment

6. After deployment, verify your webhook is properly set by visiting:
   ```
   https://api.telegram.org/bot{YOUR_TELEGRAM_TOKEN}/getWebhookInfo
   ```

## Data Format

The application expects a CSV file with the following columns:

- **Proses ID**: Process identifier
- **Proses Tipi**: Process type classification
- **Proses Addımı**: Process step
- **Emal Həcmi (ton)**: Processing volume in tons
- **Temperatur (°C)**: Process temperature
- **Təzyiq (bar)**: Pressure in bars
- **Prosesin Müddəti (saat)**: Process duration in hours
- **İstifadə Edilən Katalizatorlar**: Catalysts used
- **Emalın Səmərəliliyi (%)**: Processing efficiency percentage
- **Enerji İstifadəsi (kWh)**: Energy usage in kWh
- **Ətraf Mühitə Təsir (g CO2 ekvivalent)**: Environmental impact in CO2 equivalent
- **Təhlükəsizlik Hadisələri**: Safety incidents count
- **Emal Məhsulları**: Processing products
- **Əməliyyat Xərcləri (AZN)**: Operational costs in AZN
- **İstifadə Edilən Avadanlıq**: Equipment used
- **İşçi Sayı**: Number of workers
- **Prosesin Başlama Tarixi**: Process start date
- **Prosesin Bitmə Tarixi**: Process end date
- **Təchizatçı Adı**: Supplier name
- **Proses Qrupları**: Process groups
- **Energy_per_ton**: Energy usage per ton
- **CO2_per_ton**: CO2 emissions per ton
- **Cost_per_ton**: Cost per ton

## Troubleshooting

### Common Issues

1. **Bot not responding to commands**:
   - Check your webhook status using `getWebhookInfo`
   - Verify your app is running using the `/health` endpoint
   - Examine the app logs for errors

2. **Data loading issues**:
   - Use the `/test-data` endpoint to check data loading
   - Verify the file path and format
   - Check the server logs for specific errors

3. **Chart generation fails**:
   - May indicate issues with the data format
   - Check if all required columns are present
   - Look for any data type issues in your CSV

4. **Gemini analysis fails**:
   - Verify your Gemini API key is valid
   - Check for any API rate limits or quota issues
   - Examine the specific error message in the logs

### Useful Endpoints

- **/**:  Basic health check
- **/health**: Confirms the bot is running
- **/test-data**: Tests if the data.csv file can be loaded successfully

## Usage

1. Start a chat with your bot on Telegram
2. Send the `/start` command to get the main menu
3. Select from the available options:
   - "Əsas Məlumatlar" (Basic Information): Shows a summary of key metrics
   - "Səmərəlilik Analizi" (Efficiency Analysis): Displays efficiency statistics
   - "Enerji İstifadəsi" (Energy Usage): Shows energy consumption patterns
   - "Ətraf Mühit Təsiri" (Environmental Impact): Displays CO2 emissions analysis
   - "Xərc Analizi" (Cost Analysis): Shows operational costs breakdown
   - "Gemini Təhlili" (Gemini Analysis): Provides AI-generated insights in Azerbaijani

## License

This project is proprietary and confidential. Unauthorized copying, distribution or use is strictly prohibited.
