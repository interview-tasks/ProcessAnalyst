# Telegram Oil & Gas Process Data Reporter

A Telegram bot that analyzes oil and gas processing data and provides insights in Azerbaijani using OpenAI.

## Features

- Automatic data loading and analysis from CSV
- Interactive Telegram bot with menu buttons
- Multiple visualizations for different metrics:
  - Process efficiency analysis
  - Energy consumption patterns
  - Environmental impact (CO2 emissions)
  - Cost analysis
- AI-powered insights in Azerbaijani using OpenAI API
- Ready for deployment on Render.com

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- A Telegram Bot (created via BotFather)
- OpenAI API key

### Local Development

1. Clone this repository

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token
   OPENAI_API_KEY=your_openai_api_key
   ENVIRONMENT=development
   ```

5. Run the application:
   ```
   python app.py
   ```

### Deployment to Render.com

1. Sign up for a Render.com account

2. Connect your GitHub repository to Render.com

3. Create a new Web Service with the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`

4. Add the following environment variables in Render.com dashboard:
   - `TELEGRAM_TOKEN`: Your Telegram bot token
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `ENVIRONMENT`: Set to `production`
   - `APP_URL`: Your application URL (e.g., https://your-app-name.onrender.com)

5. Deploy the service

## Usage

1. Start a chat with your Telegram bot

2. Use the provided menu buttons to:
   - View summary statistics
   - Analyze process efficiency
   - Explore energy consumption patterns
   - Check environmental impact
   - Review cost analysis
   - Get AI-powered insights in Azerbaijani

## Data Format

The application expects a CSV file named `data.csv` with the following columns:
- Proses ID
- Proses Tipi
- Proses Addımı
- Emal Həcmi (ton)
- Temperatur (°C)
- Təzyiq (bar)
- Prosesin Müddəti (saat)
- İstifadə Edilən Katalizatorlar
- Emalın Səmərəliliyi (%)
- Enerji İstifadəsi (kWh)
- Ətraf Mühitə Təsir (g CO2 ekvivalent)
- Təhlükəsizlik Hadisələri
- Emal Məhsulları
- Əməliyyat Xərcləri (AZN)
- İstifadə Edilən Avadanlıq
- İşçi Sayı
- Prosesin Başlama Tarixi
- Prosesin Bitmə Tarixi
- Təchizatçı Adı
- Proses Qrupları
- Energy_per_ton
- CO2_per_ton
- Cost_per_ton

## Security Note

The provided OpenAI API key in your request (sk-None-oXPtwwbJnBlFKWNrUz8TT3BlbkFJlUxme5uI8uwY7dMjDmzu) appears to be invalid or a placeholder. Make sure to use a valid API key in your .env file when deploying the application.