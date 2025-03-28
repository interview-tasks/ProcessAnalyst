import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import telebot
from telebot import types
import openai
from io import BytesIO
import plotly.express as px
import plotly.io as pio
from dotenv import load_dotenv
from flask import Flask, request
import time

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)

# Initialize OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Initialize Telegram bot
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Load the data
def load_data():
    try:
        data = pd.read_csv('data.csv')
        logger.info(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

# Function to generate insights using OpenAI in Azerbaijani
def generate_insights(data_description):
    try:
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sən dəqiq və analitik neft və qaz emalı prosesləri üzrə məlumatlar haqqında Azərbaycan dilində təhlil təqdim edən köməkçisən."},
                {"role": "user", "content": f"Aşağıdakı məlumatları təhlil et və biznes üçün əhəmiyyətli nəticələri Azərbaycan dilində təqdim et: {data_description}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating insights with OpenAI: {e}")
        return "OpenAI ilə təhlil yaradılarkən xəta baş verdi."

# Function to create charts
def create_efficiency_chart(data):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Proses Tipi', y='Emalın Səmərəliliyi (%)', data=data)
    plt.title('Proses Tipinə görə Emal Səmərəliliyi', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

def create_energy_chart(data):
    plt.figure(figsize=(12, 8))
    fig = px.scatter(data, 
                    x='Emal Həcmi (ton)', 
                    y='Enerji İstifadəsi (kWh)',
                    color='Proses Tipi',
                    size='Energy_per_ton',
                    hover_data=['Proses ID', 'Təzyiq (bar)', 'Temperatur (°C)'])
    fig.update_layout(title='Emal Həcmi və Enerji İstifadəsi Arasında Əlaqə')
    
    buffer = BytesIO()
    pio.write_image(fig, buffer, format='png')
    buffer.seek(0)
    return buffer

def create_environmental_chart(data):
    plt.figure(figsize=(12, 8))
    avg_co2_by_type = data.groupby('Proses Tipi')['Ətraf Mühitə Təsir (g CO2 ekvivalent)'].mean().reset_index()
    avg_co2_by_type = avg_co2_by_type.sort_values('Ətraf Mühitə Təsir (g CO2 ekvivalent)', ascending=False)
    
    plt.barh(avg_co2_by_type['Proses Tipi'], avg_co2_by_type['Ətraf Mühitə Təsir (g CO2 ekvivalent)'])
    plt.title('Proses Tipinə görə Ortalama CO2 Emissiyası', fontsize=16)
    plt.xlabel('CO2 Emissiyası (g)', fontsize=12)
    plt.ylabel('Proses Tipi', fontsize=12)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

def create_cost_chart(data):
    plt.figure(figsize=(12, 8))
    cost_by_type = data.groupby('Proses Tipi')['Əməliyyat Xərcləri (AZN)'].sum().reset_index()
    cost_by_type = cost_by_type.sort_values('Əməliyyat Xərcləri (AZN)', ascending=False)
    
    plt.bar(cost_by_type['Proses Tipi'], cost_by_type['Əməliyyat Xərcləri (AZN)'] / 1000)
    plt.title('Proses Tipinə görə Ümumi Əməliyyat Xərcləri', fontsize=16)
    plt.xlabel('Proses Tipi', fontsize=12)
    plt.ylabel('Əməliyyat Xərcləri (Min AZN)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer

# Generate a summary of the data
def generate_data_summary(data):
    summary = {
        'total_processes': data.shape[0],
        'process_types': data['Proses Tipi'].nunique(),
        'avg_efficiency': data['Emalın Səmərəliliyi (%)'].mean(),
        'total_energy': data['Enerji İstifadəsi (kWh)'].sum(),
        'total_cost': data['Əməliyyat Xərcləri (AZN)'].sum(),
        'avg_co2': data['Ətraf Mühitə Təsir (g CO2 ekvivalent)'].mean(),
        'max_volume': data['Emal Həcmi (ton)'].max(),
        'safety_incidents': data['Təhlükəsizlik Hadisələri'].sum()
    }
    
    # Top and bottom performers
    top_efficiency = data.sort_values('Emalın Səmərəliliyi (%)', ascending=False).head(3)
    low_efficiency = data.sort_values('Emalın Səmərəliliyi (%)', ascending=True).head(3)
    
    # Most efficient process type
    process_efficiency = data.groupby('Proses Tipi')['Emalın Səmərəliliyi (%)'].mean().reset_index()
    best_process = process_efficiency.loc[process_efficiency['Emalın Səmərəliliyi (%)'].idxmax()]
    
    summary_text = f"""
Ümumi Məlumat Təhlili:
- Ümumi proses sayı: {summary['total_processes']}
- Fərqli proses tipləri: {summary['process_types']}
- Ortalama emal səmərəliliyi: {summary['avg_efficiency']:.2f}%
- Ümumi enerji istifadəsi: {summary['total_energy']:,} kWh
- Ümumi əməliyyat xərcləri: {summary['total_cost']:,} AZN
- Ortalama CO2 emissiyası: {summary['avg_co2']:,.2f} g
- Maksimum emal həcmi: {summary['max_volume']:,} ton
- Qeydə alınmış təhlükəsizlik hadisələri: {summary['safety_incidents']}

Ən Yüksək Səmərəliliyə Malik Proseslər:
"""
    
    for i, row in top_efficiency.iterrows():
        summary_text += f"- Proses ID: {row['Proses ID']}, Tipi: {row['Proses Tipi']}, Səmərəlilik: {row['Emalın Səmərəliliyi (%)']}%\n"
    
    summary_text += f"\nƏn Aşağı Səmərəliliyə Malik Proseslər:\n"
    
    for i, row in low_efficiency.iterrows():
        summary_text += f"- Proses ID: {row['Proses ID']}, Tipi: {row['Proses Tipi']}, Səmərəlilik: {row['Emalın Səmərəliliyi (%)']}%\n"
    
    summary_text += f"\nƏn Səmərəli Proses Tipi: {best_process['Proses Tipi']} (Ortalama {best_process['Emalın Səmərəliliyi (%)']}%)"
    
    return summary_text

# Bot command handlers
@bot.message_handler(commands=['start'])
def send_welcome(message):
    logger.info(f"Received /start command from user {message.from_user.id}")
    try:
        markup = types.ReplyKeyboardMarkup(row_width=2)
        item1 = types.KeyboardButton('Əsas Məlumatlar')
        item2 = types.KeyboardButton('Səmərəlilik Analizi')
        item3 = types.KeyboardButton('Enerji İstifadəsi')
        item4 = types.KeyboardButton('Ətraf Mühit Təsiri')
        item5 = types.KeyboardButton('Xərc Analizi')
        item6 = types.KeyboardButton('OpenAI Təhlili')
        
        markup.add(item1, item2, item3, item4, item5, item6)
        logger.info("Sending welcome message with keyboard markup")
        bot.reply_to(message, "Xoş gəlmisiniz! Lütfən, aşağıdakı seçimlərdən birini seçin:", reply_markup=markup)
    except Exception as e:
        logger.error(f"Error in send_welcome: {e}")
        bot.reply_to(message, "Xəta baş verdi. Zəhmət olmasa bir az sonra yenidən cəhd edin.")
        
@bot.message_handler(func=lambda message: message.text == 'Əsas Məlumatlar')
def send_summary(message):
    data = load_data()
    if data is not None:
        summary = generate_data_summary(data)
        bot.send_message(message.chat.id, summary)
    else:
        bot.send_message(message.chat.id, "Məlumatların yüklənməsində xəta baş verdi.")

@bot.message_handler(func=lambda message: message.text == 'Səmərəlilik Analizi')
def send_efficiency_chart(message):
    data = load_data()
    if data is not None:
        chart = create_efficiency_chart(data)
        bot.send_photo(message.chat.id, chart, caption="Proses Tipinə görə Emal Səmərəliliyi")
    else:
        bot.send_message(message.chat.id, "Məlumatların yüklənməsində xəta baş verdi.")

@bot.message_handler(func=lambda message: message.text == 'Enerji İstifadəsi')
def send_energy_chart(message):
    data = load_data()
    if data is not None:
        chart = create_energy_chart(data)
        bot.send_photo(message.chat.id, chart, caption="Emal Həcmi və Enerji İstifadəsi Arasında Əlaqə")
    else:
        bot.send_message(message.chat.id, "Məlumatların yüklənməsində xəta baş verdi.")

@bot.message_handler(func=lambda message: message.text == 'Ətraf Mühit Təsiri')
def send_environmental_chart(message):
    data = load_data()
    if data is not None:
        chart = create_environmental_chart(data)
        bot.send_photo(message.chat.id, chart, caption="Proses Tipinə görə Ortalama CO2 Emissiyası")
    else:
        bot.send_message(message.chat.id, "Məlumatların yüklənməsində xəta baş verdi.")

@bot.message_handler(func=lambda message: message.text == 'Xərc Analizi')
def send_cost_chart(message):
    data = load_data()
    if data is not None:
        chart = create_cost_chart(data)
        bot.send_photo(message.chat.id, chart, caption="Proses Tipinə görə Ümumi Əməliyyat Xərcləri")
    else:
        bot.send_message(message.chat.id, "Məlumatların yüklənməsində xəta baş verdi.")

@bot.message_handler(func=lambda message: message.text == 'OpenAI Təhlili')
def send_ai_insights(message):
    data = load_data()
    if data is not None:
        # Generate data description for OpenAI
        summary = generate_data_summary(data)
        
        bot.send_message(message.chat.id, "OpenAI təhlili hazırlanır, xahiş edirik gözləyin...")
        
        insights = generate_insights(summary)
        
        # Split message if it's too long
        if len(insights) > 4000:
            chunks = [insights[i:i+4000] for i in range(0, len(insights), 4000)]
            for chunk in chunks:
                bot.send_message(message.chat.id, chunk)
        else:
            bot.send_message(message.chat.id, insights)
    else:
        bot.send_message(message.chat.id, "Məlumatların yüklənməsində xəta baş verdi.")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Xahiş edirik mövcud olan düymələrdən birini seçin.")

# Flask route to handle webhook
@app.route(f'/{TELEGRAM_TOKEN}', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_str = request.get_data().decode('UTF-8')
        update = types.Update.de_json(json_str)
        bot.process_new_updates([update])
        return ''
    return '', 403

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return 'Bot is running!'

@app.route('/')
def index():
    return 'Telegram Bot is running!'

if __name__ == '__main__':
    # Get the port from environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))
    
    # For production, use webhook mode
    if os.environ.get('ENVIRONMENT') == 'production':
        # Remove any existing webhook first
        bot.remove_webhook()
        time.sleep(0.5)  # Give Telegram servers some time to process
        
        # Set webhook
        url = os.environ.get('APP_URL', '')
        if url:
            webhook_url = f"{url}/{TELEGRAM_TOKEN}"
            bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook set to {webhook_url}")
        else:
            logger.error("APP_URL environment variable not set")
        
        # Only run the Flask app (no polling) in production
        app.run(host='0.0.0.0', port=port)
    else:
        # For development, just use polling mode
        bot.remove_webhook()
        logger.info("Starting bot in polling mode for development")
        bot.infinity_polling()