from flask import Flask, render_template, jsonify
import pandas as pd
import json
import os

app = Flask(__name__)

# Load the data
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
df = pd.read_csv(DATA_PATH)

# Convert dates to datetime
df['Prosesin Başlama Tarixi'] = pd.to_datetime(df['Prosesin Başlama Tarixi'])
df['Prosesin Bitmə Tarixi'] = pd.to_datetime(df['Prosesin Bitmə Tarixi'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data/summary', methods=['GET'])
def get_summary():
    """Return summary statistics for the dashboard"""
    summary = {
        'total_processes': len(df),
        'avg_efficiency': round(df['Emalın Səmərəliliyi (%)'].mean(), 2),
        'total_energy': int(df['Enerji İstifadəsi (kWh)'].sum()),
        'total_cost': int(df['Əməliyyat Xərcləri (AZN)'].sum()),
        'avg_co2': round(df['CO2_per_ton'].mean(), 2),
        'process_types': df['Proses Tipi'].value_counts().to_dict(),
        'safety_incidents': int(df['Təhlükəsizlik Hadisələri'].sum())
    }
    return jsonify(summary)

@app.route('/api/data/process_types', methods=['GET'])
def get_process_types():
    """Return counts of each process type"""
    counts = df['Proses Tipi'].value_counts().reset_index()
    counts.columns = ['process_type', 'count']
    return jsonify(counts.to_dict(orient='records'))

@app.route('/api/data/efficiency_by_process', methods=['GET'])
def get_efficiency_by_process():
    """Return average efficiency by process type"""
    efficiency = df.groupby('Proses Tipi')['Emalın Səmərəliliyi (%)'].mean().reset_index()
    efficiency.columns = ['process_type', 'avg_efficiency']
    return jsonify(efficiency.to_dict(orient='records'))

@app.route('/api/data/energy_vs_efficiency', methods=['GET'])
def get_energy_vs_efficiency():
    """Return energy usage vs efficiency data for scatter plot"""
    data = df[['Enerji İstifadəsi (kWh)', 'Emalın Səmərəliliyi (%)', 'Proses Tipi']].copy()
    data.columns = ['energy_usage', 'efficiency', 'process_type']
    
    # Convert to records for JSON serialization
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/data/co2_vs_cost', methods=['GET'])
def get_co2_vs_cost():
    """Return CO2 emissions vs operational cost"""
    data = df[['CO2_per_ton', 'Cost_per_ton', 'Proses Tipi']].copy()
    data.columns = ['co2_per_ton', 'cost_per_ton', 'process_type']
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/data/catalyst_efficiency', methods=['GET'])
def get_catalyst_efficiency():
    """Return average efficiency by catalyst type"""
    catalyst_data = df.groupby('İstifadə Edilən Katalizatorlar')['Emalın Səmərəliliyi (%)'].mean().reset_index()
    catalyst_data = catalyst_data.sort_values('Emalın Səmərəliliyi (%)', ascending=False)
    catalyst_data.columns = ['catalyst', 'avg_efficiency']
    
    # Take top 10 for readability
    return jsonify(catalyst_data.head(10).to_dict(orient='records'))

@app.route('/api/data/process_duration', methods=['GET'])
def get_process_duration():
    """Return average process duration by process type"""
    duration = df.groupby('Proses Tipi')['Prosesin Müddəti (saat)'].mean().reset_index()
    duration.columns = ['process_type', 'avg_duration']
    return jsonify(duration.to_dict(orient='records'))

@app.route('/api/data/efficiency_by_temp_pressure', methods=['GET'])
def get_efficiency_by_temp_pressure():
    """Return efficiency data by temperature and pressure"""
    data = df[['Temperatur (°C)', 'Təzyiq (bar)', 'Emalın Səmərəliliyi (%)', 'Proses Tipi']].copy()
    data.columns = ['temperature', 'pressure', 'efficiency', 'process_type']
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/data/timeline', methods=['GET'])
def get_timeline():
    """Return process timeline data"""
    timeline = df[['Proses ID', 'Proses Tipi', 'Prosesin Başlama Tarixi', 'Prosesin Bitmə Tarixi', 'Emalın Səmərəliliyi (%)']].copy()
    
    # Convert dates to string for JSON serialization
    timeline['Prosesin Başlama Tarixi'] = timeline['Prosesin Başlama Tarixi'].dt.strftime('%Y-%m-%d')
    timeline['Prosesin Bitmə Tarixi'] = timeline['Prosesin Bitmə Tarixi'].dt.strftime('%Y-%m-%d')
    
    timeline.columns = ['process_id', 'process_type', 'start_date', 'end_date', 'efficiency']
    
    # Only return most recent 50 processes for performance
    return jsonify(timeline.tail(50).to_dict(orient='records'))

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    
    # Use 0.0.0.0 to bind to all interfaces
    # This is important for Render.com deployment
    app.run(host='0.0.0.0', port=port, debug=False)