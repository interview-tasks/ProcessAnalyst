from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__)

@app.route('/status')
def status():
    # Return information about the file structure to help debug
    dashboard_exists = os.path.exists('dashboard')
    index_exists = os.path.exists('dashboard/index.html')
    
    file_structure = {
        'cwd': os.getcwd(),
        'dashboard_exists': dashboard_exists,
        'index_exists': index_exists,
        'files_in_root': os.listdir('.') if os.path.exists('.') else [],
        'files_in_dashboard': os.listdir('dashboard') if dashboard_exists else []
    }
    
    return jsonify(file_structure)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # First, try to serve the file from the dashboard directory
    dashboard_path = os.path.join('dashboard', path)
    
    # For debugging
    print(f"Requested path: {path}")
    print(f"Looking for file at: {dashboard_path}")
    
    if path and os.path.exists(dashboard_path) and os.path.isfile(dashboard_path):
        directory, filename = os.path.split(dashboard_path)
        return send_from_directory(directory, filename)
    
    # If path is empty or file doesn't exist, try to serve index.html
    if os.path.exists('dashboard/index.html'):
        return send_from_directory('dashboard', 'index.html')
    
    # If all else fails, return a helpful error message
    return f"""
    <html>
    <head><title>File Not Found</title></head>
    <body>
        <h1>SOCAR Dashboard Error</h1>
        <p>Requested path: {path}</p>
        <p>Working directory: {os.getcwd()}</p>
        <p>Dashboard directory exists: {os.path.exists('dashboard')}</p>
        <p>Index.html exists: {os.path.exists('dashboard/index.html')}</p>
        <h2>Files in root:</h2>
        <ul>{"".join([f"<li>{f}</li>" for f in os.listdir('.')])}</ul>
        <h2>Files in dashboard (if exists):</h2>
        <ul>{"".join([f"<li>{f}</li>" for f in os.listdir('dashboard')]) if os.path.exists('dashboard') else "Dashboard directory not found"}</ul>
    </body>
    </html>
    """, 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)