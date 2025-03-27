from flask import Flask, send_from_directory, jsonify
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/debug')
def debug():
    """Return detailed information about the file structure"""
    logger.info("Debug endpoint accessed")
    
    # Get file structure info
    cwd = os.getcwd()
    dir_contents = {}
    
    # Check main directories
    dir_contents['root'] = os.listdir(cwd) if os.path.exists(cwd) else []
    
    # Check for dashboard directory
    dashboard_path = os.path.join(cwd, 'dashboard')
    dashboard_exists = os.path.exists(dashboard_path)
    if dashboard_exists:
        dir_contents['dashboard'] = os.listdir(dashboard_path)
        
        # Check for key subdirectories
        for subdir in ['assets', 'data', 'charts']:
            subdir_path = os.path.join(dashboard_path, subdir)
            if os.path.exists(subdir_path):
                dir_contents[f'dashboard/{subdir}'] = os.listdir(subdir_path)
    
    # Check for key files
    key_files = {
        'index.html': os.path.exists(os.path.join(cwd, 'index.html')),
        'dashboard/index.html': os.path.exists(os.path.join(dashboard_path, 'index.html')),
        'app.py': os.path.exists(os.path.join(cwd, 'app.py')),
        'requirements.txt': os.path.exists(os.path.join(cwd, 'requirements.txt'))
    }
    
    return jsonify({
        'current_working_directory': cwd,
        'directory_contents': dir_contents,
        'key_files': key_files
    })

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_files(path):
    """Serve files from the correct location"""
    logger.info(f"Request for path: '{path}'")
    
    # Define possible locations for index.html
    root_index = os.path.join(os.getcwd(), 'index.html')
    dashboard_index = os.path.join(os.getcwd(), 'dashboard', 'index.html')
    
    # Check if path is empty (root request)
    if not path:
        logger.info("Empty path requested, looking for index.html")
        
        # Try dashboard/index.html first
        if os.path.exists(dashboard_index):
            logger.info(f"Serving dashboard/index.html")
            return send_from_directory(os.path.join(os.getcwd(), 'dashboard'), 'index.html')
        
        # Try root index.html next
        if os.path.exists(root_index):
            logger.info(f"Serving root index.html")
            return send_from_directory(os.getcwd(), 'index.html')
        
        # If neither exists, return helpful error
        logger.error("No index.html found!")
        return "No index.html found in either root or dashboard directory!", 404
    
    # Check if file exists in the dashboard directory
    dashboard_path = os.path.join(os.getcwd(), 'dashboard', path)
    if os.path.exists(dashboard_path) and os.path.isfile(dashboard_path):
        logger.info(f"File found in dashboard directory: {dashboard_path}")
        return send_from_directory(os.path.join(os.getcwd(), 'dashboard'), path)
    
    # Check if file exists in the root directory
    root_path = os.path.join(os.getcwd(), path)
    if os.path.exists(root_path) and os.path.isfile(root_path):
        logger.info(f"File found in root directory: {root_path}")
        return send_from_directory(os.getcwd(), path)
    
    # If we get here, the file wasn't found
    logger.error(f"File not found: {path}")
    return f"File not found: {path}", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)