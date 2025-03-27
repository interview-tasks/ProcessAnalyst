from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__, static_folder='.')

@app.route('/status')
def status():
    # Return information about the file structure
    cwd = os.getcwd()
    return jsonify({
        'cwd': cwd,
        'files_in_cwd': os.listdir(cwd),
        'index_exists': os.path.exists('index.html'),
        'assets_exists': os.path.exists('assets'),
        'data_exists': os.path.exists('data'),
        'charts_exists': os.path.exists('charts')
    })

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # Handle the root path
    if not path:
        return send_from_directory('.', 'index.html')
    
    # For all other paths, serve from the current directory
    if os.path.exists(path) and os.path.isfile(path):
        directory, file = os.path.split(path)
        if directory:
            return send_from_directory(directory, file)
        else:
            return send_from_directory('.', file)
    
    # If the direct path doesn't exist, try normal flask static serving
    try:
        return app.send_static_file(path)
    except:
        pass
    
    # If all else fails, return 404
    return f"File not found: {path}", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)