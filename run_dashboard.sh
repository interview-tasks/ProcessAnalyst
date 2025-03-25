#!/bin/bash

# Create required directories
mkdir -p socar-dashboard/data
mkdir -p socar-dashboard/charts
mkdir -p socar-dashboard/assets

# Run data preparation
echo "Preparing data..."
python scripts/prepare_data.py

# Generate visualizations
echo "Generating visualizations..."
python scripts/generate_visualizations.py

# Report success
echo "Dashboard preparation complete!"
echo "Open socar-dashboard/index.html in your browser or deploy to GitHub Pages."
