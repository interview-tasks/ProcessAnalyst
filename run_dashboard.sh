#!/bin/bash

# SOCAR Process Analysis Dashboard - Local Development Server

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null
then
    echo "Error: Python is required but not found. Please install Python 3."
    exit 1
fi

# Determine Python command
PYTHON_CMD="python"
if ! command -v python &> /dev/null && command -v python3 &> /dev/null
then
    PYTHON_CMD="python3"
fi

# Check if any port is provided as an argument
PORT=${1:-8000}

echo "-------------------------------------------------------"
echo "       SOCAR Process Analysis Dashboard"
echo "-------------------------------------------------------"
echo "Starting local server on port $PORT..."
echo ""

# Create necessary directories if they don't exist
mkdir -p dashboard/assets/css
mkdir -p dashboard/assets/js
mkdir -p dashboard/assets/img
mkdir -p dashboard/data
mkdir -p dashboard/charts

# Make sure to have the latest data for the dashboard
echo "Checking data preparation..."
if [ -f "dashboard/scripts/prepare_data.py" ]; then
    echo "Running data preparation script..."
    cd dashboard
    $PYTHON_CMD scripts/prepare_data.py
    cd ..
    echo "Data preparation complete."
else
    echo "Data preparation script not found. Using existing data."
fi

# Generate visualizations if needed
echo "Checking visualization generation..."
if [ -f "dashboard/scripts/generate_visualizations.py" ]; then
    echo "Running visualization generation script..."
    cd dashboard
    $PYTHON_CMD scripts/generate_visualizations.py
    cd ..
    echo "Visualization generation complete."
else
    echo "Visualization script not found. Using existing visualizations."
fi

echo ""
echo "-------------------------------------------------------"
echo "Starting server at http://localhost:$PORT"
echo "CTRL+C to stop the server"
echo "-------------------------------------------------------"

# Start a simple HTTP server in the dashboard directory
cd dashboard
$PYTHON_CMD -m http.server $PORT

# This part will execute after the server is stopped
echo ""
echo "Server stopped."