#!/usr/bin/env python3
"""
SOCAR Process Analysis Dashboard Runner

This script runs the complete data preparation and visualization generation pipeline
for the SOCAR Process Analysis Dashboard.

Usage:
    python run_dashboard.py [options]

Options:
    --input-file PATH       Path to the input data file (default: analysis/data/data.csv)
    --output-dir PATH       Path to the output directory (default: dashboard/socar-dashboard)
    --data-config PATH      Path to the data configuration file (default: data_config.yaml)
    --viz-config PATH       Path to the visualization configuration file (default: visualization_config.yaml)
    --skip-data-prep        Skip the data preparation step
    --skip-visualization    Skip the visualization generation step
    --open-dashboard        Open the dashboard in a browser when done
    --create-report         Generate an HTML report with all visualizations
    --create-index          Create a dashboard index HTML file
    --quality-check         Perform quality checks on generated visualizations
    --add-timestamp         Add timestamp to output directories
    --create-zip            Create a ZIP archive of all outputs
    --debug                 Enable debug logging
"""

import os
import sys
import time
import argparse
import subprocess
import logging
import webbrowser
import shutil
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dashboard_runner")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SOCAR Process Analysis Dashboard Runner')
    
    # Input/output paths
    parser.add_argument('--input-file', type=str, default='analysis/data/data.csv',
                       help='Path to the input data file')
    parser.add_argument('--output-dir', type=str, default='dashboard/socar-dashboard',
                       help='Path to the output directory')
    parser.add_argument('--data-config', type=str, default='data_config.yaml',
                       help='Path to the data configuration file')
    parser.add_argument('--viz-config', type=str, default='visualization_config.yaml',
                       help='Path to the visualization configuration file')
    
    # Process control
    parser.add_argument('--skip-data-prep', action='store_true',
                       help='Skip the data preparation step')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='Skip the visualization generation step')
    
    # Additional features
    parser.add_argument('--open-dashboard', action='store_true',
                       help='Open the dashboard in a browser when done')
    parser.add_argument('--create-report', action='store_true',
                       help='Generate an HTML report with all visualizations')
    parser.add_argument('--create-index', action='store_true',
                       help='Create a dashboard index HTML file')
    parser.add_argument('--quality-check', action='store_true',
                       help='Perform quality checks on generated visualizations')
    parser.add_argument('--add-timestamp', action='store_true',
                       help='Add timestamp to output directories')
    parser.add_argument('--create-zip', action='store_true',
                       help='Create a ZIP archive of all outputs')
    
    # Logging control
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()

def run_data_preparation(input_file, output_dir, config_file, debug=False):
    """
    Run the data preparation script
    
    Args:
        input_file: Path to the input data file
        output_dir: Path to the output directory
        config_file: Path to the configuration file
        debug: Enable debug logging
    
    Returns:
        Success status (True/False)
    """
    logger.info("Starting data preparation")
    start_time = time.time()
    
    try:
        # Create command with arguments
        cmd = [
            sys.executable, 
            "dashboard/scripts/prepare_data.py",
            "--input", input_file,
            "--output", output_dir,
            "--config", config_file
        ]
        
        if debug:
            cmd.append("--debug")
        
        # Run the process and capture output
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Get return code
        process.wait()
        return_code = process.returncode
        
        # Check for errors
        if return_code != 0:
            error_output = process.stderr.read()
            logger.error(f"Data preparation failed with return code {return_code}")
            logger.error(f"Error output: {error_output}")
            return False
        
        # Data preparation successful
        elapsed_time = time.time() - start_time
        logger.info(f"Data preparation completed successfully in {elapsed_time:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error running data preparation: {e}", exc_info=True)
        return False

def run_visualization_generation(data_dir, output_dir, config_file, additional_args=None, debug=False):
    """
    Run the visualization generation script
    
    Args:
        data_dir: Path to the data directory
        output_dir: Path to the output directory
        config_file: Path to the configuration file
        additional_args: Additional arguments to pass to the script
        debug: Enable debug logging
    
    Returns:
        Success status (True/False)
    """
    logger.info("Starting visualization generation")
    start_time = time.time()
    
    try:
        # Create command with arguments
        cmd = [
            sys.executable, 
            "dashboard/scripts/generate_visualizations.py",
            "--data-dir", data_dir,
            "--output-dir", output_dir,
            "--config", config_file
        ]
        
        if debug:
            cmd.append("--debug")
        
        # Add any additional arguments
        if additional_args:
            cmd.extend(additional_args)
        
        # Run the process and capture output
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            logger.info(line.strip())
        
        # Get return code
        process.wait()
        return_code = process.returncode
        
        # Check for errors
        if return_code != 0:
            error_output = process.stderr.read()
            logger.error(f"Visualization generation failed with return code {return_code}")
            logger.error(f"Error output: {error_output}")
            return False
        
        # Visualization generation successful
        elapsed_time = time.time() - start_time
        logger.info(f"Visualization generation completed successfully in {elapsed_time:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error running visualization generation: {e}", exc_info=True)
        return False

def create_zip_archive(source_dir, zip_name=None):
    """
    Create a ZIP archive of the specified directory
    
    Args:
        source_dir: Directory to archive
        zip_name: Name for the ZIP file (optional)
    
    Returns:
        Path to the created ZIP file
    """
    try:
        import zipfile
        
        # Create zip filename
        if zip_name is None:
            zip_name = f"{os.path.basename(source_dir)}_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        # Ensure source_dir exists
        if not os.path.exists(source_dir):
            logger.error(f"Cannot create ZIP archive: Source directory {source_dir} does not exist")
            return None
        
        # Create zip file
        zip_path = os.path.join(os.path.dirname(source_dir), zip_name)
        logger.info(f"Creating ZIP archive: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                    zipf.write(file_path, arcname)
        
        logger.info(f"ZIP archive created successfully: {zip_path}")
        return zip_path
    
    except Exception as e:
        logger.error(f"Error creating ZIP archive: {e}", exc_info=True)
        return None

def copy_index_html(output_dir):
    """
    Copy the improved index.html to the output directory
    
    Args:
        output_dir: Path to the output directory
    
    Returns:
        Success status (True/False)
    """
    try:
        # Source index.html (from project root)
        source_index = 'index.html'
        
        # Target in the output directory
        target_index = os.path.join(output_dir, 'index.html')
        
        # Copy the file
        logger.info(f"Copying index.html to {target_index}")
        shutil.copy2(source_index, target_index)
        
        # Adjust relative paths if needed
        with open(target_index, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # If we're putting the index.html in a subdirectory, we may need to adjust paths
        # This depends on your specific directory structure
        adjusted_content = content.replace('dashboard/assets/', '../assets/')
        
        with open(target_index, 'w', encoding='utf-8') as f:
            f.write(adjusted_content)
        
        logger.info("index.html copied and adjusted successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error copying index.html: {e}", exc_info=True)
        return False

def main():
    """Main function to run the dashboard generation pipeline"""
    start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Add timestamp to output directory if requested
    if args.add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir_base = args.output_dir
        args.output_dir = f"{output_dir_base}_{timestamp}"
        logger.info(f"Using timestamped output directory: {args.output_dir}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define paths for various outputs
    data_dir = os.path.join(args.output_dir, 'data')
    charts_dir = os.path.join(args.output_dir, 'charts')
    
    # Track success status
    success = True
    
    # Run data preparation (unless skipped)
    if not args.skip_data_prep:
        success = run_data_preparation(
            args.input_file, 
            data_dir, 
            args.data_config,
            args.debug
        )
        
        if not success:
            logger.error("Data preparation failed. Cannot proceed with visualization generation.")
            if not args.skip_visualization:
                return 1
    else:
        logger.info("Skipping data preparation as requested")
    
    # Run visualization generation (unless skipped)
    if not args.skip_visualization and success:
        # Build additional arguments
        additional_args = []
        
        if args.create_report:
            additional_args.append('--html-report')
        
        if args.create_index:
            additional_args.append('--dashboard-index')
        
        if args.quality_check:
            additional_args.append('--quality-check')
        
        if args.create_zip:
            additional_args.append('--create-zip')
        
        # Run visualization generation
        success = run_visualization_generation(
            data_dir,
            charts_dir,
            args.viz_config,
            additional_args,
            args.debug
        )
        
        if not success:
            logger.error("Visualization generation failed.")
            return 1
    else:
        if not args.skip_visualization:
            logger.info("Cannot run visualization generation due to failed data preparation.")
        else:
            logger.info("Skipping visualization generation as requested")
    
    # Copy index.html to output directory
    copy_index_html(args.output_dir)
    
    # Create ZIP archive if requested
    if args.create_zip:
        zip_path = create_zip_archive(args.output_dir)
        if zip_path:
            logger.info(f"ZIP archive created: {zip_path}")
    
    # Open dashboard in browser if requested
    if args.open_dashboard:
        index_path = os.path.join(args.output_dir, 'index.html')
        if os.path.exists(index_path):
            logger.info(f"Opening dashboard in browser: {index_path}")
            webbrowser.open(f"file://{os.path.abspath(index_path)}")
        else:
            logger.error(f"Cannot open dashboard: index.html not found at {index_path}")
    
    # Log total execution time
    elapsed_time = time.time() - start_time
    logger.info(f"Dashboard generation process completed in {elapsed_time:.2f} seconds")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())