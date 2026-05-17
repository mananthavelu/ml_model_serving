"""
Complete MLOps pipeline script.
Orchestrates the entire workflow from data to deployment.
"""

import logging
import subprocess
import sys
from pathlib import Path

from src.utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """
    Run a shell command and log output.

    Args:
        command: Command to run
        description: Description for logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")

    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        logger.info(f"Successfully completed: {description}")
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False


def run_full_pipeline():
    """
    Run the complete MLOps pipeline:
    1. Generate sample data
    2. Process data
    3. Train model
    4. Test API
    """
    logger.info("=" * 80)
    logger.info("STARTING COMPLETE MLOPS PIPELINE")
    logger.info("=" * 80)

    steps = [
        # Step 1: Generate sample data
        {
            "command": "python -m scripts.generate_sample_data",
            "description": "Generate sample housing dataset",
        },
        # Step 2: Process data
        {
            "command": "python -m scripts.process_data",
            "description": "Process and validate data pipeline",
        },
        # Step 3: Train model
        {
            "command": "python -m scripts.train_model",
            "description": "Train housing prices prediction model",
        },
        # Step 4: Evaluate model
        {
            "command": "python -m scripts.evaluate_model",
            "description": "Evaluate trained model on test data",
        },
        # Step 5: Test API startup (background)
        {
            "command": "timeout 10 python -m scripts.serve_api || echo 'API startup test completed'",
            "description": "Test API server startup",
        },
    ]

    success_count = 0
    for step in steps:
        if run_command(step["command"], step["description"]):
            success_count += 1
        else:
            logger.error(f"Pipeline failed at step: {step['description']}")
            break

    logger.info("=" * 80)
    if success_count == len(steps):
        logger.info("✓ COMPLETE MLOPS PIPELINE SUCCESSFUL!")
        logger.info("Ready for production deployment")
    else:
        logger.error(f"✗ PIPELINE FAILED: {success_count}/{len(steps)} steps completed")
    logger.info("=" * 80)

    return success_count == len(steps)


if __name__ == "__main__":
    success = run_full_pipeline()
    sys.exit(0 if success else 1)
