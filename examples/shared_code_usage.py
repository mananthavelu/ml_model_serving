"""
Example showing shared code usage across different deployment environments.

This demonstrates how src/utils/ provides common functionality that can be used
by both development scripts and production services.
"""

# Example 1: Development script usage
from src.utils import setup_logging, load_config, get_config_path

# Setup logging for development
setup_logging("DEBUG")

# Load configuration
config = load_config(get_config_path())
print(f"Loaded config with {len(config)} sections")

# Example 2: Production API usage (hypothetical)
# In a production API, you might only need config loading
from src.utils import load_config

class HousingAPI:
    def __init__(self):
        self.config = load_config("config/config.yaml")
        # Use config for API settings...

# Example 3: Batch processing job usage
# In a production batch job, you might need logging + config + directory utils
from src.utils import setup_logging, load_config, ensure_directory

def run_batch_processing():
    setup_logging("INFO")
    config = load_config("config/config.yaml")

    # Ensure output directories exist
    ensure_directory("data/processed")
    ensure_directory("models/artifacts")

    # Run processing logic...
    print("Batch processing complete")

if __name__ == "__main__":
    run_batch_processing()