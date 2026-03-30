"""
Main entry point for the Drug Toxicity Prediction Application
Provides both Streamlit web interface and command-line functionality
"""

import sys
import argparse
import logging
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drug_toxicity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def run_streamlit_app():
    """Run the Streamlit web application"""
    try:
        import streamlit.web.cli as stcli
        from streamlit import config as st_config
        
        # Set Streamlit configuration
        st_config.set_option("server.headless", False)
        st_config.set_option("browser.gatherUsageStats", False)
        
        logger.info("Starting Streamlit application...")
        
        # Run the Streamlit app
        sys.argv = ["streamlit", "run", "streamlit_app.py"]
        stcli.main()
        
    except ImportError:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        return False
    except Exception as e:
        logger.error(f"Error running Streamlit app: {e}")
        return False
    
    return True

def run_fastapi_server():
    """Run the FastAPI server"""
    try:
        import uvicorn
        from app import app
        
        logger.info("Starting FastAPI server...")
        
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError:
        logger.error("FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn")
        return False
    except Exception as e:
        logger.error(f"Error running FastAPI server: {e}")
        return False
    
    return True

def run_cli_prediction(smiles: str, model_type: str = "ensemble"):
    """Run command-line prediction"""
    try:
        import numpy as np
        from src.data_processing import MolecularDescriptorCalculator
        from src.ml_models import EnsembleModel
        
        logger.info(f"Running CLI prediction for: {smiles}")
        
        # Calculate descriptors
        descriptor_calc = MolecularDescriptorCalculator()
        descriptors = descriptor_calc.calculate_descriptors([smiles])
        
        if len(descriptors) == 0:
            logger.error("Invalid SMILES string")
            return False
        
        # Make prediction (demo mode if no model)
        try:
            ensemble_model = EnsembleModel()
            # Try to load trained model
            if Path("models/ensemble_model.pkl").exists():
                ensemble_model.load_model("models/ensemble_model.pkl")
                prediction = ensemble_model.predict(descriptors)
                probabilities = ensemble_model.predict_proba(descriptors)
            else:
                # Demo prediction
                probabilities = [[0.7, 0.3]]  # [non-toxic, toxic]
                prediction = [0]
                logger.warning("No trained model found. Using demo predictions.")
            
            # Display results
            print(f"\n{'='*60}")
            print(f"DRUG TOXICITY PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"SMILES: {smiles}")
            print(f"Predicted Class: {'Toxic' if prediction[0] == 1 else 'Non-Toxic'}")
            print(f"Confidence: {probabilities[0][np.argmax(probabilities[0])]:.3f}")
            
            if len(probabilities[0]) > 2:  # Multi-class
                print(f"\nDetailed Predictions:")
                for i, prob in enumerate(probabilities[0]):
                    print(f"  Endpoint {i}: {prob:.3f}")
            
            print(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return False
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        return False
    except Exception as e:
        logger.error(f"Error in CLI prediction: {e}")
        return False
    
    return True

def setup_environment():
    """Setup the environment and create necessary directories"""
    directories = [
        "data",
        "models", 
        "outputs",
        "outputs/reports",
        "logs",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment setup completed")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "numpy",
        "pandas", 
        "sklearn",
        "rdkit",
        "streamlit",
        "plotly",
        "matplotlib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Drug Toxicity Prediction Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run Streamlit web app
  python main.py --server                 # Run FastAPI server
  python main.py --predict "CCO"          # CLI prediction
  python main.py --setup                   # Setup environment
  python main.py --check-deps             # Check dependencies
        """
    )
    
    parser.add_argument(
        "--server", 
        action="store_true",
        help="Run FastAPI server"
    )
    
    parser.add_argument(
        "--predict",
        type=str,
        metavar="SMILES",
        help="Run CLI prediction for given SMILES"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["ensemble", "xgboost", "lightgbm", "random_forest"],
        help="Model type for prediction (default: ensemble)"
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup environment and create directories"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if required dependencies are installed"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit app (default: 8501)"
    )
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Please install missing dependencies before running the application")
        return 1
    
    # Handle different modes
    if args.check_deps:
        return 0
    
    elif args.setup:
        logger.info("Environment setup completed")
        return 0
    
    elif args.server:
        return 0 if run_fastapi_server() else 1
    
    elif args.predict:
        return 0 if run_cli_prediction(args.predict, args.model) else 1
    
    else:
        # Default: run Streamlit app
        return 0 if run_streamlit_app() else 1

if __name__ == "__main__":
    sys.exit(main())
