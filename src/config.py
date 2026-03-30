"""
Configuration file for Drug Toxicity Prediction Application
Contains paths, model parameters, and application settings
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# Project Structure
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset Configuration
DATASETS = {
    "tox21": {
        "url": "https://www.kaggle.com/datasets/epicskills/tox21-dataset",
        "local_path": DATA_DIR / "tox21.csv",
        "description": "Tox21 toxicity dataset with ~12,000 compounds"
    },
    "zinc250k": {
        "url": "https://www.kaggle.com/datasets/basu369victor/zinc250k",
        "local_path": DATA_DIR / "zinc250k.csv",
        "description": "250,000 drug-like molecules with properties"
    }
}

# Molecular Descriptor Configuration
DESCRIPTOR_CONFIG = {
    "rdkit_descriptors": [
        "MolWt", "LogP", "NumHDonors", "NumHAcceptors", 
        "TPSA", "NumRotatableBonds", "FractionCSP3"
    ],
    "mordred_descriptors": True,  # Use all Mordred descriptors
    "custom_descriptors": [
        "bertz_ct", "balaban_j", "wiener_index", "electrotopological"
    ]
}

# Model Configuration
MODEL_CONFIG = {
    "ensemble_models": {
        "xgboost": {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "lightgbm": {
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        },
        "catboost": {
            "iterations": 500,
            "depth": 8,
            "learning_rate": 0.01,
            "random_seed": 42,
            "verbose": False
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 10,
            "random_state": 42
        }
    },
    "deep_learning": {
        "gnn": {
            "hidden_channels": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "epochs": 100
        },
        "mlp": {
            "hidden_layers": [512, 256, 128],
            "dropout": 0.3,
            "epochs": 100
        }
    }
}

# Toxicity Types Configuration
TOXICITY_TYPES = {
    "NR": {
        "full_name": "Nuclear Receptors",
        "assays": ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma"]
    },
    "SR": {
        "full_name": "Stress Response",
        "assays": ["SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
    }
}

# Visualization Configuration
VIZ_CONFIG = {
    "color_palette": {
        "low_toxicity": "#2E8B57",  # Sea green
        "medium_toxicity": "#FFD700",  # Gold
        "high_toxicity": "#DC143C",  # Crimson
        "neutral": "#708090"  # Slate gray
    },
    "molecule_viewer": {
        "width": 400,
        "height": 400,
        "background": "white",
        "style": "stick"
    }
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "title": "Drug Toxicity Prediction API",
    "description": "Advanced AI-powered drug toxicity prediction system",
    "version": "1.0.0"
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "Drug Toxicity Predictor",
    "page_icon": "💊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Feature Importance Configuration
FEATURE_IMPORTANCE_CONFIG = {
    "method": "shap",  # Options: shap, lime, permutation
    "n_samples": 100,
    "background_size": 50
}

# Similarity Search Configuration
SIMILARITY_CONFIG = {
    "method": "tanimoto",  # Options: tanimoto, cosine, euclidean
    "fingerprint_type": "morgan",  # Options: morgan, maccs, rdkit
    "radius": 2,
    "n_bits": 2048,
    "threshold": 0.7
}

# Report Generation Configuration
REPORT_CONFIG = {
    "template_path": BASE_DIR / "templates" / "report_template.html",
    "output_format": ["pdf", "html"],
    "include_3d_structure": True,
    "include_similar_compounds": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": LOGS_DIR / "drug_toxicity.log"
}

# Advanced Features Configuration
ADVANCED_FEATURES = {
    "chemical_space_analysis": {
        "method": "umap",  # Options: umap, tsne, pca
        "n_components": 2,
        "min_dist": 0.1,
        "n_neighbors": 15
    },
    "adversarial_validation": {
        "enabled": True,
        "threshold": 0.7
    },
    "uncertainty_quantification": {
        "method": "monte_carlo",  # Options: monte_carlo, ensemble, bayesian
        "n_samples": 100
    },
    "multi_task_learning": {
        "enabled": True,
        "shared_layers": 2
    }
}

# Chatbot Configuration
CHATBOT_CONFIG = {
    "model_name": "microsoft/DialoGPT-medium",
    "max_length": 128,
    "temperature": 0.7,
    "knowledge_base": DATA_DIR / "knowledge_base.json"
}

# Environment Variables
ENV_VARS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
    "DATABASE_URL": os.getenv("DATABASE_URL", "sqlite:///drug_toxicity.db")
}

# Validation Configuration
VALIDATION_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "cv_folds": 5,
    "random_state": 42,
    "stratify": True
}

# Performance Monitoring
PERFORMANCE_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
    "cross_validation": True,
    "calibration": True,
    "threshold_optimization": True
}
