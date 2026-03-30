"""
Train.py — Full Model Training Pipeline
==========================================
Run this script to:
1. Process the Tox21 dataset (or generate synthetic data)
2. Train Random Forest + XGBoost ensemble models
3. Evaluate performance per toxicity endpoint
4. Build the similarity search database
5. Initialize SHAP explainers
6. Save everything to the models/ directory

Usage:
    python train.py                     # Train with default settings
    python train.py --data ./data       # Specify data directory
    python train.py --quick             # Quick training (fewer estimators)

Step-by-step explanation:
    Step 1: Load/generate Tox21 molecular data
    Step 2: Compute molecular descriptors (RDKit + Morgan + MACCS)
    Step 3: Split data into train/test sets (80/20 stratified)
    Step 4: Train RF + XGBoost for each of 12 endpoints
    Step 5: Evaluate AUC-ROC, AUC-PR, F1 per endpoint
    Step 6: Build fingerprint similarity database
    Step 7: Initialize SHAP explainers with background data
    Step 8: Save all models and metadata
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split

from src.data_processing.descriptor_generator import process_dataset, TOX21_LABELS
from src.models.classical_models import MultiTaskRandomForest, MultiTaskXGBoost, EnsembleModel, evaluate_models
from src.similarity.fingerprint_search import SimilaritySearcher
from src.explainability.shap_explainer import ToxicityExplainer, detect_structural_alerts

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Drug Toxicity Prediction Models')
    parser.add_argument('--data', type=str, default='./data', help='Data directory path')
    parser.add_argument('--models', type=str, default='./models', help='Models output directory')
    parser.add_argument('--quick', action='store_true', help='Quick training with fewer estimators')
    parser.add_argument('--no-shap', action='store_true', help='Skip SHAP explainer initialization')
    return parser.parse_args()


def main():
    args = parse_args()
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / args.data
    models_dir = base_dir / args.models
    processed_dir = data_dir / 'processed'
    
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  DRUG TOXICITY PREDICTION — TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Models directory: {models_dir}")
    
    # ============================================================
    # STEP 1: PROCESS DATASET
    # ============================================================
    logger.info("\n--- STEP 1: Processing Dataset ---")
    
    # Check if already processed
    if (processed_dir / 'features.parquet').exists():
        logger.info("Loading pre-processed data...")
        X = pd.read_parquet(processed_dir / 'features.parquet')
        y = pd.read_parquet(processed_dir / 'labels.parquet')
        smiles = pd.read_csv(processed_dir / 'smiles.csv').squeeze()
        feature_names = pd.read_csv(processed_dir / 'feature_names.csv').squeeze().tolist()
        logger.info(f"Loaded: {X.shape[0]} compounds, {X.shape[1]} features")
    else:
        X, y, smiles, feature_names = process_dataset(
            str(data_dir), str(processed_dir)
        )
    
    X_arr = X.values.astype(np.float32)
    
    # ============================================================
    # STEP 2: TRAIN/TEST SPLIT
    # ============================================================
    logger.info("\n--- STEP 2: Train/Test Split (80/20) ---")
    
    # Use first available label for stratified split
    strat_col = None
    for label in TOX21_LABELS:
        if label in y.columns and y[label].notna().sum() > 50:
            strat_col = y[label].fillna(0).astype(int)
            break
    
    idx = np.arange(len(X_arr))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=strat_col.values if strat_col is not None else None
    )
    
    X_train, X_test = X_arr[train_idx], X_arr[test_idx]
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    smiles_train = smiles.iloc[train_idx].reset_index(drop=True)
    
    logger.info(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # ============================================================
    # STEP 3: TRAIN MODELS
    # ============================================================
    logger.info("\n--- STEP 3: Training Models ---")
    
    n_estimators_rf = 100 if args.quick else 200
    n_estimators_xgb = 150 if args.quick else 300
    
    # Train Ensemble (RF + XGBoost)
    ensemble = EnsembleModel(rf_weight=0.4, xgb_weight=0.6)
    
    # Train RF
    logger.info("\n[RF] Training Random Forest...")
    ensemble.rf = MultiTaskRandomForest(
        n_estimators=n_estimators_rf,
        n_jobs=-1,
        random_state=42
    )
    ensemble.rf.fit(X_train, y_train, feature_names)
    
    # Train XGBoost
    logger.info("\n[XGB] Training XGBoost...")
    ensemble.xgb = MultiTaskXGBoost(
        n_estimators=n_estimators_xgb,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    ensemble.xgb.fit(X_train, y_train, feature_names)
    
    ensemble.feature_names = feature_names
    ensemble.label_names = list(y_train.columns)
    ensemble.is_fitted = True
    
    # ============================================================
    # STEP 4: EVALUATE MODELS
    # ============================================================
    logger.info("\n--- STEP 4: Evaluating Models ---")
    
    logger.info("\n[RF] Evaluation:")
    rf_results = evaluate_models(ensemble.rf, X_test, y_test)
    logger.info(f"\n{rf_results[['endpoint', 'n_samples', 'AUC-ROC', 'AUC-PR', 'F1']].to_string(index=False)}")
    
    logger.info("\n[XGB] Evaluation:")
    xgb_results = evaluate_models(ensemble.xgb, X_test, y_test)
    logger.info(f"\n{xgb_results[['endpoint', 'n_samples', 'AUC-ROC', 'AUC-PR', 'F1']].to_string(index=False)}")
    
    logger.info("\n[Ensemble] Evaluation:")
    ens_results = evaluate_models(ensemble, X_test, y_test)
    logger.info(f"\n{ens_results[['endpoint', 'n_samples', 'AUC-ROC', 'AUC-PR', 'F1']].to_string(index=False)}")
    
    mean_auc = ens_results['AUC-ROC'].mean()
    logger.info(f"\n>>> Mean Ensemble AUC-ROC: {mean_auc:.4f}")
    
    # Save evaluation results
    rf_results.to_csv(models_dir / 'rf_evaluation.csv', index=False)
    xgb_results.to_csv(models_dir / 'xgb_evaluation.csv', index=False)
    ens_results.to_csv(models_dir / 'ensemble_evaluation.csv', index=False)
    
    # ============================================================
    # STEP 5: SAVE MODELS
    # ============================================================
    logger.info("\n--- STEP 5: Saving Models ---")
    ensemble.save(str(models_dir))
    
    # Save feature names
    pd.Series(feature_names).to_csv(models_dir / 'feature_names.csv', index=False)
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'n_train': int(X_train.shape[0]),
        'n_test': int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
        'n_endpoints': len(ensemble.label_names),
        'endpoints': ensemble.label_names,
        'mean_auc_roc': float(mean_auc),
        'rf_n_estimators': n_estimators_rf,
        'xgb_n_estimators': n_estimators_xgb,
        'quick_mode': args.quick,
    }
    with open(models_dir / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {models_dir / 'training_metadata.json'}")
    
    # ============================================================
    # STEP 6: BUILD SIMILARITY DATABASE
    # ============================================================
    logger.info("\n--- STEP 6: Building Similarity Search Database ---")
    
    searcher = SimilaritySearcher(radius=2, n_bits=2048)
    searcher.build_database(smiles_train, y_train)
    searcher.save(str(models_dir))
    
    # ============================================================
    # STEP 7: INITIALIZE SHAP EXPLAINERS (optional)
    # ============================================================
    if not args.no_shap:
        logger.info("\n--- STEP 7: Initializing SHAP Explainers ---")
        try:
            import joblib
            
            # Use small background set for SHAP (100 samples is sufficient)
            bg_size = min(100, X_train.shape[0])
            bg_indices = np.random.choice(X_train.shape[0], bg_size, replace=False)
            X_background = X_train[bg_indices]
            
            # Initialize for RF
            rf_explainer = ToxicityExplainer(ensemble.rf, feature_names)
            rf_explainer.initialize_explainers(
                X_background,
                labels=list(ensemble.rf.models.keys())
            )
            joblib.dump(rf_explainer, models_dir / 'rf_shap_explainer.pkl')
            
            # Initialize for XGB
            xgb_explainer = ToxicityExplainer(ensemble.xgb, feature_names)
            xgb_explainer.initialize_explainers(
                X_background,
                labels=list(ensemble.xgb.models.keys())
            )
            joblib.dump(xgb_explainer, models_dir / 'xgb_shap_explainer.pkl')
            
            # Save background data for future use
            np.save(models_dir / 'shap_background.npy', X_background)
            
            logger.info(f"SHAP explainers saved to {models_dir}")
        except Exception as e:
            logger.warning(f"SHAP initialization failed (non-critical): {e}")
    else:
        logger.info("Skipping SHAP initialization (--no-shap flag)")
    
    # ============================================================
    # DONE
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Mean AUC-ROC: {mean_auc:.4f}")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Run 'python app.py' to start the prediction server")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return ensemble, mean_auc


if __name__ == '__main__':
    main()
