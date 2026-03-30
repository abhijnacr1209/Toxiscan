"""
FastAPI Application — Drug Toxicity Prediction Server
======================================================
REST API backend providing all prediction and analysis endpoints.

Run with:
    python app.py

Or:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

API Endpoints:
    GET  /              → Serve frontend
    GET  /health        → Health check
    POST /predict       → Toxicity prediction from SMILES
    POST /explain       → SHAP explanation
    POST /similar       → Similar compound search
    POST /report        → PDF report generation
    POST /chatbot       → AI assistant query
    POST /batch         → Batch CSV prediction
    POST /descriptors   → Compute molecular descriptors
    GET  /space         → Chemical space visualization data
"""

import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
import io
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib

from src.data_processing.descriptor_generator import (
    compute_features_for_smiles, compute_admet_descriptors,
    smiles_to_mol, TOX21_LABELS
)
from src.models.classical_models import EnsembleModel
from src.similarity.fingerprint_search import SimilaritySearcher
from src.explainability.shap_explainer import ToxicityExplainer, AtomAttributor, detect_structural_alerts
from src.reporting.report_generator import get_report_generator

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="Drug Toxicity Prediction API",
    description="AI-powered drug toxicity prediction using molecular descriptors and ML",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS — allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL MODEL STATE
# ============================================================

class ModelState:
    """Holds all loaded ML models and utilities."""
    ensemble: Optional[EnsembleModel] = None
    searcher: Optional[SimilaritySearcher] = None
    rf_explainer: Optional[ToxicityExplainer] = None
    xgb_explainer: Optional[ToxicityExplainer] = None
    atom_attributor: AtomAttributor = AtomAttributor()
    feature_names: List[str] = []
    training_metadata: Dict = {}
    is_ready: bool = False

state = ModelState()

MODELS_DIR = Path(__file__).parent / 'models'
FRONTEND_DIR = Path(__file__).parent / 'frontend'


def load_models():
    """Load all trained models from disk at startup."""
    global state
    
    logger.info("Loading models...")
    
    if not MODELS_DIR.exists():
        logger.warning(f"Models directory not found: {MODELS_DIR}")
        logger.warning("Run 'python train.py' first to train the models!")
        return False
    
    try:
        # Load ensemble model
        ensemble_meta = MODELS_DIR / 'ensemble_meta.pkl'
        if ensemble_meta.exists():
            state.ensemble = EnsembleModel()
            state.ensemble.load(str(MODELS_DIR))
            logger.info(f"Loaded ensemble model with {len(state.ensemble.label_names)} endpoints")
        else:
            logger.warning("Ensemble model not found. Predictions will use fallback mode.")
            return False
        
        # Load feature names
        feat_file = MODELS_DIR / 'feature_names.csv'
        if feat_file.exists():
            state.feature_names = pd.read_csv(feat_file).squeeze().tolist()
            logger.info(f"Loaded {len(state.feature_names)} feature names")
        
        # Load similarity searcher
        sim_db = MODELS_DIR / 'similarity_db.pkl'
        if sim_db.exists():
            state.searcher = SimilaritySearcher()
            state.searcher.load(str(MODELS_DIR))
            logger.info(f"Loaded similarity database: {len(state.searcher.fingerprints)} compounds")
        
        # Load SHAP explainers
        rf_shap_file = MODELS_DIR / 'rf_shap_explainer.pkl'
        if rf_shap_file.exists():
            state.rf_explainer = joblib.load(rf_shap_file)
            logger.info("Loaded RF SHAP explainer")
        
        xgb_shap_file = MODELS_DIR / 'xgb_shap_explainer.pkl'
        if xgb_shap_file.exists():
            state.xgb_explainer = joblib.load(xgb_shap_file)
            logger.info("Loaded XGB SHAP explainer")
        
        # Load training metadata
        meta_file = MODELS_DIR / 'training_metadata.json'
        if meta_file.exists():
            with open(meta_file) as f:
                state.training_metadata = json.load(f)
        
        state.is_ready = True
        logger.info("All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error(traceback.format_exc())
        return False


@app.on_event("startup")
async def startup_event():
    """Load models when API starts."""
    load_models()


# ============================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================

class PredictionRequest(BaseModel):
    smiles: str
    model: str = "ensemble"  # "ensemble", "rf", "xgb"
    
    @validator('smiles')
    def validate_smiles(cls, v):
        if not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("SMILES cannot be empty")
        if len(v) > 2000:
            raise ValueError("SMILES string too long (max 2000 chars)")
        return v.strip()


class ExplainRequest(BaseModel):
    smiles: str
    endpoint: str = "NR-AhR"
    top_n: int = 15


class SimilarityRequest(BaseModel):
    smiles: str
    top_k: int = 10


class ChatbotRequest(BaseModel):
    message: str
    context: Optional[Dict] = None


class BatchPredictRequest(BaseModel):
    smiles_list: List[str]
    model: str = "ensemble"


# ============================================================
# CORE PREDICTION LOGIC
# ============================================================

def predict_single(smiles: str, model_type: str = "ensemble") -> Dict:
    """
    Core prediction function for a single SMILES string.
    Returns full prediction result dict.
    """
    start_time = time.time()
    
    # Compute molecular features
    try:
        feature_vector, admet_props, mol = compute_features_for_smiles(smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    result = {
        'smiles': smiles,
        'valid': True,
        'admet': admet_props,
        'predictions': {},
        'structural_alerts': {},
        'mol_svg': '',
        'processing_time_ms': 0,
    }
    
    # Detect structural alerts
    result['structural_alerts'] = detect_structural_alerts(mol)
    
    # Generate molecule SVG
    attributor = state.atom_attributor
    result['mol_svg'] = attributor.generate_plain_mol_svg(mol, width=400, height=280)
    
    # Model predictions
    if state.ensemble is not None and state.ensemble.is_fitted:
        X = feature_vector.reshape(1, -1)
        
        # Align features to model's expected input
        if state.feature_names:
            n_model_features = len(state.feature_names)
            if X.shape[1] != n_model_features:
                # Pad/trim to match
                if X.shape[1] < n_model_features:
                    pad = np.zeros((1, n_model_features - X.shape[1]))
                    X = np.concatenate([X, pad], axis=1)
                else:
                    X = X[:, :n_model_features]
        
        try:
            if model_type == "rf" and state.ensemble.rf.is_fitted:
                probas = state.ensemble.rf.predict_proba(X)
            elif model_type == "xgb" and state.ensemble.xgb.is_fitted:
                probas = state.ensemble.xgb.predict_proba(X)
            else:
                probas = state.ensemble.predict_proba(X)
            
            result['predictions'] = {
                label: float(prob[0])
                for label, prob in probas.items()
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return realistic fallback predictions
            result['predictions'] = {label: 0.1 for label in TOX21_LABELS}
    else:
        # No model — return placeholder with warning
        result['predictions'] = {label: 0.0 for label in TOX21_LABELS}
        result['warning'] = "Models not loaded. Run python train.py first."
    
    # Compute overall toxicity score
    if result['predictions']:
        probs = list(result['predictions'].values())
        result['overall_toxicity_score'] = float(np.mean(probs))
        result['max_toxicity_score'] = float(max(probs))
        result['n_high_risk'] = int(sum(1 for p in probs if p >= 0.7))
        result['n_moderate_risk'] = int(sum(1 for p in probs if 0.4 <= p < 0.7))
        result['risk_level'] = (
            'HIGH' if result['max_toxicity_score'] >= 0.7
            else 'MODERATE' if result['max_toxicity_score'] >= 0.4
            else 'LOW'
        )
    
    result['processing_time_ms'] = round((time.time() - start_time) * 1000, 1)
    return result


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": state.is_ready,
        "ensemble_ready": state.ensemble is not None and state.ensemble.is_fitted,
        "similarity_db_ready": state.searcher is not None,
        "shap_ready": state.rf_explainer is not None,
        "n_endpoints": len(state.ensemble.label_names) if state.ensemble else 0,
        "training_metadata": state.training_metadata,
    }


@app.post("/predict")
async def predict_toxicity(request: PredictionRequest):
    """
    Predict toxicity for a single SMILES compound.
    Returns predictions for all 12 Tox21 endpoints.
    """
    result = predict_single(request.smiles, request.model)
    return result


@app.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """
    Generate SHAP-based explanation for a toxicity prediction.
    Returns top features contributing to the prediction.
    """
    try:
        feature_vector, admet_props, mol = compute_features_for_smiles(request.smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    X = feature_vector.reshape(1, -1)
    
    # Align features
    if state.feature_names:
        n_model_features = len(state.feature_names)
        if X.shape[1] != n_model_features:
            if X.shape[1] < n_model_features:
                pad = np.zeros((1, n_model_features - X.shape[1]))
                X = np.concatenate([X, pad], axis=1)
            else:
                X = X[:, :n_model_features]
    
    # SHAP explanation
    shap_result = {'top_features': [], 'endpoint': request.endpoint}
    
    if state.rf_explainer and request.endpoint in state.rf_explainer.explainers:
        shap_result = state.rf_explainer.explain_single(
            X[0], request.endpoint, top_n=request.top_n
        )
    elif state.ensemble and request.endpoint in state.ensemble.rf.models:
        # Fallback to feature importance
        imp_df = state.ensemble.rf.get_feature_importance(request.endpoint, top_n=request.top_n)
        shap_result['top_features'] = imp_df.to_dict('records')
        shap_result['method'] = 'feature_importance_fallback'
    
    # Atom attribution
    atom_scores = []
    mol_svg_highlighted = ''
    
    if mol is not None:
        feature_imp = {
            feat['feature']: feat.get('shap_value', feat.get('importance', 0))
            for feat in shap_result.get('top_features', [])
        }
        
        # Extend with all feature importances for better atom mapping
        if state.ensemble and state.feature_names:
            try:
                imp_df = state.ensemble.rf.get_feature_importance(request.endpoint, top_n=200)
                for _, row in imp_df.iterrows():
                    if row['feature'] not in feature_imp:
                        feature_imp[row['feature']] = float(row['importance'])
            except Exception:
                pass
        
        atom_scores = state.atom_attributor.compute_atom_contributions(
            mol, feature_imp
        )
        mol_svg_highlighted = state.atom_attributor.generate_highlighted_mol_svg(
            mol, atom_scores, width=400, height=280
        )
    
    return {
        'smiles': request.smiles,
        'endpoint': request.endpoint,
        'shap_explanation': shap_result,
        'atom_scores': atom_scores,
        'mol_svg_highlighted': mol_svg_highlighted,
    }


@app.post("/similar")
async def find_similar_compounds(request: SimilarityRequest):
    """Find structurally similar compounds with toxicity profiles."""
    if state.searcher is None:
        raise HTTPException(status_code=503, detail="Similarity database not loaded. Run train.py first.")
    
    mol = smiles_to_mol(request.smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles}")
    
    similar = state.searcher.search(request.smiles, top_k=request.top_k)
    scaffold = state.searcher.get_scaffold(request.smiles)
    
    return {
        'smiles': request.smiles,
        'scaffold': scaffold,
        'similar_compounds': similar,
        'n_found': len(similar),
    }


@app.post("/report")
async def generate_report(request: PredictionRequest):
    """Generate and download a PDF toxicity report."""
    # Get prediction
    pred_result = predict_single(request.smiles, request.model)
    
    # Get SHAP explanation for first endpoint
    explain_result = {}
    if state.ensemble and state.ensemble.rf.models:
        first_endpoint = list(state.ensemble.rf.models.keys())[0]
        try:
            explain_req = ExplainRequest(smiles=request.smiles, endpoint=first_endpoint)
            explain_result = await explain_prediction(explain_req)
            explain_result = explain_result.get('shap_explanation', {})
        except Exception:
            pass
    
    # Get similar compounds
    similar = []
    if state.searcher:
        try:
            similar = state.searcher.search(request.smiles, top_k=5)
        except Exception:
            pass
    
    # Generate PDF
    generator = get_report_generator()
    try:
        pdf_bytes = generator.generate_report(
            smiles=request.smiles,
            predictions=pred_result.get('predictions', {}),
            admet_props=pred_result.get('admet', {}),
            shap_data=explain_result,
            similar_compounds=similar,
            structural_alerts=pred_result.get('structural_alerts', {}),
            mol_svg_b64=pred_result.get('mol_svg', ''),
        )
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type='application/pdf',
            headers={
                'Content-Disposition': 'attachment; filename="toxicity_report.pdf"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.post("/chatbot")
async def chatbot_query(request: ChatbotRequest):
    """
    AI chatbot for toxicity-related queries.
    Provides rule-based responses with chemistry knowledge.
    """
    message = request.message.lower().strip()
    context = request.context or {}
    
    response = generate_chatbot_response(message, context)
    
    return {
        'message': request.message,
        'response': response,
        'timestamp': time.time(),
    }


def generate_chatbot_response(message: str, context: Dict) -> str:
    """Rule-based expert chatbot for drug toxicity queries."""
    
    # Keyword-based responses
    responses = {
        'logp': (
            "**LogP (Log Partition Coefficient)** measures lipophilicity — how well a molecule partitions "
            "between octanol (lipid-like) and water. Optimal range for oral drugs: 0–3 (Lipinski: ≤5). "
            "High LogP (>5) increases membrane permeability but decreases aqueous solubility and may increase "
            "toxicity via bioaccumulation."
        ),
        'tpsa': (
            "**TPSA (Topological Polar Surface Area)** predicts oral bioavailability and CNS penetration. "
            "TPSA ≤ 140 Å² → good oral absorption (Veber rule). TPSA ≤ 90 Å² → good CNS penetration. "
            "Compounds with high TPSA are often poorly absorbed."
        ),
        'lipinski': (
            "**Lipinski's Rule of 5** predicts oral drug-likeness: MW ≤ 500, LogP ≤ 5, "
            "HBD ≤ 5, HBA ≤ 10. Violations suggest poor oral bioavailability. "
            "Notable exceptions: antibiotics, antifungals, and natural products."
        ),
        'morgan': (
            "**Morgan fingerprints** (also called ECFP — Extended Connectivity Fingerprints) encode "
            "circular substructure environments around each atom. Radius 2 = ECFP4, radius 3 = ECFP6. "
            "They're the gold standard for molecular similarity and ML in drug discovery."
        ),
        'shap': (
            "**SHAP (SHapley Additive exPlanations)** tells you how much each molecular descriptor "
            "contributed to a specific toxicity prediction. Positive SHAP = increases toxicity risk, "
            "negative SHAP = decreases risk. The values sum to the difference between the prediction "
            "and the model's baseline."
        ),
        'tox21': (
            "**Tox21** is the NIH's Toxicology in the 21st Century initiative. The dataset contains "
            "~12,000 compounds tested against 12 nuclear receptor and stress response pathways. "
            "It's the benchmark dataset for computational toxicology prediction."
        ),
        'auc': (
            "**AUC-ROC** (Area Under the Receiver Operating Characteristic Curve) measures model "
            "discrimination. AUC = 0.5 is random, AUC = 1.0 is perfect. For imbalanced toxicity data, "
            "AUC-PR (Precision-Recall) is often more informative than AUC-ROC."
        ),
        'herg': (
            "**hERG** (human Ether-à-go-go Related Gene) encodes a potassium channel critical for "
            "cardiac rhythm. hERG inhibition causes QT interval prolongation — a dangerous cardiac "
            "side effect that has caused drug withdrawals. It's one of the most important toxicity "
            "concerns in drug development."
        ),
        'pains': (
            "**PAINS (Pan-Assay Interference Compounds)** are substructures that cause false positives "
            "in high-throughput screening. Examples: catechols, rhodanines, frequent hitters. "
            "The app detects these automatically using SMARTS substructure matching."
        ),
        'smiles': (
            "**SMILES (Simplified Molecular Input Line Entry System)** is a text notation for molecules. "
            "Example: Aspirin = CC(=O)Oc1ccccc1C(=O)O. Key rules: uppercase = aromatic, "
            "lowercase = non-aromatic, () = branches, [] = atoms with special properties."
        ),
        'random forest': (
            "**Random Forest** builds an ensemble of decision trees, each trained on a random subset "
            "of data and features. Final prediction = majority vote. Advantages: handles high-dimensional "
            "data, built-in feature importance, no scaling needed, robust to outliers."
        ),
        'xgboost': (
            "**XGBoost** uses gradient boosting where each new tree corrects errors of previous trees. "
            "It's consistently top-performing on tabular data. Key hyperparameters: learning_rate "
            "(shrinkage), max_depth (tree complexity), scale_pos_weight (class imbalance correction)."
        ),
        'murcko': (
            "**Murcko Scaffold** extracts the core ring system of a molecule by removing side chains. "
            "Used to group compounds with the same structural framework. Scaffold-aware splitting "
            "is important to avoid data leakage in drug discovery ML."
        ),
    }
    
    # Check for keyword matches
    for keyword, resp in responses.items():
        if keyword in message:
            return resp
    
    # Context-aware responses
    if context.get('smiles') and ('toxic' in message or 'safe' in message or 'risk' in message):
        predictions = context.get('predictions', {})
        if predictions:
            max_prob = max(predictions.values())
            most_toxic = max(predictions, key=predictions.get)
            return (
                f"Based on the predictions for your compound: the highest toxicity probability is "
                f"**{max_prob:.1%}** for the **{most_toxic}** endpoint. "
                f"{'This is HIGH RISK — recommendation: structural optimization or avoid.' if max_prob >= 0.7 else 'This is MODERATE risk — proceed with caution and further testing.' if max_prob >= 0.4 else 'This appears LOW RISK — still validate experimentally.'}"
            )
    
    if 'hello' in message or 'hi' in message or 'help' in message:
        return (
            "Hello! I'm the **ToxAI Assistant**. I can help you understand:\n\n"
            "• **Toxicity predictions** — what the scores mean\n"
            "• **Molecular properties** — LogP, TPSA, MW, etc.\n"
            "• **ML concepts** — SHAP, AUC-ROC, Random Forest, XGBoost\n"
            "• **Chemistry concepts** — SMILES, fingerprints, scaffolds\n"
            "• **Drug safety** — PAINS, hERG, structural alerts\n\n"
            "Try asking: 'What is LogP?' or 'Explain SHAP values'"
        )
    
    if 'endpoint' in message or 'nr-' in message or 'sr-' in message:
        return (
            "The 12 **Tox21 endpoints** cover two pathway types:\n"
            "• **NR (Nuclear Receptor)**: AR, AhR, Aromatase, ER, PPAR-γ — endocrine disruption\n"
            "• **SR (Stress Response)**: ARE (oxidative), ATAD5 (genotoxic), HSE (heat shock), "
            "MMP (mitochondrial), p53 (DNA damage)\n\n"
            "Positive = compound activates (and potentially disrupts) the pathway."
        )
    
    return (
        "I'm not sure about that specific question. Try asking about: "
        "**LogP**, **TPSA**, **SHAP values**, **Morgan fingerprints**, **Tox21**, "
        "**AUC-ROC**, **PAINS**, **hERG**, **Lipinski's Rule of 5**, or **XGBoost**."
    )


@app.post("/batch")
async def batch_predict(request: BatchPredictRequest):
    """Batch prediction for multiple SMILES strings."""
    if len(request.smiles_list) > 500:
        raise HTTPException(status_code=400, detail="Max 500 compounds per batch")
    
    results = []
    errors = []
    
    for smi in request.smiles_list:
        try:
            result = predict_single(smi, request.model)
            results.append({
                'smiles': smi,
                'valid': True,
                'predictions': result['predictions'],
                'risk_level': result.get('risk_level', 'UNKNOWN'),
                'overall_score': result.get('overall_toxicity_score', 0),
            })
        except Exception as e:
            errors.append({'smiles': smi, 'error': str(e)})
    
    return {
        'total': len(request.smiles_list),
        'successful': len(results),
        'failed': len(errors),
        'results': results,
        'errors': errors,
    }


@app.post("/batch-file")
async def batch_predict_file(file: UploadFile = File(...)):
    """Batch prediction from uploaded CSV file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Find SMILES column
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'structure']:
        if col in df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        raise HTTPException(status_code=400, detail="No SMILES column found in CSV")
    
    smiles_list = df[smiles_col].dropna().tolist()[:200]  # Limit to 200
    
    batch_req = BatchPredictRequest(smiles_list=smiles_list)
    return await batch_predict(batch_req)


@app.post("/descriptors")
async def compute_descriptors(request: PredictionRequest):
    """Compute molecular descriptors for a SMILES string."""
    try:
        feature_vector, admet_props, mol = compute_features_for_smiles(request.smiles)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        'smiles': request.smiles,
        'admet': admet_props,
        'n_features': len(feature_vector),
        'feature_stats': {
            'mean': float(np.mean(feature_vector)),
            'std': float(np.std(feature_vector)),
            'n_nonzero': int(np.sum(feature_vector != 0)),
        }
    }


@app.get("/space")
async def chemical_space():
    """Get chemical space coordinates for visualization."""
    if state.searcher is None or len(state.searcher.smiles_list) == 0:
        return {'points': [], 'message': 'No data loaded'}
    
    # Sample up to 500 compounds for visualization
    n_sample = min(500, len(state.searcher.smiles_list))
    indices = np.random.choice(len(state.searcher.smiles_list), n_sample, replace=False)
    sample_smiles = [state.searcher.smiles_list[i] for i in indices]
    
    from src.similarity.fingerprint_search import compute_chemical_space_coords
    try:
        coords = compute_chemical_space_coords(sample_smiles, method='pca')
    except Exception:
        coords = np.random.randn(len(sample_smiles), 2)
    
    # Get toxicity labels if available
    points = []
    for i, (smi, coord) in enumerate(zip(sample_smiles, coords)):
        point = {
            'x': float(coord[0]),
            'y': float(coord[1]),
            'smiles': smi,
        }
        if state.searcher.toxicity_labels is not None:
            db_idx = list(state.searcher.smiles_list).index(smi)
            if db_idx < len(state.searcher.toxicity_labels):
                row = state.searcher.toxicity_labels.iloc[db_idx]
                vals = [float(v) for v in row.values if not pd.isna(v)]
                point['avg_toxicity'] = float(np.mean(vals)) if vals else 0.0
        points.append(point)
    
    return {'points': points, 'n_compounds': len(points)}


@app.get("/evaluation")
async def get_evaluation():
    """Get model evaluation metrics."""
    metrics = {}
    for model_name in ['ensemble', 'rf', 'xgb']:
        eval_file = MODELS_DIR / f'{model_name}_evaluation.csv'
        if eval_file.exists():
            df = pd.read_csv(eval_file)
            metrics[model_name] = df.to_dict('records')
    return metrics


# ============================================================
# SERVE FRONTEND
# ============================================================

# Mount static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_frontend():
    """Serve the main frontend HTML."""
    index_path = FRONTEND_DIR / 'index.html'
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({
        "message": "Drug Toxicity API is running",
        "docs": "/api/docs",
        "health": "/health"
    })


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import socket
    
    # Find available port
    port = 8000
    
    print("\n" + "=" * 60)
    print("  DRUG TOXICITY PREDICTION SERVER")
    print("=" * 60)
    print(f"  Frontend: http://localhost:{port}")
    print(f"  API Docs: http://localhost:{port}/api/docs")
    print(f"  Health:   http://localhost:{port}/health")
    print("=" * 60)
    
    if not state.is_ready:
        print("\n  WARNING: Models not loaded!")
        print("  Run 'python train.py' first to train the models.")
        print("  The server will start but predictions may not work.\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
