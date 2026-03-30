"""
Advanced Machine Learning Models for Drug Toxicity Prediction
Implements ensemble methods, deep learning, and uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import pickle
import json
import logging
from pathlib import Path
from tqdm import tqdm
import warnings

# Classical ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Advanced ensemble models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Graph Neural Networks
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader as GeoDataLoader
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    logging.warning("PyTorch Geometric not available. GNN models will be disabled.")

# Uncertainty quantification
from scipy import stats
from sklearn.utils import resample

# Hyperparameter optimization
import optuna

# Configuration
from .config import MODEL_CONFIG, VALIDATION_CONFIG, PERFORMANCE_CONFIG, ADVANCED_FEATURES

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugToxicityDataset(Dataset):
    """Custom PyTorch dataset for drug toxicity prediction"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MolecularGraphDataset(Dataset):
    """Dataset for graph neural networks with molecular graphs"""
    
    def __init__(self, graphs: List[Data], labels: np.ndarray):
        self.graphs = graphs
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


class MLPModel(nn.Module):
    """Multi-layer perceptron for toxicity prediction"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.3):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for molecular property prediction"""
    
    def __init__(self, num_features: int, hidden_channels: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2):
        super(GraphNeuralNetwork, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()


class EnsembleModel:
    """
    Advanced ensemble model combining multiple algorithms
    """
    
    def __init__(self, model_config: Dict = None):
        self.model_config = model_config or MODEL_CONFIG["ensemble_models"]
        self.models = {}
        self.ensemble = None
        self.is_fitted = False
        self.feature_names = None
        self.calibrated = False
        
    def train_individual_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Train individual models and collect performance metrics
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of model performances
        """
        logger.info("Training individual ensemble models")
        
        self.feature_names = X_train.columns.tolist()
        performances = {}
        
        # Train XGBoost
        logger.info("Training XGBoost model")
        xgb_model = xgb.XGBClassifier(**self.model_config["xgboost"])
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=50, verbose=False)
        self.models["xgboost"] = xgb_model
        performances["xgboost"] = self._evaluate_model(xgb_model, X_val, y_val)
        
        # Train LightGBM
        logger.info("Training LightGBM model")
        lgb_model = lgb.LGBMClassifier(**self.model_config["lightgbm"])
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        self.models["lightgbm"] = lgb_model
        performances["lightgbm"] = self._evaluate_model(lgb_model, X_val, y_val)
        
        # Train CatBoost
        logger.info("Training CatBoost model")
        cat_model = cb.CatBoostClassifier(**self.model_config["catboost"])
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), 
                     early_stopping_rounds=50, verbose=False)
        self.models["catboost"] = cat_model
        performances["catboost"] = self._evaluate_model(cat_model, X_val, y_val)
        
        # Train Random Forest
        logger.info("Training Random Forest model")
        rf_model = RandomForestClassifier(**self.model_config["random_forest"])
        rf_model.fit(X_train, y_train)
        self.models["random_forest"] = rf_model
        performances["random_forest"] = self._evaluate_model(rf_model, X_val, y_val)
        
        self.is_fitted = True
        logger.info("All individual models trained successfully")
        
        return performances
    
    def create_voting_ensemble(self, voting: str = "soft", weights: Optional[List[float]] = None):
        """
        Create voting ensemble from trained models
        
        Args:
            voting: Voting method ('hard' or 'soft')
            weights: Weights for each model in the ensemble
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before creating ensemble")
        
        logger.info(f"Creating {voting} voting ensemble")
        
        # Prepare estimators for voting
        estimators = [(name, model) for name, model in self.models.items()]
        
        # Create voting classifier
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
        
        # Fit the ensemble on all training data
        all_models = list(self.models.values())
        if len(all_models) > 0:
            # Use the first model's training data
            X_train = np.column_stack([model.predict_proba(self.feature_names)[:, 1] 
                                     for model in all_models])
            # This is a simplified approach - in practice, you'd need the original training data
            logger.warning("Simplified ensemble fitting - consider retraining with full data")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if self.ensemble is not None:
            return self.ensemble.predict(X)
        else:
            # Use weighted average of individual predictions
            predictions = []
            for model in self.models.values():
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)
            
            # Average predictions
            ensemble_pred = np.mean(predictions, axis=0)
            return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if self.ensemble is not None:
            return self.ensemble.predict_proba(X)
        else:
            # Average probabilities from individual models
            probabilities = []
            for model in self.models.values():
                prob = model.predict_proba(X)[:, 1]
                probabilities.append(prob)
            
            ensemble_prob = np.mean(probabilities, axis=0)
            return np.column_stack([1 - ensemble_prob, ensemble_prob])
    
    def _evaluate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Evaluate individual model performance"""
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        return {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average='weighted'),
            "recall": recall_score(y_val, y_pred, average='weighted'),
            "f1": f1_score(y_val, y_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_val, y_prob)
        }
    
    def get_feature_importance(self, model_name: str = "xgboost") -> pd.DataFrame:
        """Get feature importance from specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'get_booster'):
            # XGBoost
            importance = model.get_booster().get_score(importance_type='weight')
            importance = [importance.get(f'f{i}', 0) for i in range(len(self.feature_names))]
        else:
            raise ValueError("Model does not support feature importance")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """Save the ensemble model"""
        model_data = {
            'models': self.models,
            'ensemble': self.ensemble,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'model_config': self.model_config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the ensemble model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.ensemble = model_data['ensemble']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        self.model_config = model_data['model_config']
        
        logger.info(f"Model loaded from {filepath}")


class DeepLearningTrainer:
    """
    Advanced deep learning trainer for toxicity prediction
    """
    
    def __init__(self, model_type: str = "mlp", config: Dict = None):
        self.model_type = model_type
        self.config = config or MODEL_CONFIG["deep_learning"][model_type]
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None
    
    def train_mlp(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Train Multi-layer Perceptron model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training history and metrics
        """
        logger.info("Training MLP model")
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create model
        input_dim = X_train.shape[1]
        self.model = MLPModel(input_dim, self.config["hidden_layers"], 
                            self.config["dropout"]).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = DrugToxicityDataset(X_train_scaled, y_train)
        val_dataset = DrugToxicityDataset(X_val_scaled, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        best_auc = 0
        
        for epoch in range(self.config["epochs"]):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            val_auc = roc_auc_score(all_labels, all_preds)
            
            # Update history
            history["train_loss"].append(train_loss / len(train_loader))
            history["val_loss"].append(val_loss / len(val_loader))
            history["val_auc"].append(val_auc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), "best_mlp_model.pth")
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                          f"Val Loss = {val_loss/len(val_loader):.4f}, Val AUC = {val_auc:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load("best_mlp_model.pth"))
        
        logger.info(f"MLP training completed. Best validation AUC: {best_auc:.4f}")
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the trained model
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Create dataset and dataloader
        dataset = DrugToxicityDataset(X, np.zeros(len(X)))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_preds.extend(preds)
                all_probs.extend(probs)
        
        return np.array(all_preds), np.array(all_probs)


class UncertaintyQuantifier:
    """
    Advanced uncertainty quantification for model predictions
    """
    
    def __init__(self, method: str = "monte_carlo", n_samples: int = 100):
        self.method = method
        self.n_samples = n_samples
        self.uncertainty_estimates = None
    
    def monte_carlo_dropout(self, model: nn.Module, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout for uncertainty estimation
        
        Args:
            model: Neural network model with dropout layers
            X: Input features
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        model.train()  # Enable dropout during inference
        
        predictions = []
        
        for _ in range(self.n_samples):
            with torch.no_grad():
                if isinstance(X, pd.DataFrame):
                    X_tensor = torch.FloatTensor(X.values).to(model.device)
                else:
                    X_tensor = torch.FloatTensor(X).to(model.device)
                
                pred = model(X_tensor).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def ensemble_uncertainty(self, models: Dict, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uncertainty estimation using ensemble disagreement
        
        Args:
            models: Dictionary of trained models
            X: Input features
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        predictions = []
        
        for name, model in models.items():
            try:
                pred = model.predict_proba(X)[:, 1]
                predictions.append(pred)
            except:
                # Fallback to predict if predict_proba not available
                pred = model.predict(X)
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    def bootstrap_uncertainty(self, model, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap uncertainty estimation
        
        Args:
            model: Trained model
            X: Input features
            y: Target labels
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        predictions = []
        
        for _ in range(self.n_samples):
            # Bootstrap sample
            X_boot, y_boot = resample(X, y, replace=True)
            
            # Train model on bootstrap sample
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_boot, y_boot)
            
            # Make predictions
            pred = model_copy.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna
    """
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.best_params = None
        self.study = None
    
    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         n_trials: int = 100) -> Dict:
        """
        Optimize XGBoost hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "random_state": 42
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=50, verbose=False)
            
            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')
        
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=n_trials)
        
        self.best_params = self.study.best_params
        logger.info(f"Best XGBoost parameters: {self.best_params}")
        
        return self.best_params
    
    def optimize_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               n_trials: int = 50) -> Dict:
        """
        Optimize neural network hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            # Suggest hyperparameters
            hidden_layers = []
            n_layers = trial.suggest_int("n_layers", 2, 4)
            for i in range(n_layers):
                hidden_layers.append(trial.suggest_int(f"layer_{i}_size", 64, 512))
            
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            
            # Create and train model
            input_dim = X_train.shape[1]
            model = MLPModel(input_dim, hidden_layers, dropout)
            
            # Training (simplified for optimization)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Quick training for evaluation
            train_dataset = DrugToxicityDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            model.train()
            for epoch in range(10):  # Quick training
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_dataset = DrugToxicityDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=32)
                
                all_preds = []
                all_labels = []
                
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    all_preds.extend(outputs.numpy())
                    all_labels.extend(batch_y.numpy())
                
                auc = roc_auc_score(all_labels, all_preds)
                return auc
        
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=n_trials)
        
        self.best_params = self.study.best_params
        logger.info(f"Best neural network parameters: {self.best_params}")
        
        return self.best_params


class ModelEvaluator:
    """
    Comprehensive model evaluation and validation
    """
    
    def __init__(self):
        self.metrics = PERFORMANCE_CONFIG["metrics"]
        self.cv_folds = VALIDATION_CONFIG["cv_folds"]
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform cross-validation on a model
        
        Args:
            model: Machine learning model
            X: Features
            y: Target
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {self.cv_folds}-fold cross-validation")
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                           random_state=VALIDATION_CONFIG["random_state"])
        
        results = {}
        
        for metric in self.metrics:
            if metric in ["accuracy", "precision", "recall", "f1"]:
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                results[metric] = {
                    "mean": scores.mean(),
                    "std": scores.std(),
                    "scores": scores.tolist()
                }
            elif metric == "roc_auc":
                scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
                results[metric] = {
                    "mean": scores.mean(),
                    "std": scores.std(),
                    "scores": scores.tolist()
                }
        
        return results
    
    def evaluate_comprehensive(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_prob: np.ndarray) -> Dict:
        """
        Comprehensive evaluation with multiple metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Basic metrics
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["precision"] = precision_score(y_true, y_pred, average='weighted')
        results["recall"] = recall_score(y_true, y_pred, average='weighted')
        results["f1"] = f1_score(y_true, y_pred, average='weighted')
        
        # ROC-AUC
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except:
            results["roc_auc"] = 0.0
        
        # Precision-Recall AUC
        try:
            from sklearn.metrics import average_precision_score
            results["pr_auc"] = average_precision_score(y_true, y_prob)
        except:
            results["pr_auc"] = 0.0
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        from sklearn.metrics import classification_report
        results["classification_report"] = classification_report(y_true, y_pred, 
                                                                 output_dict=True)
        
        return results
    
    def calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                         n_bins: int = 10) -> Dict:
        """
        Calculate calibration curve for probability predictions
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Calibration curve data
        """
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        return {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "n_bins": n_bins
        }


# Utility functions
def save_model_metadata(model: Any, filepath: str, metadata: Dict = None):
    """Save model with metadata"""
    model_data = {
        "model": model,
        "metadata": metadata or {},
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)


def load_model_with_metadata(filepath: str) -> Tuple[Any, Dict]:
    """Load model with metadata"""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data["model"], model_data["metadata"]


def create_model_report(evaluation_results: Dict, model_name: str) -> str:
    """Create a comprehensive model evaluation report"""
    report = f"""
# Model Evaluation Report: {model_name}

## Performance Metrics
"""
    
    for metric, value in evaluation_results.items():
        if isinstance(value, dict) and "mean" in value:
            report += f"- **{metric.upper()}**: {value['mean']:.4f} ± {value['std']:.4f}\n"
        else:
            report += f"- **{metric.upper()}**: {value:.4f}\n"
    
    report += "\n## Detailed Results\n"
    report += f"Generated at: {pd.Timestamp.now()}\n"
    
    return report
