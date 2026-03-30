"""
Advanced Explainable AI Module for Drug Toxicity Prediction
Implements SHAP, LIME, counterfactual explanations, and molecular highlighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Explainability libraries
import shap
import lime
import lime.lime_tabular

# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDraw2D
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem

# Machine learning
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
from .config import FEATURE_IMPORTANCE_CONFIG, SIMILARITY_CONFIG

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Advanced SHAP explanations for molecular toxicity prediction
    """
    
    def __init__(self, model, X_background: pd.DataFrame, 
                 feature_names: List[str] = None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained machine learning model
            X_background: Background dataset for SHAP values
            feature_names: List of feature names
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names or X_background.columns.tolist()
        self.explainer = None
        self.shap_values = None
        
        # Initialize appropriate SHAP explainer based on model type
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer"""
        try:
            # Try TreeExplainer first (works for tree-based models)
            if hasattr(self.model, 'feature_importances_') or \
               hasattr(self.model, 'get_booster') or \
               'xgboost' in str(type(self.model)).lower() or \
               'lightgbm' in str(type(self.model)).lower() or \
               'catboost' in str(type(self.model)).lower():
                self.explainer = shap.TreeExplainer(self.model, self.X_background)
                logger.info("Using TreeExplainer for SHAP")
            
            # Try KernelExplainer for other models
            else:
                # Use a smaller background dataset for KernelExplainer
                background_sample = self.X_background.sample(
                    n=min(100, len(self.X_background)), random_state=42
                )
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, background_sample
                )
                logger.info("Using KernelExplainer for SHAP")
                
        except Exception as e:
            logger.warning(f"Failed to initialize TreeExplainer: {e}")
            # Fallback to KernelExplainer
            background_sample = self.X_background.sample(
                n=min(100, len(self.X_background)), random_state=42
            )
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background_sample
            )
            logger.info("Using fallback KernelExplainer for SHAP")
    
    def explain_instance(self, X_instance: pd.DataFrame, 
                        class_index: int = 1) -> Dict:
        """
        Explain a single instance using SHAP
        
        Args:
            X_instance: Single instance to explain
            class_index: Class index to explain (default: 1 for toxic)
            
        Returns:
            Dictionary with SHAP explanations
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_instance)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[class_index]
        
        # Create explanation dictionary
        explanation = {
            "shap_values": shap_values[0] if len(shap_values) == 1 else shap_values,
            "feature_names": self.feature_names,
            "base_value": self.explainer.expected_value[class_index] if \
                         isinstance(self.explainer.expected_value, list) else \
                         self.explainer.expected_value,
            "prediction": self.model.predict_proba(X_instance)[0][class_index],
            "feature_importance": self._get_feature_importance(shap_values[0])
        }
        
        return explanation
    
    def explain_batch(self, X_batch: pd.DataFrame, 
                     class_index: int = 1) -> Dict:
        """
        Explain a batch of instances
        
        Args:
            X_batch: Batch of instances to explain
            class_index: Class index to explain
            
        Returns:
            Dictionary with batch SHAP explanations
        """
        logger.info(f"Calculating SHAP values for {len(X_batch)} instances")
        
        # Calculate SHAP values for the batch
        shap_values = self.explainer.shap_values(X_batch)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[class_index]
        
        # Calculate mean absolute SHAP values for global importance
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        return {
            "shap_values": shap_values,
            "feature_names": self.feature_names,
            "mean_shap_values": mean_shap,
            "feature_importance": feature_importance,
            "base_value": self.explainer.expected_value[class_index] if \
                         isinstance(self.explainer.expected_value, list) else \
                         self.explainer.expected_value
        }
    
    def _get_feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Get feature importance from SHAP values"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values,
            'abs_shap_value': np.abs(shap_values)
        }).sort_values('abs_shap_value', ascending=False)
        
        return importance_df
    
    def create_summary_plot(self, X_data: pd.DataFrame, 
                           max_display: int = 20,
                           plot_type: str = "bar") -> go.Figure:
        """
        Create SHAP summary plot
        
        Args:
            X_data: Data to plot
            max_display: Maximum number of features to display
            plot_type: Type of plot ('bar', 'dot', 'violin')
            
        Returns:
            Plotly figure
        """
        # Calculate SHAP values
        shap_explanations = self.explain_batch(X_data)
        shap_values = shap_explanations["shap_values"]
        
        # Select top features
        top_indices = np.argsort(np.mean(np.abs(shap_values), axis=0))[-max_display:]
        top_shap_values = shap_values[:, top_indices]
        top_features = [self.feature_names[i] for i in top_indices]
        
        if plot_type == "bar":
            # Create bar plot of mean absolute SHAP values
            mean_shap = np.mean(np.abs(top_shap_values), axis=0)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=mean_shap,
                    y=top_features,
                    orientation='h',
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="SHAP Feature Importance (Mean Absolute Value)",
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="Features",
                height=max(400, len(top_features) * 25)
            )
        
        elif plot_type == "dot":
            # Create dot plot (beeswarm)
            fig = go.Figure()
            
            for i, feature in enumerate(top_features):
                fig.add_trace(go.Scatter(
                    x=top_shap_values[:, i],
                    y=[feature] * len(top_shap_values),
                    mode='markers',
                    name=feature,
                    opacity=0.6
                ))
            
            fig.update_layout(
                title="SHAP Values Distribution",
                xaxis_title="SHAP Value",
                yaxis_title="Features",
                height=max(400, len(top_features) * 25),
                showlegend=False
            )
        
        return fig
    
    def create_waterfall_plot(self, X_instance: pd.DataFrame,
                             max_features: int = 15) -> go.Figure:
        """
        Create SHAP waterfall plot for a single instance
        
        Args:
            X_instance: Single instance to explain
            max_features: Maximum number of features to show
            
        Returns:
            Plotly figure
        """
        # Get SHAP explanation
        explanation = self.explain_instance(X_instance)
        shap_values = explanation["shap_values"]
        base_value = explanation["base_value"]
        
        # Select top features
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-max_features:]
        
        # Prepare data for waterfall plot
        features = [self.feature_names[i] for i in top_indices]
        values = shap_values[top_indices]
        
        # Calculate cumulative values
        cumulative = np.zeros(len(values) + 1)
        cumulative[0] = base_value
        for i in range(len(values)):
            cumulative[i + 1] = cumulative[i] + values[i]
        
        # Create waterfall plot
        fig = go.Figure()
        
        # Add base value
        fig.add_trace(go.Scatter(
            x=[0], 
            y=[base_value],
            mode='markers',
            name='Base Value',
            marker=dict(color='blue', size=10)
        ))
        
        # Add feature contributions
        colors = ['red' if v < 0 else 'green' for v in values]
        
        for i, (feature, value, color) in enumerate(zip(features, values, colors)):
            fig.add_trace(go.Scatter(
                x=[i + 1, i + 1],
                y=[cumulative[i], cumulative[i + 1]],
                mode='lines',
                line=dict(color=color, width=2),
                name=feature,
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=[i + 1],
                y=[cumulative[i + 1]],
                mode='markers',
                marker=dict(color=color, size=8),
                name=feature,
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"SHAP Waterfall Plot (Prediction: {explanation['prediction']:.3f})",
            xaxis_title="Features",
            yaxis_title="Prediction Value",
            height=500
        )
        
        # Update x-axis labels
        fig.update_xaxes(
            ticktext=['Base'] + features,
            tickvals=list(range(len(features) + 1)),
            tickangle=45
        )
        
        return fig


class LIMEExplainer:
    """
    LIME explanations for molecular toxicity prediction
    """
    
    def __init__(self, X_train: pd.DataFrame, 
                 feature_names: List[str] = None,
                 mode: str = "classification"):
        """
        Initialize LIME explainer
        
        Args:
            X_train: Training data
            feature_names: List of feature names
            mode: 'classification' or 'regression'
        """
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.mode = mode
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            mode=mode,
            discretize_continuous=True,
            random_state=42
        )
    
    def explain_instance(self, model, X_instance: pd.DataFrame,
                        num_features: int = 10,
                        class_index: int = 1) -> Dict:
        """
        Explain a single instance using LIME
        
        Args:
            model: Trained model
            X_instance: Instance to explain
            num_features: Number of features to include in explanation
            class_index: Class index to explain
            
        Returns:
            Dictionary with LIME explanation
        """
        # Get prediction function
        if self.mode == "classification":
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            data_row=X_instance.values[0],
            predict_fn=predict_fn,
            num_features=num_features,
            labels=[class_index]
        )
        
        # Extract feature contributions
        feature_contributions = explanation.as_list(label=class_index)
        
        return {
            "feature_contributions": feature_contributions,
            "intercept": explanation.intercept[class_index],
            "prediction": model.predict_proba(X_instance)[0][class_index],
            "local_pred": explanation.local_pred[class_index],
            "score": explanation.score
        }
    
    def create_lime_plot(self, explanation: Dict) -> go.Figure:
        """
        Create LIME explanation plot
        
        Args:
            explanation: LIME explanation dictionary
            
        Returns:
            Plotly figure
        """
        feature_contributions = explanation["feature_contributions"]
        
        features = [feat for feat, _ in feature_contributions]
        contributions = [contrib for _, contrib in feature_contributions]
        
        colors = ['red' if contrib < 0 else 'green' for contrib in contributions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=contributions,
                y=features,
                orientation='h',
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title=f"LIME Explanation (Prediction: {explanation['prediction']:.3f})",
            xaxis_title="Feature Contribution",
            yaxis_title="Features",
            height=max(400, len(features) * 30)
        )
        
        return fig


class MolecularHighlighter:
    """
    Advanced molecular structure highlighting based on feature importance
    """
    
    def __init__(self):
        self.color_map = {
            "high_importance": (1.0, 0.0, 0.0),  # Red
            "medium_importance": (1.0, 1.0, 0.0),  # Yellow
            "low_importance": (0.0, 1.0, 0.0),  # Green
            "negative_importance": (0.0, 0.0, 1.0)  # Blue
        }
    
    def highlight_molecule_by_atoms(self, smiles: str, 
                                   atom_importance: Dict[int, float],
                                   importance_thresholds: Tuple[float, float] = (0.3, 0.7)) -> str:
        """
        Highlight molecule atoms based on importance values
        
        Args:
            smiles: SMILES string of the molecule
            atom_importance: Dictionary mapping atom indices to importance values
            importance_thresholds: Thresholds for low/medium/high importance
            
        Returns:
            Path to the highlighted molecule image
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Normalize importance values to [0, 1]
        if atom_importance:
            importance_values = list(atom_importance.values())
            min_val, max_val = min(importance_values), max(importance_values)
            if max_val > min_val:
                normalized_importance = {
                    idx: (val - min_val) / (max_val - min_val)
                    for idx, val in atom_importance.items()
                }
            else:
                normalized_importance = {idx: 0.5 for idx in atom_importance}
        else:
            normalized_importance = {}
        
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
        drawer.SetFontSize(0.8)
        
        # Prepare atom colors
        atom_colors = {}
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx in normalized_importance:
                importance = normalized_importance[atom_idx]
                if importance < importance_thresholds[0]:
                    color = self.color_map["low_importance"]
                elif importance < importance_thresholds[1]:
                    color = self.color_map["medium_importance"]
                else:
                    color = self.color_map["high_importance"]
                atom_colors[atom_idx] = color
        
        # Draw molecule with highlighting
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol, highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors
        )
        drawer.FinishDrawing()
        
        # Save image
        output_path = f"highlighted_molecule_{hash(smiles)}.png"
        drawer.WriteToFile(output_path)
        
        return output_path
    
    def highlight_molecule_by_substructures(self, smiles: str,
                                          substructure_importance: Dict[str, float]) -> str:
        """
        Highlight molecule substructures based on importance
        
        Args:
            smiles: SMILES string
            substructure_importance: Dictionary mapping SMARTS patterns to importance
            
        Returns:
            Path to highlighted molecule image
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Find substructures
        substructure_atoms = {}
        for smarts, importance in substructure_importance.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    for atom_idx in match:
                        if atom_idx not in substructure_atoms:
                            substructure_atoms[atom_idx] = importance
                        else:
                            substructure_atoms[atom_idx] = max(
                                substructure_atoms[atom_idx], importance
                            )
        
        return self.highlight_molecule_by_atoms(smiles, substructure_atoms)
    
    def create_importance_legend(self) -> go.Figure:
        """Create a legend for importance colors"""
        fig = go.Figure()
        
        labels = ["High Importance", "Medium Importance", "Low Importance", "Negative"]
        colors = ["red", "yellow", "green", "blue"]
        
        for label, color in zip(labels, colors):
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(color=color, size=20),
                name=label
            ))
        
        fig.update_layout(
            title="Importance Color Legend",
            showlegend=True,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=200
        )
        
        return fig


class CounterfactualExplainer:
    """
    Generate counterfactual explanations for toxicity predictions
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize counterfactual explainer
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
    
    def generate_counterfactual(self, X_instance: pd.DataFrame,
                              target_class: int = 0,
                              max_iterations: int = 100,
                              learning_rate: float = 0.01) -> Dict:
        """
        Generate counterfactual explanation
        
        Args:
            X_instance: Original instance
            target_class: Target class for counterfactual
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            
        Returns:
            Counterfactual explanation
        """
        # Convert to numpy array
        original = X_instance.values[0].copy()
        counterfactual = original.copy()
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Get current prediction
            current_pred = self.model.predict_proba(counterfactual.reshape(1, -1))[0]
            
            # Check if target reached
            if np.argmax(current_pred) == target_class:
                break
            
            # Calculate gradient (simplified)
            epsilon = 1e-5
            gradients = np.zeros_like(counterfactual)
            
            for i in range(len(counterfactual)):
                perturbed = counterfactual.copy()
                perturbed[i] += epsilon
                
                perturbed_pred = self.model.predict_proba(perturbed.reshape(1, -1))[0]
                gradient = (perturbed_pred[target_class] - current_pred[target_class]) / epsilon
                gradients[i] = gradient
            
            # Update counterfactual
            counterfactual += learning_rate * gradients
            
            # Project back to valid feature space (simplified)
            counterfactual = np.clip(counterfactual, 0, 1)
        
        # Calculate feature changes
        feature_changes = counterfactual - original
        feature_importance = np.abs(feature_changes)
        
        return {
            "original_instance": original,
            "counterfactual_instance": counterfactual,
            "feature_changes": feature_changes,
            "feature_importance": feature_importance,
            "original_prediction": self.model.predict(original.reshape(1, -1))[0],
            "counterfactual_prediction": self.model.predict(counterfactual.reshape(1, -1))[0],
            "iterations": iteration + 1
        }
    
    def create_counterfactual_plot(self, explanation: Dict) -> go.Figure:
        """
        Create visualization of counterfactual changes
        
        Args:
            explanation: Counterfactual explanation
            
        Returns:
            Plotly figure
        """
        feature_changes = explanation["feature_changes"]
        feature_importance = explanation["feature_importance"]
        
        # Sort by importance
        sorted_indices = np.argsort(feature_importance)[::-1][:20]  # Top 20 features
        
        features = [self.feature_names[i] for i in sorted_indices]
        changes = [feature_changes[i] for i in sorted_indices]
        importance = [feature_importance[i] for i in sorted_indices]
        
        colors = ['red' if change > 0 else 'blue' for change in changes]
        
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=colors,
            name='Feature Change'
        ))
        
        fig.update_layout(
            title="Counterfactual Feature Changes",
            xaxis_title="Magnitude of Change",
            yaxis_title="Features",
            height=max(400, len(features) * 25)
        )
        
        return fig


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis
    """
    
    def __init__(self):
        self.methods = ["shap", "lime", "permutation", "intrinsic"]
    
    def analyze_importance(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame = None) -> Dict:
        """
        Comprehensive feature importance analysis
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            
        Returns:
            Dictionary with importance analyses
        """
        results = {}
        
        # Intrinsic feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            results["intrinsic"] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Permutation importance
        if X_test is not None:
            perm_importance = permutation_importance(
                model, X_test, y_train, n_repeats=10, random_state=42
            )
            results["permutation"] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
        
        return results
    
    def create_importance_comparison_plot(self, importance_results: Dict) -> go.Figure:
        """
        Create comparison plot of different importance methods
        
        Args:
            importance_results: Results from importance analysis
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(importance_results.keys()),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, (method, data) in enumerate(importance_results.items()):
            row = i // 2 + 1
            col = i % 2 + 1
            
            # Get top 10 features
            top_features = data.head(10)
            
            fig.add_trace(
                go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    name=method,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Feature Importance Comparison",
            height=800,
            showlegend=False
        )
        
        return fig


class ExplainabilityDashboard:
    """
    Main dashboard for explainability features
    """
    
    def __init__(self, model, X_background: pd.DataFrame, 
                 feature_names: List[str] = None):
        """
        Initialize explainability dashboard
        
        Args:
            model: Trained model
            X_background: Background dataset
            feature_names: Feature names
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names or X_background.columns.tolist()
        
        # Initialize explainers
        self.shap_explainer = SHAPExplainer(model, X_background, feature_names)
        self.lime_explainer = LIMEExplainer(X_background, feature_names)
        self.molecular_highlighter = MolecularHighlighter()
        self.counterfactual_explainer = CounterfactualExplainer(model, feature_names)
        self.importance_analyzer = FeatureImportanceAnalyzer()
    
    def explain_molecule(self, smiles: str, X_instance: pd.DataFrame,
                        include_shap: bool = True,
                        include_lime: bool = True,
                        include_counterfactual: bool = True) -> Dict:
        """
        Comprehensive explanation for a single molecule
        
        Args:
            smiles: SMILES string
            X_instance: Feature representation of the molecule
            include_shap: Include SHAP explanations
            include_lime: Include LIME explanations
            include_counterfactual: Include counterfactual explanations
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanation = {
            "smiles": smiles,
            "prediction": self.model.predict_proba(X_instance)[0][1],
            "predicted_class": self.model.predict(X_instance)[0]
        }
        
        # SHAP explanation
        if include_shap:
            explanation["shap"] = self.shap_explainer.explain_instance(X_instance)
        
        # LIME explanation
        if include_lime:
            explanation["lime"] = self.lime_explainer.explain_instance(
                self.model, X_instance
            )
        
        # Counterfactual explanation
        if include_counterfactual:
            explanation["counterfactual"] = self.counterfactual_explainer.generate_counterfactual(
                X_instance
            )
        
        return explanation
    
    def create_comprehensive_report(self, explanation: Dict) -> str:
        """
        Create a comprehensive explanation report
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            Formatted report string
        """
        report = f"""
# Molecular Toxicity Explanation Report

## Molecule Information
- **SMILES**: {explanation['smiles']}
- **Predicted Toxicity**: {explanation['prediction']:.3f}
- **Predicted Class**: {'Toxic' if explanation['predicted_class'] == 1 else 'Non-Toxic'}

## SHAP Explanation
"""
        
        if "shap" in explanation:
            shap_data = explanation["shap"]
            top_features = shap_data["feature_importance"].head(10)
            
            for _, row in top_features.iterrows():
                direction = "increases" if row["shap_value"] > 0 else "decreases"
                report += f"- **{row['feature']}**: {direction} toxicity risk (SHAP: {row['shap_value']:.4f})\n"
        
        report += "\n## LIME Explanation\n"
        
        if "lime" in explanation:
            lime_data = explanation["lime"]
            for feature, contribution in lime_data["feature_contributions"][:10]:
                direction = "increases" if contribution > 0 else "decreases"
                report += f"- **{feature}**: {direction} toxicity risk (LIME: {contribution:.4f})\n"
        
        report += "\n## Counterfactual Analysis\n"
        
        if "counterfactual" in explanation:
            cf_data = explanation["counterfactual"]
            report += f"- **Original Prediction**: {cf_data['original_prediction']}\n"
            report += f"- **Counterfactual Prediction**: {cf_data['counterfactual_prediction']}\n"
            report += f"- **Iterations Required**: {cf_data['iterations']}\n"
        
        return report
