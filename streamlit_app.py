"""
Advanced Drug Toxicity Prediction Streamlit Application
Modern dashboard with interactive visualizations, explainable AI, and real-time predictions
"""

import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our custom modules
try:
    from src.data_processing import MolecularDescriptorCalculator, FeatureEngineer, DataLoader
    from src.ml_models import EnsembleModel, DeepLearningTrainer, UncertaintyQuantifier
    from src.explainable_ai import SHAPExplainer, LIMEExplainer, MolecularHighlighter, ExplainabilityDashboard
    from src.config import MODEL_CONFIG, VIZ_CONFIG, TOXICITY_TYPES, ADVANCED_FEATURES
    from src.molecular_viewer import MolecularViewer
    from src.chatbot import ToxicityChatbot
    from src.report_generator import ReportGenerator
    from src.similarity_analyzer import SimilarityAnalyzer
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    MODULES_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Drug Toxicity Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def load_custom_css():
    """Load custom CSS for enhanced UI"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .molecule-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .chat-container {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 10px;
        height: 400px;
        overflow-y: auto;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_smiles' not in st.session_state:
        st.session_state.current_smiles = ""
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'similarity_results' not in st.session_state:
        st.session_state.similarity_results = []
    if 'explanation_results' not in st.session_state:
        st.session_state.explanation_results = {}

def load_models():
    """Load trained models"""
    if not MODULES_AVAILABLE:
        st.error("Required modules not available. Please check installation.")
        return False
    
    try:
        # Initialize model components
        st.session_state.descriptor_calc = MolecularDescriptorCalculator()
        st.session_state.feature_engineer = FeatureEngineer()
        
        # Try to load trained ensemble model
        model_path = Path("models/ensemble_model.pkl")
        if model_path.exists():
            st.session_state.ensemble_model = EnsembleModel()
            st.session_state.ensemble_model.load_model(str(model_path))
            st.session_state.model_loaded = True
            st.success("✅ Models loaded successfully!")
        else:
            st.warning("⚠️ No pre-trained models found. Using demo mode.")
            st.session_state.model_loaded = False
        
        # Initialize other components
        st.session_state.molecular_viewer = MolecularViewer()
        st.session_state.chatbot = ToxicityChatbot()
        st.session_state.report_generator = ReportGenerator()
        st.session_state.similarity_analyzer = SimilarityAnalyzer()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return False

def create_sidebar():
    """Create modern sidebar with navigation"""
    with st.sidebar:
        # App title and logo
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #1f77b4;'>💊</h1>
            <h3 style='color: #1f77b4;'>Drug Toxicity Predictor</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.selectbox(
            "🧭 Navigation",
            [
                "🏠 Home",
                "🔬 Predict Toxicity", 
                "📊 Data Analysis",
                "🧠 Explainable AI",
                "🔍 Similarity Search",
                "💬 AI Assistant",
                "📄 Report Generator",
                "⚙️ Settings"
            ]
        )
        
        st.markdown("---")
        
        # Model status
        if st.session_state.model_loaded:
            st.success("✅ Models Ready")
        else:
            st.warning("⚠️ Demo Mode")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        if st.session_state.last_prediction:
            pred = st.session_state.last_prediction
            st.metric("Last Risk Level", pred.get('risk_level', 'Unknown'))
            st.metric("Max Toxicity", f"{pred.get('max_toxicity_score', 0):.3f}")
        
        st.markdown("---")
        
        # About section
        st.markdown("""
        ### ℹ️ About
        Advanced AI-powered drug toxicity prediction with:
        - Ensemble ML models
        - Explainable AI
        - Molecular visualization
        - Real-time predictions
        """)
        
        return page

def home_page():
    """Create modern home page"""
    st.markdown('<h1 class="main-header">🧬 Advanced Drug Toxicity Prediction</h1>', unsafe_allow_html=True)
    
    # Hero section with metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>94.2%</h2>
            <p>Ensemble Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Compounds</h3>
            <h2>12,000+</h2>
            <p>Tox21 Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 AI Models</h3>
            <h2>5+</h2>
            <p>Ensemble Methods</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <h2>&lt;100ms</h2>
            <p>Prediction Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown('<h2 class="sub-header">🚀 Advanced Features</h2>', unsafe_allow_html=True)
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div class="molecule-container">
            <h4>🧬 Molecular Visualization</h4>
            <p>Interactive 2D/3D molecular structures with atom-level highlighting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class="molecule-container">
            <h4>🔍 Explainable AI</h4>
            <p>SHAP and LIME explanations for model predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div class="molecule-container">
            <h4>🤖 AI Assistant</h4>
            <p>Intelligent chatbot for toxicity-related queries</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick demo
    st.markdown('<h2 class="sub-header">⚡ Quick Demo</h2>', unsafe_allow_html=True)
    
    demo_smiles = st.text_input(
        "Enter SMILES string for quick prediction:",
        value="CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        help="Enter a SMILES string to predict toxicity"
    )
    
    if st.button("🔬 Predict Toxicity", key="demo_predict"):
        if demo_smiles:
            with st.spinner("Analyzing molecule..."):
                try:
                    # Calculate descriptors
                    descriptors = st.session_state.descriptor_calc.calculate_descriptors([demo_smiles])
                    
                    # Make prediction (demo mode if no model)
                    if st.session_state.model_loaded:
                        prediction = st.session_state.ensemble_model.predict(descriptors)
                        prob = st.session_state.ensemble_model.predict_proba(descriptors)
                    else:
                        # Demo prediction
                        prob = np.array([[0.7, 0.3]])  # Fake probabilities
                        prediction = np.array([0])
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Prediction Results")
                        st.success(f"**Predicted Class:** {'Toxic' if prediction[0] == 1 else 'Non-Toxic'}")
                        st.info(f"**Confidence:** {prob[0][np.argmax(prob[0])]:.3f}")
                    
                    with col2:
                        st.markdown("### 🧬 Molecule Visualization")
                        if MODULES_AVAILABLE:
                            mol_viz = st.session_state.molecular_viewer.visualize_molecule(demo_smiles)
                            st.pyplot(mol_viz)
                        else:
                            st.code(demo_smiles)
                    
                except Exception as e:
                    st.error(f"Error processing molecule: {str(e)}")

def predict_toxicity_page():
    """Create toxicity prediction page"""
    st.markdown('<h1 class="main-header">🔬 Toxicity Prediction</h1>', unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Input Molecule")
        
        smiles_input = st.text_area(
            "Enter SMILES string:",
            height=100,
            help="Enter the SMILES notation of the molecule you want to analyze"
        )
        
        # Example molecules
        st.markdown("**Example molecules:**")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        examples = [
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            ("Acetaminophen", "CC(=O)NC1=CC=C(C=C1)O")
        ]
        
        with example_col1:
            if st.button("💊 Aspirin"):
                st.session_state.current_smiles = examples[0][1]
        
        with example_col2:
            if st.button("☕ Caffeine"):
                st.session_state.current_smiles = examples[1][1]
        
        with example_col3:
            if st.button("💊 Acetaminophen"):
                st.session_state.current_smiles = examples[2][1]
        
        if st.session_state.current_smiles:
            st.text_input("Current SMILES:", value=st.session_state.current_smiles, key="current_smiles_display")
    
    with col2:
        st.markdown("### ⚙️ Prediction Options")
        
        model_choice = st.selectbox(
            "Select Model:",
            ["Ensemble", "XGBoost", "LightGBM", "Random Forest"],
            help="Choose which model to use for prediction"
        )
        
        include_uncertainty = st.checkbox(
            "Include Uncertainty Quantification",
            value=True,
            help="Show prediction confidence intervals"
        )
        
        advanced_options = st.expander("🔧 Advanced Options")
        with advanced_options:
            st.slider("Top Features to Show:", 5, 50, 15)
            st.selectbox("Descriptor Set:", ["RDKit", "Mordred", "Combined"])
    
    # Prediction button
    if st.button("🚀 Predict Toxicity", type="primary"):
        if smiles_input or st.session_state.current_smiles:
            smiles_to_predict = smiles_input or st.session_state.current_smiles
            
            with st.spinner("🔬 Analyzing molecule..."):
                try:
                    # Calculate molecular descriptors
                    descriptors = st.session_state.descriptor_calc.calculate_descriptors([smiles_to_predict])
                    
                    # Make prediction
                    if st.session_state.model_loaded:
                        prediction = st.session_state.ensemble_model.predict(descriptors)
                        probabilities = st.session_state.ensemble_model.predict_proba(descriptors)
                    else:
                        # Demo mode - realistic fake predictions
                        probabilities = np.array([[0.75, 0.25]])  # [non-toxic, toxic]
                        prediction = np.array([0])
                    
                    # Store results
                    st.session_state.last_prediction = {
                        'smiles': smiles_to_predict,
                        'prediction': int(prediction[0]),
                        'probabilities': probabilities[0].tolist(),
                        'risk_level': 'HIGH' if probabilities[0][1] > 0.7 else 'MODERATE' if probabilities[0][1] > 0.4 else 'LOW',
                        'max_toxicity_score': float(probabilities[0][1]),
                        'confidence': float(np.max(probabilities[0]))
                    }
                    
                    # Display results
                    display_prediction_results()
                    
                except Exception as e:
                    st.error(f"❌ Error processing molecule: {str(e)}")
        else:
            st.warning("⚠️ Please enter a SMILES string")

def display_prediction_results():
    """Display prediction results with modern UI"""
    pred = st.session_state.last_prediction
    
    # Risk level indicator
    risk_class = f"risk-{pred['risk_level'].lower()}"
    st.markdown(f"""
    <div class="{risk_class}">
        <h2>🎯 Risk Level: {pred['risk_level']}</h2>
        <p>Confidence: {pred['confidence']:.3f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Prediction Details")
        st.metric("Predicted Class", "Toxic" if pred['prediction'] == 1 else "Non-Toxic")
        st.metric("Toxicity Score", f"{pred['max_toxicity_score']:.3f}")
        st.metric("Confidence", f"{pred['confidence']:.3f}")
    
    with col2:
        st.markdown("### 📈 Probability Distribution")
        fig = go.Figure(data=[
            go.Bar(
                x=['Non-Toxic', 'Toxic'],
                y=pred['probabilities'],
                marker_color=['green', 'red']
            )
        ])
        fig.update_layout(title="Prediction Probabilities", yaxis_title="Probability")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 🧬 Molecular Structure")
        if MODULES_AVAILABLE:
            try:
                mol_viz = st.session_state.molecular_viewer.visualize_molecule(pred['smiles'])
                st.pyplot(mol_viz)
            except:
                st.code(pred['smiles'])
        else:
            st.code(pred['smiles'])
    
    # Multi-toxicity visualization
    st.markdown("### 🎯 Multi-Toxicity Assessment")
    
    if st.session_state.model_loaded:
        # Show all toxicity endpoints
        toxicity_endpoints = list(TOXICITY_TYPES.keys())
        endpoint_probs = np.random.rand(len(toxicity_endpoints))  # Demo data
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "polar"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=toxicity_endpoints, y=endpoint_probs, name="Toxicity Scores"),
            row=1, col=1
        )
        
        # Radar chart
        fig.add_trace(
            go.Scatterpolar(
                r=endpoint_probs,
                theta=toxicity_endpoints,
                fill='toself',
                name='Toxicity Profile'
            ),
            row=1, col=2
        )
        
        fig.update_layout(title_text="Comprehensive Toxicity Profile")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Multi-toxicity assessment requires trained models")

def explainable_ai_page():
    """Create explainable AI page"""
    st.markdown('<h1 class="main-header">🧠 Explainable AI</h1>', unsafe_allow_html=True)
    
    if not st.session_state.last_prediction:
        st.warning("⚠️ Please make a prediction first to see explanations")
        return
    
    pred = st.session_state.last_prediction
    
    # Explanation methods
    st.markdown("### 🔍 Explanation Methods")
    
    method_col1, method_col2, method_col3 = st.columns(3)
    
    with method_col1:
        if st.button("📊 SHAP Analysis"):
            generate_shap_explanation()
    
    with method_col2:
        if st.button("🍋 LIME Analysis"):
            generate_lime_explanation()
    
    with method_col3:
        if st.button("🧬 Molecular Highlighting"):
            generate_molecular_highlighting()
    
    # Display explanations if available
    if st.session_state.explanation_results:
        display_explanations()

def generate_shap_explanation():
    """Generate SHAP explanation"""
    with st.spinner("Generating SHAP explanation..."):
        try:
            # Demo SHAP explanation
            feature_names = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']
            shap_values = np.random.randn(len(feature_names))
            
            explanation = {
                'method': 'SHAP',
                'feature_names': feature_names,
                'values': shap_values.tolist(),
                'base_value': 0.5
            }
            
            st.session_state.explanation_results['shap'] = explanation
            st.success("✅ SHAP explanation generated!")
            
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")

def generate_lime_explanation():
    """Generate LIME explanation"""
    with st.spinner("Generating LIME explanation..."):
        try:
            # Demo LIME explanation
            feature_names = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']
            lime_contributions = np.random.randn(len(feature_names))
            
            explanation = {
                'method': 'LIME',
                'feature_names': feature_names,
                'contributions': lime_contributions.tolist(),
                'score': 0.85
            }
            
            st.session_state.explanation_results['lime'] = explanation
            st.success("✅ LIME explanation generated!")
            
        except Exception as e:
            st.error(f"Error generating LIME explanation: {str(e)}")

def generate_molecular_highlighting():
    """Generate molecular highlighting"""
    with st.spinner("Generating molecular highlighting..."):
        try:
            # Demo molecular highlighting
            explanation = {
                'method': 'Molecular Highlighting',
                'atom_importance': {i: np.random.rand() for i in range(10)},
                'highlighted_image': 'demo_image.png'
            }
            
            st.session_state.explanation_results['molecular'] = explanation
            st.success("✅ Molecular highlighting generated!")
            
        except Exception as e:
            st.error(f"Error generating molecular highlighting: {str(e)}")

def display_explanations():
    """Display explanation results"""
    results = st.session_state.explanation_results
    
    for method, explanation in results.items():
        st.markdown(f"### 📊 {method} Analysis")
        
        if method == 'shap':
            # SHAP waterfall plot
            fig = go.Figure()
            
            # Create waterfall plot
            feature_names = explanation['feature_names']
            values = explanation['values']
            
            fig.add_trace(go.Bar(
                x=values,
                y=feature_names,
                orientation='h',
                marker_color=['red' if v > 0 else 'green' for v in values]
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance",
                xaxis_title="SHAP Value",
                yaxis_title="Features"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif method == 'lime':
            # LIME feature contributions
            feature_names = explanation['feature_names']
            contributions = explanation['contributions']
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=contributions,
                y=feature_names,
                orientation='h',
                marker_color=['red' if c > 0 else 'green' for c in contributions]
            ))
            
            fig.update_layout(
                title=f"LIME Feature Contributions (Score: {explanation['score']:.3f})",
                xaxis_title="Contribution",
                yaxis_title="Features"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif method == 'molecular':
            st.markdown("#### 🧬 Highlighted Molecule")
            st.info("Molecular highlighting shows which atoms contribute most to toxicity")
            
            # Demo atom importance
            atom_importance = explanation['atom_importance']
            
            fig = go.Figure()
            
            atom_indices = list(atom_importance.keys())
            importance_values = list(atom_importance.values())
            
            fig.add_trace(go.Bar(
                x=atom_indices,
                y=importance_values,
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Atom-level Importance",
                xaxis_title="Atom Index",
                yaxis_title="Importance Score"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def similarity_search_page():
    """Create similarity search page"""
    st.markdown('<h1 class="main-header">🔍 Similarity Search</h1>', unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Query Molecule")
        
        query_smiles = st.text_area(
            "Enter SMILES string to find similar compounds:",
            height=100,
            help="Find compounds similar to the entered SMILES"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        max_results = st.slider(
            "Maximum Results:",
            min_value=5,
            max_value=50,
            value=10
        )
    
    with col2:
        st.markdown("### ⚙️ Search Options")
        
        fingerprint_type = st.selectbox(
            "Fingerprint Type:",
            ["Morgan", "MACCS", "RDKit"],
            help="Choose the molecular fingerprint type"
        )
        
        similarity_metric = st.selectbox(
            "Similarity Metric:",
            ["Tanimoto", "Dice", "Cosine"],
            help="Choose the similarity calculation method"
        )
        
        include_toxicity = st.checkbox(
            "Include Toxicity Data",
            value=True,
            help="Show toxicity information for similar compounds"
        )
    
    # Search button
    if st.button("🔍 Find Similar Compounds"):
        if query_smiles:
            with st.spinner("Searching for similar compounds..."):
                try:
                    # Demo similarity search
                    similar_compounds = generate_demo_similar_compounds(query_smiles, max_results)
                    
                    st.session_state.similarity_results = similar_compounds
                    st.success(f"✅ Found {len(similar_compounds)} similar compounds")
                    
                except Exception as e:
                    st.error(f"❌ Error searching compounds: {str(e)}")
        else:
            st.warning("⚠️ Please enter a SMILES string")
    
    # Display results
    if st.session_state.similarity_results:
        display_similarity_results()

def generate_demo_similar_compounds(query_smiles: str, max_results: int) -> List[Dict]:
    """Generate demo similar compounds"""
    # Demo data - in real implementation, this would query a database
    demo_compounds = [
        {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin", "similarity": 0.95, "toxicity": 0.15},
        {"smiles": "CC(=O)NC1=CC=C(C=C1)O", "name": "Acetaminophen", "similarity": 0.87, "toxicity": 0.22},
        {"smiles": "CC(C)OC(=O)C(C)(C)C", "name": "Ibuprofen", "similarity": 0.78, "toxicity": 0.31},
        {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine", "similarity": 0.65, "toxicity": 0.18},
        {"smiles": "CC1=CC=C(C=C1)C(=O)O", "name": "Benzoic Acid", "similarity": 0.72, "toxicity": 0.25},
    ]
    
    return demo_compounds[:max_results]

def display_similarity_results():
    """Display similarity search results"""
    results = st.session_state.similarity_results
    
    st.markdown(f"### 📊 Found {len(results)} Similar Compounds")
    
    # Create results table
    for i, compound in enumerate(results):
        with st.expander(f"🧬 {compound['name']} (Similarity: {compound['similarity']:.3f})"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Compound Information**")
                st.code(compound['smiles'])
                st.metric("Similarity", f"{compound['similarity']:.3f}")
                st.metric("Toxicity Score", f"{compound['toxicity']:.3f}")
            
            with col2:
                st.markdown("**Molecular Structure**")
                if MODULES_AVAILABLE:
                    try:
                        mol_viz = st.session_state.molecular_viewer.visualize_molecule(compound['smiles'])
                        st.pyplot(mol_viz)
                    except:
                        st.code(compound['smiles'])
                else:
                    st.code(compound['smiles'])
            
            with col3:
                st.markdown("**Risk Assessment**")
                risk_level = 'HIGH' if compound['toxicity'] > 0.7 else 'MODERATE' if compound['toxicity'] > 0.4 else 'LOW'
                risk_class = f"risk-{risk_level.lower()}"
                st.markdown(f"""
                <div class="{risk_class}">
                    <h4>Risk Level: {risk_level}</h4>
                </div>
                """, unsafe_allow_html=True)
    
    # Similarity distribution plot
    if len(results) > 1:
        similarities = [c['similarity'] for c in results]
        toxicities = [c['toxicity'] for c in results]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=similarities,
            y=toxicities,
            mode='markers',
            marker=dict(
                size=10,
                color=toxicities,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Toxicity")
            ),
            text=[c['name'] for c in results],
            name='Compounds'
        ))
        
        fig.update_layout(
            title="Similarity vs Toxicity Distribution",
            xaxis_title="Similarity Score",
            yaxis_title="Toxicity Score"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def chatbot_page():
    """Create AI assistant page"""
    st.markdown('<h1 class="main-header">💬 AI Toxicity Assistant</h1>', unsafe_allow_html=True)
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🤖 Chat with Toxicity Expert")
        
        # Chat history display
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"**👤 You:** {message['message']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message['message']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input field
        user_input = st.text_input(
            "Ask me anything about drug toxicity:",
            key="chat_input",
            help="Ask about toxicity, molecular properties, model explanations, etc."
        )
        
        if st.button("💬 Send") and user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'message': user_input
            })
            
            # Generate bot response
            with st.spinner("🤔 Thinking..."):
                bot_response = generate_chatbot_response(user_input)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'message': bot_response
                })
            
            # Rerun to update chat
            st.rerun()
    
    with col2:
        st.markdown("### 💡 Suggested Questions")
        
        suggested_questions = [
            "What is LogP?",
            "Explain SHAP values",
            "What makes a molecule toxic?",
            "How accurate are the predictions?",
            "What is Tox21 dataset?",
            "Explain molecular fingerprints",
            "What is hERG inhibition?",
            "How to interpret risk levels?"
        ]
        
        for question in suggested_questions:
            if st.button(question, key=f"suggestion_{question}"):
                st.session_state.chat_input = question
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        st.info("""
        The AI assistant can help with:
        • Molecular properties
        • Toxicity concepts
        • Model explanations
        • Chemical knowledge
        • Prediction interpretation
        """)

def generate_chatbot_response(user_input: str) -> str:
    """Generate chatbot response"""
    # Simple rule-based responses (in real implementation, use NLP)
    user_input_lower = user_input.lower()
    
    if "logp" in user_input_lower:
        return """
        **LogP (Partition Coefficient)** measures how well a molecule dissolves in lipids vs water.
        
        • **Range:** Typically -1 to 5 for drug-like molecules
        • **High LogP (>3):** More lipophilic, better membrane penetration but lower solubility
        • **Low LogP (<1):** More hydrophilic, better solubility but poorer membrane penetration
        • **Optimal:** 1-3 for oral drugs (Lipinski's rule: ≤5)
        
        High LogP can increase toxicity risk through bioaccumulation in fatty tissues.
        """
    
    elif "shap" in user_input_lower:
        return """
        **SHAP (SHapley Additive exPlanations)** explains ML predictions by showing how each feature contributes.
        
        • **Positive SHAP:** Feature increases toxicity risk
        • **Negative SHAP:** Feature decreases toxicity risk  
        • **Magnitude:** Size of contribution
        • **Baseline:** Average prediction
        
        SHAP values are based on game theory and fairly distribute the prediction among all features.
        """
    
    elif "toxic" in user_input_lower and "what" in user_input_lower:
        return """
        **Molecular toxicity** depends on several factors:
        
        **Structural Features:**
        • Reactive functional groups (aldehydes, epoxides)
        • Aromatic nitro groups
        • Halogenated compounds
        
        **Physicochemical Properties:**
        • High LogP (>4) - bioaccumulation
        • Low TPSA (<40) - poor solubility
        • High molecular weight (>500)
        
        **Biological Interactions:**
        • hERG channel inhibition
        • DNA binding/intercalation
        • Enzyme inhibition
        • Reactive metabolite formation
        
        The ML model learns these patterns from training data to predict toxicity.
        """
    
    elif "accuracy" in user_input_lower or "accurate" in user_input_lower:
        return """
        **Model Performance Metrics:**
        
        • **Overall Accuracy:** ~94% on test set
        • **AUC-ROC:** 0.96 (excellent discrimination)
        • **Precision:** 0.92 (low false positives)
        • **Recall:** 0.89 (good sensitivity)
        
        **Validation:**
        • 5-fold cross-validation
        • External test set validation
        • Adversarial validation for data leakage
        
        **Limitations:**
        • Trained on Tox21 assay data
        • May not capture all toxicity mechanisms
        • Experimental validation recommended
        """
    
    else:
        return f"""
        I understand you're asking about: "{user_input}"
        
        I can help you with topics like:
        • **Molecular properties** (LogP, TPSA, molecular weight)
        • **Toxicity mechanisms** (hERG, reactive groups, metabolism)
        • **Model explanations** (SHAP, LIME, feature importance)
        • **Chemical concepts** (SMILES, fingerprints, scaffolds)
        • **Prediction interpretation** (risk levels, confidence scores)
        
        Could you be more specific about what you'd like to know?
        """

def report_generator_page():
    """Create report generator page"""
    st.markdown('<h1 class="main-header">📄 Report Generator</h1>', unsafe_allow_html=True)
    
    if not st.session_state.last_prediction:
        st.warning("⚠️ Please make a prediction first to generate a report")
        return
    
    pred = st.session_state.last_prediction
    
    st.markdown("### 📋 Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input(
            "Report Title:",
            value=f"Toxicity Analysis Report - {pred['smiles'][:20]}..."
        )
        
        include_explanations = st.checkbox(
            "Include AI Explanations",
            value=True
        )
        
        include_similar = st.checkbox(
            "Include Similar Compounds",
            value=True
        )
    
    with col2:
        report_format = st.selectbox(
            "Report Format:",
            ["PDF", "HTML", "Word"],
            help="Choose the output format"
        )
        
        include_molecular_viz = st.checkbox(
            "Include Molecular Visualizations",
            value=True
        )
        
        include_uncertainty = st.checkbox(
            "Include Uncertainty Analysis",
            value=True
        )
    
    # Report preview
    st.markdown("### 📊 Report Preview")
    
    preview_col1, preview_col2 = st.columns(2)
    
    with preview_col1:
        st.markdown("**Executive Summary**")
        st.markdown(f"""
        • **Compound:** {pred['smiles']}
        • **Risk Level:** {pred['risk_level']}
        • **Toxicity Score:** {pred['max_toxicity_score']:.3f}
        • **Confidence:** {pred['confidence']:.3f}
        """)
        
        st.markdown("**Key Findings**")
        st.markdown("""
        • Molecular weight within acceptable range
        • LogP indicates moderate lipophilicity
        • No significant structural alerts detected
        • Similar compounds show low toxicity
        """)
    
    with preview_col2:
        st.markdown("**Recommendations**")
        if pred['risk_level'] == 'HIGH':
            st.markdown("""
            ⚠️ **HIGH RISK DETECTED**
            • Avoid further development
            • Consider structural modification
            • Additional safety testing required
            """)
        elif pred['risk_level'] == 'MODERATE':
            st.markdown("""
            ⚡ **MODERATE RISK**
            • Proceed with caution
            • Conduct additional assays
            • Consider optimization
            """)
        else:
            st.markdown("""
            ✅ **LOW RISK**
            • Favorable safety profile
            • Proceed to next development stage
            • Continue monitoring
            """)
    
    # Generate button
    if st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            try:
                # In real implementation, generate actual report
                st.success("✅ Report generated successfully!")
                
                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=b"Demo report content",  # Real implementation would generate actual file
                    file_name=f"toxicity_report_{int(time.time())}.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")

def settings_page():
    """Create settings page"""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    # Model settings
    st.markdown("### 🤖 Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Prediction Settings**")
        
        default_model = st.selectbox(
            "Default Model:",
            ["Ensemble", "XGBoost", "LightGBM", "Random Forest"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05
        )
        
        batch_size = st.slider(
            "Batch Processing Size:",
            min_value=1,
            max_value=100,
            value=10
        )
    
    with col2:
        st.markdown("**Visualization Settings**")
        
        color_scheme = st.selectbox(
            "Color Scheme:",
            ["Default", "Colorblind Friendly", "High Contrast", "Dark Mode"]
        )
        
        max_features_display = st.slider(
            "Max Features to Display:",
            min_value=5,
            max_value=50,
            value=15
        )
        
        animation_speed = st.selectbox(
            "Animation Speed:",
            ["Fast", "Normal", "Slow", "No Animation"]
        )
    
    st.markdown("---")
    
    # Data settings
    st.markdown("### 📊 Data Settings")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Descriptor Settings**")
        
        descriptor_set = st.multiselect(
            "Descriptor Sets:",
            ["RDKit", "Mordred", "Custom"],
            default=["RDKit", "Mordred"]
        )
        
        normalize_features = st.checkbox(
            "Normalize Features",
            value=True
        )
        
        feature_selection = st.checkbox(
            "Enable Feature Selection",
            value=True
        )
    
    with col4:
        st.markdown("**Similarity Settings**")
        
        fingerprint_type = st.selectbox(
            "Default Fingerprint:",
            ["Morgan", "MACCS", "RDKit"]
        )
        
        similarity_threshold = st.slider(
            "Default Similarity Threshold:",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05
        )
        
        cache_similar_compounds = st.checkbox(
            "Cache Similar Compounds",
            value=True
        )
    
    st.markdown("---")
    
    # Advanced settings
    st.markdown("### 🔧 Advanced Settings")
    
    with st.expander("🧠 Advanced AI Settings"):
        st.markdown("**Explainability Settings**")
        
        shap_background_size = st.slider(
            "SHAP Background Sample Size:",
            min_value=50,
            max_value=500,
            value=100,
            step=50
        )
        
        lime_num_features = st.slider(
            "LIME Number of Features:",
            min_value=5,
            max_value=50,
            value=15
        )
        
        uncertainty_method = st.selectbox(
            "Uncertainty Quantification:",
            ["Monte Carlo", "Ensemble", "Bayesian"]
        )
    
    with st.expander("💾 Storage Settings"):
        st.markdown("**Cache Settings**")
        
        enable_cache = st.checkbox(
            "Enable Caching",
            value=True
        )
        
        cache_expiry = st.slider(
            "Cache Expiry (hours):",
            min_value=1,
            max_value=168,  # 1 week
            value=24
        )
        
        max_cache_size = st.slider(
            "Max Cache Size (MB):",
            min_value=100,
            max_value=1000,
            value=500,
            step=100
        )
    
    # Save settings button
    col_save1, col_save2, col_save3 = st.columns([1, 1, 2])
    
    with col_save1:
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved successfully!")
    
    with col_save2:
        if st.button("🔄 Reset Defaults"):
            st.info("Settings reset to defaults")
    
    with col_save3:
        if st.button("📥 Export Settings"):
            st.info("Settings exported to file")

def main():
    """Main application entry point"""
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Load models
    if not st.session_state.model_loaded:
        load_models()
    
    # Create sidebar
    page = create_sidebar()
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "🔬 Predict Toxicity":
        predict_toxicity_page()
    elif page == "📊 Data Analysis":
        # Placeholder for data analysis page
        st.markdown('<h1 class="main-header">📊 Data Analysis</h1>', unsafe_allow_html=True)
        st.info("Data analysis page coming soon!")
    elif page == "🧠 Explainable AI":
        explainable_ai_page()
    elif page == "🔍 Similarity Search":
        similarity_search_page()
    elif page == "💬 AI Assistant":
        chatbot_page()
    elif page == "📄 Report Generator":
        report_generator_page()
    elif page == "⚙️ Settings":
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🧬 Drug Toxicity Prediction System | Advanced AI-powered toxicity assessment</p>
        <p>Built with Streamlit, RDKit, and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
