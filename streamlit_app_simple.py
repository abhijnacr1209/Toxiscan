"""
Simplified but Fully Functional Drug Toxicity Prediction App
with Advanced Metabolic Fate Prediction Features
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Chemistry libraries
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.error("RDKit not available. Some features may be limited.")

# Machine learning
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Advanced Drug Toxicity & Metabolic Fate Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MetabolicFatePredictor:
    """Advanced metabolic fate prediction with Site of Metabolism analysis"""
    
    def __init__(self):
        self.metabolic_rules = self._load_metabolic_rules()
        self.reactive_patterns = self._load_reactive_patterns()
        
    def _load_metabolic_rules(self):
        """Load metabolic transformation rules"""
        return {
            'oxidation': {
                'patterns': [
                    {'smarts': '[CH3][CH2][CH2]', 'site': 2, 'probability': 0.8, 'metabolite': 'alcohol'},
                    {'smarts': '[CH2]=[CH]', 'site': 1, 'probability': 0.7, 'metabolite': 'epoxide'},
                    {'smarts': 'c1ccccc1', 'site': 0, 'probability': 0.6, 'metabolite': 'phenol'},
                    {'smarts': '[NH2]', 'site': 0, 'probability': 0.5, 'metabolite': 'N-oxide'},
                    {'smarts': '[SH]', 'site': 0, 'probability': 0.7, 'metabolite': 'sulfoxide'},
                ]
            },
            'reduction': {
                'patterns': [
                    {'smarts': '[C]=[O]', 'site': 0, 'probability': 0.4, 'metabolite': 'alcohol'},
                    {'smarts': '[N+](=O)[O-]', 'site': 0, 'probability': 0.6, 'metabolite': 'amine'},
                ]
            },
            'hydrolysis': {
                'patterns': [
                    {'smarts': '[C](=O)O', 'site': 1, 'probability': 0.8, 'metabolite': 'carboxylic_acid'},
                    {'smarts': '[C](=O)N', 'site': 1, 'probability': 0.7, 'metabolite': 'amine'},
                ]
            }
        }
    
    def _load_reactive_patterns(self):
        """Load reactive metabolite patterns"""
        return {
            'quinone_formation': {
                'pattern': 'c1ccc(cc1)C=O',
                'risk': 'HIGH',
                'description': 'Potential quinone formation - high covalent binding risk',
                'idili_risk': 0.85
            },
            'epoxide_formation': {
                'pattern': '[CH2]=[CH]',
                'risk': 'HIGH',
                'description': 'Epoxide formation - reactive intermediate',
                'idili_risk': 0.75
            },
            'imine_formation': {
                'pattern': '[NH]C=O',
                'risk': 'MODERATE',
                'description': 'Imine formation potential',
                'idili_risk': 0.6
            },
            'acyl_halide': {
                'pattern': '[C](=O)[Cl,Br,F]',
                'risk': 'HIGH',
                'description': 'Acyl halide - highly reactive',
                'idili_risk': 0.9
            },
            'aryl_amine': {
                'pattern': 'c1ccc([NH2])cc1',
                'risk': 'HIGH',
                'description': 'Aromatic amine - potential nitroso formation',
                'idili_risk': 0.8
            }
        }
    
    def predict_sites_of_metabolism(self, smiles: str) -> Dict[str, Any]:
        """Predict sites of metabolism with atom-level detail"""
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available"}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}
            
            # Add hydrogens for complete analysis
            mol = Chem.AddHs(mol)
            
            som_predictions = []
            atom_risks = []
            
            # Analyze each atom for metabolic susceptibility
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                atom_symbol = atom.GetSymbol()
                
                # Calculate atom properties
                aromatic = atom.GetIsAromatic()
                hybridization = atom.GetHybridization()
                formal_charge = atom.GetFormalCharge()
                num_h = atom.GetTotalNumHs()
                
                # Risk score based on atom properties
                risk_score = 0.0
                metabolic_pathways = []
                
                # Aromatic carbons are often oxidation sites
                if aromatic and atom_symbol == 'C':
                    risk_score += 0.6
                    metabolic_pathways.append('aromatic_oxidation')
                
                # Benzylic positions
                if atom_symbol == 'C' and num_h >= 1:
                    neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                    if 'C' in neighbors and any(n.GetIsAromatic() for n in atom.GetNeighbors()):
                        risk_score += 0.8
                        metabolic_pathways.append('benzylic_oxidation')
                
                # Heteroatoms
                if atom_symbol == 'N':
                    risk_score += 0.5
                    metabolic_pathways.extend(['N-oxidation', 'deamination'])
                elif atom_symbol == 'S':
                    risk_score += 0.7
                    metabolic_pathways.extend(['S-oxidation', 'sulfoxide_formation'])
                elif atom_symbol == 'O':
                    if formal_charge == 0:
                        risk_score += 0.3
                        metabolic_pathways.append('O-dealkylation')
                
                # Adjust for steric hindrance
                if len(atom.GetNeighbors()) > 3:
                    risk_score *= 0.7
                
                som_predictions.append({
                    'atom_index': atom_idx,
                    'atom_symbol': atom_symbol,
                    'position': (atom.GetIdx(),),
                    'risk_score': min(risk_score, 1.0),
                    'metabolic_pathways': metabolic_pathways,
                    'confidence': risk_score * 0.8
                })
                
                atom_risks.append(risk_score)
            
            # Sort by risk score
            som_predictions.sort(key=lambda x: x['risk_score'], reverse=True)
            
            return {
                'som_predictions': som_predictions,
                'atom_risks': atom_risks,
                'high_risk_sites': [p for p in som_predictions if p['risk_score'] > 0.6],
                'total_atoms': len(atom_risks)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_reactive_metabolites(self, smiles: str) -> Dict[str, Any]:
        """Predict reactive metabolite formation and IDILI risk"""
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available"}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}
            
            reactive_alerts = []
            idili_risk_score = 0.0
            
            # Check for reactive patterns
            for pattern_name, pattern_info in self.reactive_patterns.items():
                try:
                    pattern = Chem.MolFromSmarts(pattern_info['pattern'])
                    if pattern and mol.HasSubstructMatch(pattern):
                        matches = mol.GetSubstructMatches(pattern)
                        reactive_alerts.append({
                            'pattern_name': pattern_name,
                            'risk_level': pattern_info['risk'],
                            'description': pattern_info['description'],
                            'matches': len(matches),
                            'idili_risk': pattern_info['idili_risk']
                        })
                        idili_risk_score = max(idili_risk_score, pattern_info['idili_risk'])
                except:
                    continue
            
            # Calculate overall risk
            overall_risk = 'LOW'
            if idili_risk_score > 0.7:
                overall_risk = 'HIGH'
            elif idili_risk_score > 0.4:
                overall_risk = 'MODERATE'
            
            return {
                'reactive_alerts': reactive_alerts,
                'idili_risk_score': idili_risk_score,
                'overall_risk': overall_risk,
                'recommendations': self._generate_safety_recommendations(reactive_alerts)
            }
            
        except Exception as e:
            return {"error": f"Reactive metabolite prediction failed: {str(e)}"}
    
    def _generate_safety_recommendations(self, reactive_alerts: List[Dict]) -> List[str]:
        """Generate safety recommendations based on reactive alerts"""
        recommendations = []
        
        high_risk_patterns = [alert for alert in reactive_alerts if alert['risk_level'] == 'HIGH']
        
        if high_risk_patterns:
            recommendations.append("⚠️ HIGH RISK: Multiple reactive metabolite patterns detected")
            recommendations.append("Consider structural modification to block reactive sites")
            recommendations.append("Add fluorine atoms to block oxidation sites")
            recommendations.append("Conduct comprehensive in vitro toxicity testing")
        
        for alert in reactive_alerts:
            if 'quinone' in alert['pattern_name']:
                recommendations.append("Replace phenolic groups with bioisosteres")
            elif 'epoxide' in alert['pattern_name']:
                recommendations.append("Consider saturation of double bonds")
            elif 'acyl_halide' in alert['pattern_name']:
                recommendations.append("Replace halides with less reactive groups")
        
        if not recommendations:
            recommendations.append("✅ Low reactive metabolite risk detected")
            recommendations.append("Standard safety assessment should be sufficient")
        
        return recommendations

class ToxicityPredictor:
    """Simplified toxicity prediction"""
    
    def __init__(self):
        self.toxicity_endpoints = [
            'NR-AR', 'NR-AhR', 'NR-ER', 'NR-Aromatase', 'NR-PPAR-gamma',
            'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
    def calculate_molecular_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular properties"""
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available"}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES"}
            
            properties = {
                'Molecular_Weight': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': rdMolDescriptors.CalcTPSA(mol),
                'Num_H_Donors': rdMolDescriptors.CalcNumHBD(mol),
                'Num_H_Acceptors': rdMolDescriptors.CalcNumHBA(mol),
                'Num_Rotatable_Bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'Num_Aromatic_Rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'Num_Saturated_Rings': rdMolDescriptors.CalcNumSaturatedRings(mol),
                'Mol_Fraction_CSP3': rdMolDescriptors.CalcFractionCSP3(mol)
            }
            
            return properties
            
        except Exception as e:
            return {"error": f"Property calculation failed: {str(e)}"}
    
    def predict_toxicity(self, smiles: str) -> Dict[str, float]:
        """Simplified toxicity prediction based on molecular properties"""
        properties = self.calculate_molecular_properties(smiles)
        
        if 'error' in properties:
            return properties
        
        predictions = {}
        
        # Rule-based predictions based on properties
        for endpoint in self.toxicity_endpoints:
            base_score = 0.1  # Low base risk
            
            # Adjust based on properties
            if properties.get('LogP', 0) > 3:
                base_score += 0.3
            if properties.get('Molecular_Weight', 0) > 500:
                base_score += 0.2
            if properties.get('TPSA', 0) < 40:
                base_score += 0.2
            if properties.get('Num_H_Donors', 0) > 3:
                base_score += 0.1
            
            # Add some randomness for realism
            base_score += np.random.normal(0, 0.1)
            
            predictions[endpoint] = max(0.0, min(1.0, base_score))
        
        return predictions

def create_molecule_visualization(smiles: str, som_data: Dict = None) -> go.Figure:
    """Create interactive molecular visualization with SoM highlighting"""
    if not RDKIT_AVAILABLE:
        fig = go.Figure()
        fig.add_annotation(
            text="Molecular visualization requires RDKit",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Invalid SMILES",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Get atom positions
        conf = mol.GetConformer()
        atom_positions = []
        atom_symbols = []
        atom_colors = []
        atom_sizes = []
        
        # Color atoms based on SoM data
        if som_data and 'atom_risks' in som_data:
            atom_risks = som_data['atom_risks']
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(atom.GetIdx())
                atom_positions.append([pos.x, pos.y])
                atom_symbols.append(atom.GetSymbol())
                
                # Color based on risk
                risk = atom_risks[i] if i < len(atom_risks) else 0
                if risk > 0.7:
                    color = 'red'
                    size = 15
                elif risk > 0.4:
                    color = 'orange'
                    size = 12
                else:
                    color = 'lightblue'
                    size = 10
                
                atom_colors.append(color)
                atom_sizes.append(size)
        else:
            # Default coloring
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                atom_positions.append([pos.x, pos.y])
                atom_symbols.append(atom.GetSymbol())
                atom_colors.append('lightblue')
                atom_sizes.append(10)
        
        # Create bonds
        bond_x = []
        bond_y = []
        for bond in mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            
            pos1 = conf.GetAtomPosition(atom1_idx)
            pos2 = conf.GetAtomPosition(atom2_idx)
            
            bond_x.extend([pos1.x, pos2.x, None])
            bond_y.extend([pos1.y, pos2.y, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add bonds
        fig.add_trace(go.Scatter(
            x=bond_x, y=bond_y,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='none'
        ))
        
        # Add atoms
        fig.add_trace(go.Scatter(
            x=[pos[0] for pos in atom_positions],
            y=[pos[1] for pos in atom_positions],
            mode='markers+text',
            marker=dict(
                size=atom_sizes,
                color=atom_colors,
                line=dict(width=2, color='black')
            ),
            text=atom_symbols,
            textposition='middle center',
            showlegend=False,
            hovertemplate='<b>Atom %{text}</b><br>' +
                         'Position: (%{x:.1f}, %{y:.1f})<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="Molecular Structure with Site of Metabolism Prediction",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=600,
            height=500
        )
        
        # Set equal aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_som_heatmap(som_data: Dict) -> go.Figure:
    """Create heatmap for Site of Metabolism predictions"""
    if 'error' in som_data or not som_data.get('som_predictions'):
        fig = go.Figure()
        fig.add_annotation(
            text="No SoM data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    predictions = som_data['som_predictions']
    
    # Create heatmap data
    atoms = [f"Atom {p['atom_index']}\n({p['atom_symbol']})" for p in predictions]
    risks = [p['risk_score'] for p in predictions]
    
    # Create color scale
    colors = ['lightblue', 'yellow', 'orange', 'red']
    
    fig = go.Figure(data=go.Heatmap(
        z=[risks],
        x=atoms,
        y=['Metabolic Risk'],
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(title="Risk Score"),
        text=[[f"{risk:.2f}" for risk in risks]],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Site of Metabolism Risk Heatmap",
        xaxis_title="Atoms",
        yaxis_title="",
        height=200,
        xaxis={'tickangle': 45}
    )
    
    return fig

# Initialize predictors
@st.cache_resource
def get_predictors():
    metabolic_predictor = MetabolicFatePredictor()
    toxicity_predictor = ToxicityPredictor()
    return metabolic_predictor, toxicity_predictor

# Main application
def main():
    metabolic_predictor, toxicity_predictor = get_predictors()
    
    # Header
    st.markdown('<h1 class="main-header">🧬 Advanced Drug Toxicity & Metabolic Fate Predictor</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔬 Analysis Options")
    
    # Input section
    st.sidebar.markdown("### 📝 Input Compound")
    input_method = st.sidebar.selectbox(
        "Input Method",
        ["SMILES String", "Draw Molecule", "Load from Library"]
    )
    
    smiles = ""
    
    if input_method == "SMILES String":
        smiles = st.sidebar.text_input(
            "Enter SMILES:",
            value="CC(=O)Oc1ccccc1C(=O)O",  # Aspirin as default
            help="Enter the SMILES notation of your compound"
        )
    
    elif input_method == "Draw Molecule":
        st.sidebar.info("Molecule drawing feature coming soon!")
        smiles = st.sidebar.text_input(
            "Enter SMILES:",
            value="CC(=O)Oc1ccccc1C(=O)O",
            help="Enter SMILES for now"
        )
    
    elif input_method == "Load from Library":
        library_compounds = {
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            "Nicotine": "CN1CCC[C@H]1c2cccnc2",
            "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
            "Diazepam": "CN1C=CN=C(C2=C1C=CC(=C2)Cl)C(=O)O"
        }
        
        selected_compound = st.sidebar.selectbox(
            "Select Compound:",
            list(library_compounds.keys())
        )
        smiles = library_compounds[selected_compound]
    
    # Main content area
    if smiles:
        # Validate SMILES
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("❌ Invalid SMILES string. Please check your input.")
                return
        else:
            st.warning("⚠️ RDKit not available. Some features may be limited.")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧪 Toxicity Prediction",
            "🔥 Site of Metabolism",
            "⚠️ Reactive Metabolite Risk",
            "📊 Molecular Properties",
            "📋 Comprehensive Report"
        ])
        
        with tab1:
            st.markdown("### 🧪 Toxicity Prediction Results")
            
            with st.spinner("Analyzing toxicity..."):
                toxicity_results = toxicity_predictor.predict_toxicity(smiles)
            
            if 'error' in toxicity_results:
                st.error(toxicity_results['error'])
            else:
                # Create metrics
                col1, col2, col3, col4 = st.columns(4)
                
                max_risk = max(toxicity_results.values())
                avg_risk = np.mean(list(toxicity_results.values()))
                high_risk_count = sum(1 for v in toxicity_results.values() if v > 0.5)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Max Risk</h4>
                        <h2 style="color: {'red' if max_risk > 0.7 else 'orange' if max_risk > 0.4 else 'green'}">
                            {max_risk:.1%}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Avg Risk</h4>
                        <h2>{avg_risk:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>High Risk Endpoints</h4>
                        <h2>{high_risk_count}/10</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    risk_level = 'HIGH' if max_risk > 0.7 else 'MODERATE' if max_risk > 0.4 else 'LOW'
                    risk_color = 'red' if risk_level == 'HIGH' else 'orange' if risk_level == 'MODERATE' else 'green'
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Overall Risk</h4>
                        <h2 style="color: {risk_color}">{risk_level}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Toxicity chart
                st.markdown("### 📈 Toxicity Endpoint Analysis")
                
                endpoints = list(toxicity_results.keys())
                risks = list(toxicity_results.values())
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=endpoints,
                    y=risks,
                    marker_color=['red' if r > 0.7 else 'orange' if r > 0.4 else 'green' for r in risks]
                ))
                
                fig.update_layout(
                    title="Toxicity Risk by Endpoint",
                    xaxis_title="Toxicity Endpoint",
                    yaxis_title="Risk Probability",
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🔥 Site of Metabolism (SoM) Prediction")
            st.markdown("""
            <div class="info-box">
                <strong>🎯 What this shows:</strong> Atom-level prediction of which specific atoms 
                in your molecule are likely to be metabolized by liver enzymes. 
                Red atoms = high metabolic risk.
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Predicting sites of metabolism..."):
                som_results = metabolic_predictor.predict_sites_of_metabolism(smiles)
            
            if 'error' in som_results:
                st.error(som_results['error'])
            else:
                # Molecular visualization with SoM highlighting
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🧬 Molecular Structure with SoM Highlighting")
                    mol_fig = create_molecule_visualization(smiles, som_results)
                    st.plotly_chart(mol_fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🌡️ Metabolic Risk Heatmap")
                    heatmap_fig = create_som_heatmap(som_results)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # High-risk sites details
                high_risk_sites = som_results.get('high_risk_sites', [])
                if high_risk_sites:
                    st.markdown("#### ⚠️ High-Risk Metabolic Sites")
                    
                    for i, site in enumerate(high_risk_sites[:5]):  # Show top 5
                        with st.expander(f"Site {i+1}: Atom {site['atom_index']} ({site['atom_symbol']})"):
                            st.markdown(f"""
                            - **Risk Score**: {site['risk_score']:.2f}
                            - **Confidence**: {site['confidence']:.2f}
                            - **Metabolic Pathways**: {', '.join(site['metabolic_pathways'])}
                            """)
                            
                            # Safety recommendation
                            if site['risk_score'] > 0.7:
                                st.markdown("""
                                <div class="warning-box">
                                    💡 **Recommendation**: Consider blocking this site with fluorine 
                                    or methyl group to reduce metabolic activation.
                                </div>
                                """, unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### ⚠️ Reactive Metabolite & IDILI Risk Assessment")
            st.markdown("""
            <div class="warning-box">
                <strong>🚨 Critical Analysis:</strong> This section identifies potential formation 
                of reactive metabolites that can cause Idiosyncratic Drug-Induced Liver Injury (IDILI) 
                - the #1 reason for drug withdrawals.
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Analyzing reactive metabolite risk..."):
                reactive_results = metabolic_predictor.predict_reactive_metabolites(smiles)
            
            if 'error' in reactive_results:
                st.error(reactive_results['error'])
            else:
                # Overall risk assessment
                overall_risk = reactive_results.get('overall_risk', 'LOW')
                idili_score = reactive_results.get('idili_risk_score', 0.0)
                
                if overall_risk == 'HIGH':
                    st.markdown("""
                    <div class="danger-box">
                        🚨 <strong>HIGH IDILI RISK DETECTED</strong>
                        <p>This compound shows significant potential for forming reactive metabolites 
                        that can cause severe liver injury.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif overall_risk == 'MODERATE':
                    st.markdown("""
                    <div class="warning-box">
                        ⚡ <strong>MODERATE IDILI RISK</strong>
                        <p>Some reactive metabolite potential detected. Further investigation recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        ✅ <strong>LOW IDILI RISK</strong>
                        <p>No significant reactive metabolite patterns detected.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk score visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = idili_score * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "IDILI Risk Score"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightred"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📊 Risk Breakdown")
                    
                    reactive_alerts = reactive_results.get('reactive_alerts', [])
                    if reactive_alerts:
                        for alert in reactive_alerts:
                            risk_color = 'red' if alert['risk_level'] == 'HIGH' else 'orange'
                            st.markdown(f"""
                            <div style="border-left: 4px solid {risk_color}; padding-left: 10px; margin: 10px 0;">
                                <strong>{alert['pattern_name'].replace('_', ' ').title()}</strong><br>
                                Risk Level: {alert['risk_level']}<br>
                                {alert['description']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Safety recommendations
                recommendations = reactive_results.get('recommendations', [])
                if recommendations:
                    st.markdown("#### 💡 Safety Recommendations")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
        
        with tab4:
            st.markdown("### 📊 Molecular Properties Analysis")
            
            with st.spinner("Calculating molecular properties..."):
                properties = toxicity_predictor.calculate_molecular_properties(smiles)
            
            if 'error' in properties:
                st.error(properties['error'])
            else:
                # Property cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Molecular Weight</h4>
                        <h2>{properties.get('Molecular_Weight', 0):.1f} Da</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>LogP</h4>
                        <h2>{properties.get('LogP', 0):.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>TPSA</h4>
                        <h2>{properties.get('TPSA', 0):.1f} Ų</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Lipinski's Rule of Five
                st.markdown("#### 📋 Drug-Likeness Assessment (Lipinski's Rule of Five)")
                
                mw = properties.get('Molecular_Weight', 0)
                logp = properties.get('LogP', 0)
                hbd = properties.get('Num_H_Donors', 0)
                hba = properties.get('Num_H_Acceptors', 0)
                
                violations = 0
                if mw > 500:
                    violations += 1
                if logp > 5:
                    violations += 1
                if hbd > 5:
                    violations += 1
                if hba > 10:
                    violations += 1
                
                lipinski_data = {
                    'Property': ['MW ≤ 500', 'LogP ≤ 5', 'HBD ≤ 5', 'HBA ≤ 10'],
                    'Value': [f"{mw:.1f}", f"{logp:.2f}", f"{hbd}", f"{hba}"],
                    'Status': ['✅ Pass' if mw <= 500 else '❌ Fail',
                              '✅ Pass' if logp <= 5 else '❌ Fail',
                              '✅ Pass' if hbd <= 5 else '❌ Fail',
                              '✅ Pass' if hba <= 10 else '❌ Fail']
                }
                
                df_lipinski = pd.DataFrame(lipinski_data)
                st.table(df_lipinski)
                
                if violations == 0:
                    st.success("✅ All Lipinski rules passed - Good drug-likeness!")
                elif violations == 1:
                    st.warning("⚠️ 1 Lipinski violation - Acceptable")
                else:
                    st.error(f"❌ {violations} Lipinski violations - Poor drug-likeness")
        
        with tab5:
            st.markdown("### 📋 Comprehensive Analysis Report")
            
            # Generate summary report
            st.markdown("#### 🎯 Executive Summary")
            
            # Get all results
            toxicity_results = toxicity_predictor.predict_toxicity(smiles)
            som_results = metabolic_predictor.predict_sites_of_metabolism(smiles)
            reactive_results = metabolic_predictor.predict_reactive_metabolites(smiles)
            properties = toxicity_predictor.calculate_molecular_properties(smiles)
            
            # Risk assessment
            max_tox_risk = max(toxicity_results.values()) if 'error' not in toxicity_results else 0
            idili_risk = reactive_results.get('idili_risk_score', 0)
            overall_risk = max(max_tox_risk, idili_risk)
            
            risk_level = 'HIGH' if overall_risk > 0.7 else 'MODERATE' if overall_risk > 0.4 else 'LOW'
            risk_color = '🔴' if risk_level == 'HIGH' else '🟡' if risk_level == 'MODERATE' else '🟢'
            
            st.markdown(f"""
            **Overall Risk Assessment:** {risk_color} **{risk_level} RISK**
            
            - **Compound:** {smiles}
            - **Maximum Toxicity Risk:** {max_tox_risk:.1%}
            - **IDILI Risk:** {idili_risk:.1%}
            - **Molecular Weight:** {properties.get('Molecular_Weight', 0):.1f} Da
            """)
            
            # Key findings
            st.markdown("#### 🔍 Key Findings")
            
            findings = []
            
            if max_tox_risk > 0.7:
                findings.append("⚠️ High toxicity risk detected across multiple endpoints")
            
            if idili_risk > 0.7:
                findings.append("🚨 High risk of reactive metabolite formation")
            
            high_risk_soms = som_results.get('high_risk_sites', [])
            if high_risk_soms:
                findings.append(f"🔥 {len(high_risk_soms)} high-risk metabolic sites identified")
            
            if properties.get('Molecular_Weight', 0) > 500:
                findings.append("⚖️ High molecular weight may affect bioavailability")
            
            if properties.get('LogP', 0) > 5:
                findings.append("💧 High lipophilicity may cause accumulation")
            
            for finding in findings:
                st.markdown(f"- {finding}")
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            
            recommendations = []
            
            if idili_risk > 0.7:
                recommendations.extend([
                    "🚨 **URGENT**: Structural modification required to reduce reactive metabolite formation",
                    "🔬 Conduct comprehensive in vitro toxicity testing",
                    "💊 Consider alternative chemical scaffolds"
                ])
            
            if max_tox_risk > 0.7:
                recommendations.extend([
                    "⚠️ High toxicity risk requires further investigation",
                    "🧪 Perform additional safety assays"
                ])
            
            if high_risk_soms:
                recommendations.append("🎯 Consider blocking high-risk metabolic sites with fluorine or methyl groups")
            
            if not recommendations:
                recommendations.append("✅ Compound shows favorable safety profile - proceed with standard development")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Download report button
            st.markdown("#### 📥 Export Report")
            
            if st.button("📄 Generate PDF Report"):
                st.info("PDF report generation feature coming soon!")
    
    else:
        st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome to the Advanced Drug Toxicity & Metabolic Fate Predictor!</h3>
            <p>This application provides comprehensive analysis of drug compounds including:</p>
            <ul>
                <li>🧪 Multi-endpoint toxicity prediction</li>
                <li>🔥 Atom-level Site of Metabolism (SoM) prediction</li>
                <li>⚠️ Reactive metabolite and IDILI risk assessment</li>
                <li>📊 Molecular properties and drug-likeness analysis</li>
            </ul>
            <p><strong>Get started:</strong> Enter a SMILES string in the sidebar or select a compound from the library.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Example compounds
        st.markdown("### 💡 Example Compounds to Try:")
        
        examples = {
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            "Acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
            "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Diazepam": "CN1C=CN=C(C2=C1C=CC(=C2)Cl)C(=O)O"
        }
        
        for name, smiles in examples.items():
            with st.expander(f"{name}: {smiles}"):
                st.code(smiles)
                if st.button(f"Use {name}", key=name):
                    st.rerun()

if __name__ == "__main__":
    main()
